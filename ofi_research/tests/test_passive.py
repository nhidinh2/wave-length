"""Phase-0 passive markout: causality and arithmetic.

The screen's whole value is that a negative reading lets us NOT buy MBO data,
so the failure mode that matters is a bug making passive quoting look better
than it is. These tests pin the sign conventions and the causality rules on
hand-built books where the right answer is known by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ofi_research.config import Config
from ofi_research import passive


def _book(n=12, mid0=100.0, spread=0.02, stv=None, ofi=None, drift=0.0):
    """A single clean segment with a constant spread and controllable flow."""
    mid = mid0 + drift * np.arange(n)
    df = pd.DataFrame({
        "segment_id": 1,
        "session_date": "2026-01-02",
        "timestamp": pd.date_range("2026-01-02 14:30", periods=n, freq="200ms"),
        "midprice": mid,
        "spread": spread,
        "bid_price_1": mid - spread / 2,
        "ask_price_1": mid + spread / 2,
        "signed_trade_volume": (np.zeros(n) if stv is None else stv),
        "OFI_L1_ref": (np.zeros(n) if ofi is None else ofi),
        "in_continuous_session": True,
        "integrity_break": False,
    })
    return df


def _with_target(df, tag="ms1000", horizon_rows=1):
    """future_mid_change = mid shifted forward by `horizon_rows`."""
    df = df.copy()
    df[f"future_mid_change_{tag}"] = (
        df["midprice"].shift(-horizon_rows) - df["midprice"])
    return df


def test_sell_aggressor_fills_us_long_at_the_bid():
    stv = np.zeros(10)
    stv[5] = -300.0          # net SELL aggressor -> hits our resting bid
    f = passive.build_passive_fills(_book(10, stv=stv), Config())
    assert len(f) == 1
    r = f.iloc[0]
    assert r["side"] == 1                       # we are long
    assert r["fill_price"] == pytest.approx(r["prev_bid"])


def test_buy_aggressor_fills_us_short_at_the_ask():
    stv = np.zeros(10)
    stv[5] = 300.0           # net BUY aggressor -> lifts our resting ask
    f = passive.build_passive_fills(_book(10, stv=stv), Config())
    assert len(f) == 1
    r = f.iloc[0]
    assert r["side"] == -1                      # we are short
    assert r["fill_price"] == pytest.approx(r["prev_ask"])


def test_fill_price_is_the_quote_standing_BEFORE_the_trade():
    """Pricing a fill at the same bar's book would use a post-trade quote."""
    stv = np.zeros(8)
    stv[4] = -100.0
    df = _book(8, stv=stv, drift=0.01)          # book moves every row
    f = passive.build_passive_fills(df, Config())
    r = f.iloc[0]
    assert r["fill_price"] == pytest.approx(df["bid_price_1"].iloc[3])
    assert r["fill_price"] != pytest.approx(df["bid_price_1"].iloc[4])


def test_conditioning_ofi_is_strictly_pre_fill():
    """Using the trade's own bar to condition would let it predict itself."""
    stv = np.zeros(8)
    stv[4] = -100.0
    ofi = np.arange(8, dtype="float64") * 10.0
    f = passive.build_passive_fills(_book(8, stv=stv, ofi=ofi), Config())
    assert f.iloc[0]["prev_ofi"] == pytest.approx(ofi[3])


def test_flat_market_markout_equals_half_spread_plus_rebate():
    """No drift: the maker keeps exactly the half-spread, plus any rebate."""
    cfg = Config()
    cfg.costs.tick_size = 0.01
    cfg.costs.maker_rebate_per_unit = 0.002
    stv = np.zeros(8)
    stv[4] = -100.0
    df = _with_target(_book(8, spread=0.02, stv=stv))
    f = passive.markout_columns(
        passive.build_passive_fills(df, cfg), cfg, ["ms1000"])
    # half-spread 0.01 + rebate 0.002
    assert f.iloc[0]["markout_ms1000"] == pytest.approx(0.012)


def test_adverse_move_after_fill_destroys_the_edge():
    """Bought at the bid, then the mid falls: markout must go negative."""
    cfg = Config()
    stv = np.zeros(8)
    stv[4] = -100.0
    df = _book(8, spread=0.02, stv=stv)
    df["midprice"] = 100.0
    df.loc[5:, "midprice"] = 99.90              # 10 cents against us
    df["bid_price_1"] = df["midprice"] - 0.01
    df["ask_price_1"] = df["midprice"] + 0.01
    df = _with_target(df)
    f = passive.markout_columns(
        passive.build_passive_fills(df, cfg), cfg, ["ms1000"])
    assert f.iloc[0]["markout_ms1000"] < 0


def test_crossing_out_costs_another_half_spread():
    cfg = Config()
    stv = np.zeros(8)
    stv[4] = -100.0
    df = _with_target(_book(8, spread=0.02, stv=stv))
    f = passive.markout_columns(
        passive.build_passive_fills(df, cfg), cfg, ["ms1000"])
    r = f.iloc[0]
    assert (r["markout_ms1000"] - r["markout_ms1000_cross_out"]
            == pytest.approx(0.01))


def test_no_fill_inferred_when_no_trade_occurs():
    assert passive.build_passive_fills(_book(10), Config()).empty


def test_fills_never_cross_a_segment_boundary():
    """The first row of a segment has no prior quote to have rested at."""
    stv = np.zeros(6)
    stv[0] = -100.0                             # trade on the very first row
    df = _book(6, stv=stv)
    df["segment_id"] = [1, 1, 1, 2, 2, 2]
    f = passive.build_passive_fills(df, Config())
    assert f.empty or (f["segment_id"] != 1).all() or len(f) == 0


def _toxic_flow(n=4000, seed=0):
    """Flow where imbalance drives BOTH the aggressor side and the next move.

    This is the mechanism the screen is trying to detect, so the fixture has to
    contain it: the aggressor arrives on the side the imbalance implies, and
    the mid then continues in that direction. A fixture that randomizes the
    aggressor side independently of OFI severs the link and would make even a
    correct implementation look signal-free.
    """
    rng = np.random.default_rng(seed)
    ofi = rng.normal(size=n) * 100.0
    step = np.concatenate([[0.0], 1e-4 * ofi[:-1]]) + rng.normal(scale=1e-5,
                                                                 size=n)
    mid = 100.0 + np.cumsum(step)
    # Negative OFI -> sell aggressors hit the bid; positive -> buys lift the ask
    stv = np.sign(ofi) * 100.0
    return pd.DataFrame({
        "segment_id": 1, "session_date": np.repeat(
            [f"2026-01-{d:02d}" for d in range(1, 11)], n // 10),
        "midprice": mid, "spread": 0.02,
        "bid_price_1": mid - 0.01, "ask_price_1": mid + 0.01,
        "signed_trade_volume": stv, "OFI_L1_ref": ofi,
        "in_continuous_session": True, "integrity_break": False,
    })


def test_fill_aligned_ofi_orders_toxic_fills_below_benign_ones():
    cfg = Config()
    df = _with_target(_toxic_flow())
    fills = passive.markout_columns(
        passive.build_passive_fills(df, cfg), cfg, ["ms1000"])
    tab = passive.markout_by_ofi_decile(fills, cfg, ["ms1000"],
                                        bucket_on="fill_aligned_ofi")
    assert not tab.empty
    mono = float(tab["decile_monotonicity_spearman"].iloc[0])
    assert mono > 0.5, f"fill-aligned OFI should order fill quality, got {mono}"
    d = tab.sort_values("ofi_decile")
    assert d["markout_ticks"].iloc[0] < d["markout_ticks"].iloc[-1]


def test_raw_ofi_is_U_shaped_not_monotone():
    """The reason fill-aligned OFI exists: raw OFI hides a real signal.

    Both tails of raw OFI are toxic, so its Spearman statistic is near zero on
    exactly the data where fill-aligned OFI separates cleanly. If this ever
    starts reading strongly monotone, the side convention has broken.
    """
    cfg = Config()
    df = _with_target(_toxic_flow())
    fills = passive.markout_columns(
        passive.build_passive_fills(df, cfg), cfg, ["ms1000"])
    raw = passive.markout_by_ofi_decile(fills, cfg, ["ms1000"],
                                        bucket_on="prev_ofi")
    aligned = passive.markout_by_ofi_decile(fills, cfg, ["ms1000"],
                                            bucket_on="fill_aligned_ofi")
    mono_raw = abs(float(raw["decile_monotonicity_spearman"].iloc[0]))
    mono_aligned = abs(float(aligned["decile_monotonicity_spearman"].iloc[0]))
    assert mono_aligned > mono_raw
    # Both extremes of raw OFI mark out worse than the middle.
    r = raw.sort_values("ofi_decile")["markout_ticks"].to_numpy()
    assert min(r[0], r[-1]) < np.median(r)


def test_abs_ofi_view_is_well_formed():
    """Structural only, deliberately.

    Whether toxicity rises with |OFI| is an empirical question about real order
    flow, and it depends on OFI being persistent enough that |OFI| before the
    fill says something about the imbalance during it. Asserting that shape on
    a fixture with iid OFI would be asserting a property of the market, not of
    this code, and would pass or fail on noise.
    """
    cfg = Config()
    df = _with_target(_toxic_flow())
    fills = passive.markout_columns(
        passive.build_passive_fills(df, cfg), cfg, ["ms1000"])
    tab = passive.markout_by_ofi_decile(fills, cfg, ["ms1000"],
                                        bucket_on="abs_ofi")
    assert not tab.empty
    assert (tab["bucket_on"] == "abs_ofi").all()
    assert tab["ofi_decile"].nunique() == 10
    assert (tab["mean_bucket_value"] >= 0).all()      # |OFI| is non-negative
    assert tab["mean_bucket_value"].is_monotonic_increasing


def test_bucketing_column_must_exist():
    cfg = Config()
    df = _with_target(_toxic_flow(n=500))
    fills = passive.markout_columns(
        passive.build_passive_fills(df, cfg), cfg, ["ms1000"])
    with pytest.raises(ValueError, match="unknown bucketing column"):
        passive.markout_by_ofi_decile(fills, cfg, ["ms1000"],
                                      bucket_on="not_a_column")


def test_unconditional_table_reports_days_not_just_rows():
    cfg = Config()
    n = 200
    stv = np.where(np.arange(n) % 2 == 0, -100.0, 100.0)
    df = _book(n, stv=stv)
    df["session_date"] = np.repeat([f"2026-01-{d:02d}" for d in range(1, 6)],
                                   n // 5)
    df = _with_target(df)
    fills = passive.markout_columns(
        passive.build_passive_fills(df, cfg), cfg, ["ms1000"])
    tab = passive.unconditional_markout(fills, cfg, ["ms1000"])
    assert int(tab["n_days"].iloc[0]) == 5
    assert tab["se_day_clustered"].notna().all()
