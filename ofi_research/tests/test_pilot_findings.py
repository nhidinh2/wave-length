"""Regressions for what the first real trading day exposed.

Each test here corresponds to something the XNAS.ITCH INTC 2026-08-05 pilot
found that synthetic data could not: a metric that inverted on flat markets, a
latency default that silently reported an unreachable optimum, and auction
prints arriving inline with continuous trading.
"""

import numpy as np
import pandas as pd
import pytest

from ofi_research.config import Config, LatencyConfig
from ofi_research.diagnostics import hac_correlation
from ofi_research.run_experiment import _oos_stats, warn_if_latency_unconfigured
from ofi_research.validation import clean_data


# --------------------------------------------------------------------------- #
# Directional accuracy must not count "unchanged" as "wrong"
# --------------------------------------------------------------------------- #
def test_directional_accuracy_ignores_unchanged_outcomes():
    """A flat mid is not a wrong direction.

    On the pilot day this read 36% at a 100 ms horizon while the correlation
    was POSITIVE — the sample was mostly zero mid-changes, and ``sign(0)``
    never equals +-1.
    """
    pred = np.array([1.0, 1.0, -1.0, -1.0, 1.0, -1.0])
    # two genuine moves, both predicted correctly; four unchanged outcomes
    y = np.array([0.0, 0.0, 0.0, 0.0, 2.0, -2.0])

    stats = _oos_stats(pred, y)
    assert stats["dir_acc"] == pytest.approx(1.0)
    assert stats["pct_unchanged"] == pytest.approx(100.0 * 4 / 6)


def test_directional_accuracy_still_penalises_real_misses():
    pred = np.array([1.0, 1.0, 1.0, 1.0, -1.0, -1.0])
    y = np.array([2.0, -2.0, 2.0, -2.0, 0.0, 0.0])
    stats = _oos_stats(pred, y)
    assert stats["dir_acc"] == pytest.approx(0.5)


def test_hac_directional_accuracy_agrees_and_reports_the_unchanged_share():
    x = pd.Series([1.0, 1.0, -1.0, -1.0] * 20)
    y = pd.Series([0.0, 2.0, 0.0, -2.0] * 20)   # half unchanged, half correct
    res = hac_correlation(x, y, "x", "y")

    assert res.directional_accuracy == pytest.approx(1.0)
    assert res.pct_target_unchanged == pytest.approx(50.0)


def test_a_positive_correlation_cannot_yield_sub_coin_flip_accuracy():
    """The pilot's headline symptom, as an invariant."""
    rng = np.random.default_rng(5)
    x = rng.normal(size=5000)
    y = np.where(rng.random(5000) < 0.8, 0.0, x)  # mostly flat, else agrees

    res = hac_correlation(pd.Series(x), pd.Series(y), "x", "y")
    assert res.pearson > 0
    assert res.directional_accuracy > 0.5
    assert res.pct_target_unchanged > 70.0


# --------------------------------------------------------------------------- #
# A zero-latency run has to say so
# --------------------------------------------------------------------------- #
def test_unconfigured_latency_is_flagged_as_an_upper_bound():
    cfg = Config()
    assert cfg.costs.latency.is_upper_bound()
    caveat = warn_if_latency_unconfigured(cfg)
    assert caveat is not None
    assert "UPPER BOUND" in caveat


def test_configured_latency_produces_no_caveat():
    cfg = Config()
    cfg.costs.latency = LatencyConfig(capture_to_user_ms=0.5,
                                      decode_decide_ms=0.2,
                                      user_to_venue_ms=0.3)
    assert not cfg.costs.latency.is_upper_bound()
    assert cfg.costs.total_latency_ms() == pytest.approx(1.0)
    assert warn_if_latency_unconfigured(cfg) is None


def test_legacy_scalar_latency_also_counts_as_configured():
    cfg = Config()
    cfg.costs.latency_ms = 5.0
    assert warn_if_latency_unconfigured(cfg) is None


# --------------------------------------------------------------------------- #
# Auction prints at the session edges
# --------------------------------------------------------------------------- #
def _session_frame():
    """One session, 09:30:00 to 16:00:00, one quote per minute."""
    ts = pd.date_range("2026-08-05T13:30:00Z", "2026-08-05T20:00:00Z",
                       freq="1min")
    n = len(ts)
    return pd.DataFrame({
        "instrument": ["INTC"] * n,
        "timestamp": ts,
        "session_date": [pd.Timestamp("2026-08-05").date()] * n,
        "sequence": np.arange(n),
        "bid_price_1": [99.99] * n, "ask_price_1": [100.01] * n,
        "bid_size_1": [100.0] * n, "ask_size_1": [100.0] * n,
    })


def test_session_edges_are_trimmed_by_default():
    """The crosses print inline in mbp-10 and are not labeled."""
    cfg = Config()
    assert cfg.data.trim_open_minutes == 1.0
    assert cfg.data.trim_close_minutes == 1.0

    df = _session_frame()
    clean, report = clean_data(df, cfg)

    reasons = set(report["reason"])
    assert reasons == {"open_trim", "close_trim"}
    assert clean["timestamp"].min() > df["timestamp"].min()
    assert clean["timestamp"].max() < df["timestamp"].max()


def test_trimming_breaks_the_ofi_chain_at_the_boundary():
    """The first continuous quote must not difference against an auction print."""
    from ofi_research.features import add_segments, compute_ofi_level

    cfg = Config()
    clean, _ = clean_data(_session_frame(), cfg)
    assert clean["integrity_break"].iloc[0]

    seg = add_segments(clean, cfg)
    assert compute_ofi_level(seg, 1).iloc[0] == 0.0


def test_trimming_can_be_disabled_to_study_the_auctions():
    cfg = Config()
    cfg.data.trim_open_minutes = None
    cfg.data.trim_close_minutes = None
    clean, report = clean_data(_session_frame(), cfg)

    assert len(clean) == len(_session_frame())
    assert report.empty


def test_trimming_is_skipped_on_spans_too_short_to_be_sessions():
    """A 90-second fixture is not a trading session; the trim must not empty it.

    The synthetic generator produces exactly this shape, and an unguarded trim
    removed 100% of it — surfacing several functions later as an unrelated
    IndexError rather than as a cleaning problem.
    """
    cfg = Config()
    ts = pd.date_range("2026-08-05T13:30:00Z", periods=90, freq="1s")
    n = len(ts)
    df = pd.DataFrame({
        "instrument": ["SYN"] * n,
        "timestamp": ts,
        "session_date": [pd.Timestamp("2026-08-05").date()] * n,
        "sequence": np.arange(n),
        "bid_price_1": [99.99] * n, "ask_price_1": [100.01] * n,
        "bid_size_1": [100.0] * n, "ask_size_1": [100.0] * n,
    })
    clean, report = clean_data(df, cfg)

    assert len(clean) == n
    assert report.empty


def test_cleaning_everything_away_fails_loudly():
    cfg = Config()
    cfg.data.trim_open_minutes = None
    cfg.data.trim_close_minutes = None
    ts = pd.date_range("2026-08-05T13:30:00Z", periods=5, freq="1s")
    df = pd.DataFrame({
        "instrument": ["X"] * 5,
        "timestamp": ts,
        "session_date": [pd.Timestamp("2026-08-05").date()] * 5,
        "sequence": np.arange(5),
        "bid_price_1": [100.5] * 5, "ask_price_1": [100.0] * 5,  # all crossed
        "bid_size_1": [100.0] * 5, "ask_size_1": [100.0] * 5,
    })
    with pytest.raises(ValueError, match="removed ALL 5 rows"):
        clean_data(df, cfg)
