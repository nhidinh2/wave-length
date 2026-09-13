"""Aggressive-taker execution against displayed depth (P0.9 / P0.5).

Queue position is irrelevant for a taker, but everything else about the fill
is not: which side is crossed, which book is used, how far the order walks,
and what happens when the displayed depth runs out.
"""

import numpy as np
import pandas as pd
import pytest

from ofi_research.backtest import run_backtest
from ofi_research.config import (
    Config, CostConfig, ExecutionConfig, LatencyConfig, SignalConfig,
)
from ofi_research.costs import (
    compute_trade_pnl, compute_trade_pnl_walked, walk_depth,
)

# A two-level book: bids 99.99 x100 / 99.98 x200, asks 100.01 x100 / 100.02 x200
BID_PX = [99.99, 99.98]
BID_SZ = [100.0, 200.0]
ASK_PX = [100.01, 100.02]
ASK_SZ = [100.0, 200.0]

EXEC = ExecutionConfig(depth_levels=2, max_participation=1.0,
                       allow_partial_fill=True)


def _walked(direction, size, cfg=None, exec_cfg=None):
    return compute_trade_pnl_walked(
        direction, size,
        entry_bid_px=BID_PX, entry_bid_sz=BID_SZ,
        entry_ask_px=ASK_PX, entry_ask_sz=ASK_SZ,
        exit_bid_px=BID_PX, exit_bid_sz=BID_SZ,
        exit_ask_px=ASK_PX, exit_ask_sz=ASK_SZ,
        cfg=cfg or CostConfig(), exec_cfg=exec_cfg or EXEC)


# --------------------------------------------------------------------------- #
# Which side gets crossed
# --------------------------------------------------------------------------- #
def test_buy_crosses_asks_and_sell_crosses_bids():
    buy = _walked(1, 50.0)
    sell = _walked(-1, 50.0)

    assert buy.entry_price == pytest.approx(100.01)   # lifted the ask
    assert buy.exit_price == pytest.approx(99.99)     # hit the bid to close
    assert sell.entry_price == pytest.approx(99.99)   # hit the bid
    assert sell.exit_price == pytest.approx(100.01)   # lifted the ask to close


def test_a_round_trip_pays_the_whole_spread_not_half():
    """Open and close both cross; mid-to-mid is the theoretical number."""
    pnl = _walked(1, 50.0)
    spread = ASK_PX[0] - BID_PX[0]

    assert pnl.gross_mid_pnl == pytest.approx(0.0)    # unchanged book
    assert pnl.executable_gross == pytest.approx(-spread * 50.0)
    assert pnl.net_pnl < 0


def test_direction_must_be_plus_or_minus_one():
    with pytest.raises(ValueError):
        _walked(0, 10.0)


# --------------------------------------------------------------------------- #
# Walking displayed depth
# --------------------------------------------------------------------------- #
def test_order_within_l1_uses_the_top_level_only():
    fill = walk_depth(ASK_PX, ASK_SZ, 60.0, EXEC)
    assert fill.filled_size == 60.0
    assert fill.levels_used == 1
    assert fill.avg_price == pytest.approx(100.01)
    assert not fill.exhausted


def test_multi_level_walk_is_volume_weighted():
    """150 shares: 100 @ 100.01 then 50 @ 100.02."""
    fill = walk_depth(ASK_PX, ASK_SZ, 150.0, EXEC)
    expected = (100 * 100.01 + 50 * 100.02) / 150

    assert fill.filled_size == 150.0
    assert fill.levels_used == 2
    assert fill.avg_price == pytest.approx(expected)
    assert fill.avg_price > ASK_PX[0]    # a large order pays worse than L1


def test_walked_price_flows_into_the_round_trip():
    pnl = _walked(1, 150.0)
    expected_entry = (100 * 100.01 + 50 * 100.02) / 150
    expected_exit = (100 * 99.99 + 50 * 99.98) / 150

    assert pnl.entry_price == pytest.approx(expected_entry)
    assert pnl.exit_price == pytest.approx(expected_exit)
    assert pnl.entry_levels_used == 2
    assert pnl.size == 150.0


def test_single_level_pnl_matches_walked_pnl_for_a_small_order():
    """The L1-only path and the walking path must not disagree at L1 size."""
    cfg = CostConfig()
    walked = _walked(1, 50.0, cfg=cfg)
    single = compute_trade_pnl(1, 50.0, BID_PX[0], ASK_PX[0],
                               BID_PX[0], ASK_PX[0], cfg)
    assert walked.net_pnl == pytest.approx(single.net_pnl)


def test_missing_deep_levels_are_skipped_not_treated_as_free():
    prices = [100.01, np.nan, 100.03]
    sizes = [100.0, np.nan, 200.0]
    fill = walk_depth(prices, sizes, 150.0, ExecutionConfig(
        depth_levels=3, max_participation=1.0))
    assert fill.filled_size == 150.0
    assert fill.avg_price == pytest.approx((100 * 100.01 + 50 * 100.03) / 150)


# --------------------------------------------------------------------------- #
# Insufficient depth and participation caps
# --------------------------------------------------------------------------- #
def test_insufficient_depth_rejects_the_trade_when_partials_are_disallowed():
    strict = ExecutionConfig(depth_levels=2, max_participation=1.0,
                             allow_partial_fill=False)
    assert _walked(1, 500.0, exec_cfg=strict) is None


def test_insufficient_depth_fills_partially_and_reports_it():
    """With no participation cap, running out of book is flagged as exhausted."""
    loose = ExecutionConfig(depth_levels=2, max_participation=None,
                            allow_partial_fill=True)
    pnl = _walked(1, 500.0, exec_cfg=loose)

    assert pnl is not None
    assert pnl.size == 300.0                 # all displayed depth, no more
    assert pnl.requested_size == 500.0
    assert pnl.depth_exhausted is True
    assert pnl.participation_capped is False


def test_a_participation_cap_is_reported_separately_from_exhaustion():
    """Two different reasons for a short fill must stay distinguishable."""
    capped = ExecutionConfig(depth_levels=2, max_participation=1.0,
                             allow_partial_fill=True)
    pnl = _walked(1, 500.0, exec_cfg=capped)

    assert pnl.size == 300.0
    assert pnl.participation_capped is True
    assert pnl.depth_exhausted is False   # the cap bound first, not the book


def test_participation_cap_limits_size_to_displayed_liquidity():
    capped = ExecutionConfig(depth_levels=2, max_participation=0.25,
                             allow_partial_fill=True)
    fill = walk_depth(ASK_PX, ASK_SZ, 300.0, capped)

    assert fill.capped_by_participation is True
    assert fill.filled_size == pytest.approx(0.25 * 300.0)
    assert not fill.exhausted


def test_empty_book_books_no_trade():
    assert compute_trade_pnl_walked(
        1, 10.0,
        entry_bid_px=BID_PX, entry_bid_sz=BID_SZ,
        entry_ask_px=[np.nan, np.nan], entry_ask_sz=[np.nan, np.nan],
        exit_bid_px=BID_PX, exit_bid_sz=BID_SZ,
        exit_ask_px=ASK_PX, exit_ask_sz=ASK_SZ,
        cfg=CostConfig(), exec_cfg=EXEC) is None


# --------------------------------------------------------------------------- #
# Fees and slippage: consistent price/share units
# --------------------------------------------------------------------------- #
def test_per_share_fees_scale_with_size():
    cfg = CostConfig(fee_per_unit=0.001)
    small, large = _walked(1, 50.0, cfg=cfg), _walked(1, 100.0, cfg=cfg)

    assert small.fees == pytest.approx(2 * 0.001 * 50.0)     # both sides
    assert large.fees == pytest.approx(2 * small.fees)


def test_basis_point_fees_scale_with_notional():
    cfg = CostConfig(fee_bps=1.0)
    pnl = _walked(1, 50.0, cfg=cfg)
    expected = (1.0 / 1e4) * (pnl.entry_price + pnl.exit_price) * 50.0
    assert pnl.fees == pytest.approx(expected)


def test_slippage_is_charged_in_ticks_on_both_sides():
    cfg = CostConfig(slippage_ticks=1.0, tick_size=0.01)
    base = _walked(1, 50.0, cfg=CostConfig(tick_size=0.01))
    slipped = _walked(1, 50.0, cfg=cfg)

    assert slipped.entry_price == pytest.approx(base.entry_price + 0.01)
    assert slipped.exit_price == pytest.approx(base.exit_price - 0.01)
    assert slipped.slippage_cost == pytest.approx(2 * 0.01 * 50.0)
    assert slipped.net_pnl < base.net_pnl


def test_impact_scales_with_filled_size_not_requested_size():
    cfg = CostConfig(impact_coefficient=0.002)
    loose = ExecutionConfig(depth_levels=2, max_participation=1.0,
                            allow_partial_fill=True)
    pnl = _walked(1, 500.0, cfg=cfg, exec_cfg=loose)
    assert pnl.impact == pytest.approx(0.002 * 300.0)


# --------------------------------------------------------------------------- #
# The book a fill is taken from (P0.5)
# --------------------------------------------------------------------------- #
def _ladder_frame(times_ms, ask_l1):
    """One instrument, one segment; ask price varies row by row."""
    n = len(times_ms)
    ts = [pd.Timestamp("2024-03-05T14:30:00Z")
          + pd.Timedelta(milliseconds=m) for m in times_ms]
    df = pd.DataFrame({
        "instrument": ["TEST"] * n,
        "timestamp": pd.to_datetime(ts, utc=True),
        "segment_id": [1] * n,
        "bid_price_1": [99.99] * n,
        "ask_price_1": list(ask_l1),
        "bid_size_1": [10_000.0] * n,
        "ask_size_1": [10_000.0] * n,
    })
    df["midprice"] = (df["bid_price_1"] + df["ask_price_1"]) / 2.0
    df["spread"] = df["ask_price_1"] - df["bid_price_1"]
    return df


def _cost_cfg(total_ms):
    return CostConfig(latency_ms=0.0, tick_size=0.01,
                      latency=LatencyConfig(capture_to_user_ms=0.0,
                                            decode_decide_ms=0.0,
                                            user_to_venue_ms=total_ms))


def test_fill_begins_from_the_post_latency_book():
    """A signal at t is filled against the book at t + total latency."""
    # rows at 0, 5, 10, ... ms; the ask improves only after 10 ms
    df = _ladder_frame([0, 5, 10, 15, 20, 25, 30],
                       [100.01, 100.01, 100.01, 100.50, 100.50,
                        100.50, 100.50])
    preds = np.array([5.0] + [np.nan] * 6)      # trade only from row 0

    res = run_backtest(df, preds, np.array([1.0] * 7),
                       _cost_cfg(12.0), SignalConfig(z_thresholds=[0.5]),
                       z_threshold=0.5, horizon_ms=10.0,
                       exec_cfg=ExecutionConfig(depth_levels=1,
                                                max_participation=1.0))

    assert len(res.ledger) == 1
    row = res.ledger.iloc[0]
    assert row["execution_delay_ms"] >= 12.0
    # the fill used the WORSE post-latency ask, not the one visible at signal
    assert row["entry_ask"] == pytest.approx(100.50)
    assert row["entry_price"] == pytest.approx(100.50)


def test_zero_latency_still_cannot_fill_on_the_signal_row():
    df = _ladder_frame([0, 5, 10, 15], [100.01, 100.20, 100.20, 100.20])
    preds = np.array([5.0, np.nan, np.nan, np.nan])

    res = run_backtest(df, preds, np.array([1.0] * 4),
                       _cost_cfg(0.0), SignalConfig(z_thresholds=[0.5]),
                       z_threshold=0.5, horizon_ms=5.0,
                       exec_cfg=ExecutionConfig(depth_levels=1,
                                                max_participation=1.0))

    assert len(res.ledger) == 1
    assert res.ledger["execution_time"].iloc[0] > \
        res.ledger["signal_time"].iloc[0]
    assert res.ledger["entry_ask"].iloc[0] == pytest.approx(100.20)


def test_no_fill_against_a_record_sharing_the_signals_ts_recv():
    """Everything captured in one packet became available at one instant."""
    # rows 0,1,2 share ts_recv=0; the next distinct ts_recv is at 5 ms
    df = _ladder_frame([0, 0, 0, 5, 10],
                       [100.01, 100.02, 100.03, 100.40, 100.40])
    preds = np.array([5.0, np.nan, np.nan, np.nan, np.nan])

    res = run_backtest(df, preds, np.array([1.0] * 5),
                       _cost_cfg(0.0), SignalConfig(z_thresholds=[0.5]),
                       z_threshold=0.5, horizon_ms=5.0,
                       exec_cfg=ExecutionConfig(depth_levels=1,
                                                max_participation=1.0,
                                                no_fill_within_signal_batch=True))

    assert len(res.ledger) == 1
    row = res.ledger.iloc[0]
    assert row["execution_time"] > row["signal_time"]
    # not 100.02 / 100.03: those shared the signal's ts_recv
    assert row["entry_ask"] == pytest.approx(100.40)


def test_routing_venue_is_reported_and_not_claimed_as_nbbo():
    df = _ladder_frame([0, 5, 10], [100.01, 100.01, 100.01])
    res = run_backtest(df, np.array([np.nan] * 3), np.array([1.0] * 3),
                       _cost_cfg(0.0), SignalConfig(), z_threshold=0.5,
                       horizon_ms=5.0,
                       exec_cfg=ExecutionConfig(routing_venue="XNAS"))
    joined = " ".join(res.notes)
    assert "XNAS" in joined
    assert "NBBO" in joined
    assert "TAKER" in joined.upper()
