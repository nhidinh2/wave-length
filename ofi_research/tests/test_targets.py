"""Target-alignment tests (section 11 / 28, P0.8).

Three target families with three different contracts: event-time targets are a
research diagnostic that may span zero clock time, clock-time targets must be
at least a horizon later, and execution-aligned targets must additionally be
reachable after the full simulated latency.
"""

import numpy as np
import pandas as pd
import pytest

from ofi_research.config import Config, LatencyConfig
from ofi_research.features import add_segments, add_basic_variables
from ofi_research.targets import (
    add_execution_aligned_targets, add_targets, assert_target_causality,
)


def _prep(df, cfg):
    df = add_segments(df, cfg)
    df = add_basic_variables(df, cfg)
    return df


def _frame(ts, mids):
    """Locked book at ``mids`` so mid == bid == ask, one instrument, one day."""
    n = len(mids)
    return pd.DataFrame({
        "instrument": ["A"] * n,
        "timestamp": pd.to_datetime(ts, utc=True),
        "session_date": [pd.Timestamp("2024-01-01").date()] * n,
        "sequence": np.arange(n),
        "bid_price_1": list(mids), "ask_price_1": list(mids),
        "bid_size_1": [100.0] * n, "ask_size_1": [100.0] * n,
        "bid_price_2": [np.nan] * n, "ask_price_2": [np.nan] * n,
        "bid_size_2": [np.nan] * n, "ask_size_2": [np.nan] * n,
    })


def test_event_horizon_alignment():
    cfg = Config()
    cfg.targets.event_horizons = [1, 2]
    cfg.targets.clock_horizons_ms = []
    n = 5
    df = pd.DataFrame({
        "instrument": ["A"] * n,
        "timestamp": pd.date_range("2024-01-01T09:00:00Z", periods=n, freq="s"),
        "session_date": [pd.Timestamp("2024-01-01").date()] * n,
        "sequence": [np.nan] * n,
        "bid_price_1": [10, 11, 12, 13, 14],
        "ask_price_1": [10, 11, 12, 13, 14],  # locked so mid = bid = ask
        "bid_size_1": [100] * n, "ask_size_1": [100] * n,
        "bid_price_2": [np.nan] * n, "ask_price_2": [np.nan] * n,
        "bid_size_2": [np.nan] * n, "ask_size_2": [np.nan] * n,
    })
    df = _prep(df, cfg)
    out = add_targets(df, cfg)
    # mid = [10,11,12,13,14]; 1-event change = +1 except last (NaN)
    ch1 = out["future_mid_change_ev1"].to_numpy()
    assert np.allclose(ch1[:-1], 1.0)
    assert np.isnan(ch1[-1])
    ch2 = out["future_mid_change_ev2"].to_numpy()
    assert np.allclose(ch2[:-2], 2.0)
    assert np.isnan(ch2[-1]) and np.isnan(ch2[-2])


def test_clock_horizon_first_quote_at_or_after():
    cfg = Config()
    cfg.targets.event_horizons = []
    cfg.targets.clock_horizons_ms = [1000.0]  # 1 second
    cfg.targets.horizon_tolerance_frac = 1.0
    cfg.targets.horizon_abs_tolerance_ms = 600.0
    # irregular timestamps: 0.0, 0.4, 1.2, 2.0 s
    ts = pd.to_datetime([
        "2024-01-01T09:00:00.000Z", "2024-01-01T09:00:00.400Z",
        "2024-01-01T09:00:01.200Z", "2024-01-01T09:00:02.000Z"])
    df = pd.DataFrame({
        "instrument": ["A"] * 4, "timestamp": ts,
        "session_date": [pd.Timestamp("2024-01-01").date()] * 4,
        "sequence": [np.nan] * 4,
        "bid_price_1": [10, 10, 12, 14], "ask_price_1": [10, 10, 12, 14],
        "bid_size_1": [100] * 4, "ask_size_1": [100] * 4,
        "bid_price_2": [np.nan] * 4, "ask_price_2": [np.nan] * 4,
        "bid_size_2": [np.nan] * 4, "ask_size_2": [np.nan] * 4,
    })
    df = _prep(df, cfg)
    out = add_targets(df, cfg)
    # row0 (t=0): first quote >= 1.0s is t=1.2s (mid 12) -> change 12-10=2
    assert out["future_mid_change_ms1000"].iloc[0] == 2.0
    # realized horizon = 1.2s
    assert abs(out["actual_horizon_ms_ms1000"].iloc[0] - 1200.0) < 1e-6


def test_clock_target_does_not_cross_day():
    cfg = Config()
    cfg.targets.event_horizons = []
    cfg.targets.clock_horizons_ms = [1000.0]
    ts = pd.to_datetime([
        "2024-01-01T15:59:59.900Z",  # last of day 1
        "2024-01-02T09:00:00.000Z"])  # first of day 2
    df = pd.DataFrame({
        "instrument": ["A", "A"], "timestamp": ts,
        "session_date": [pd.Timestamp("2024-01-01").date(),
                         pd.Timestamp("2024-01-02").date()],
        "sequence": [np.nan, np.nan],
        "bid_price_1": [10, 20], "ask_price_1": [10, 20],
        "bid_size_1": [100, 100], "ask_size_1": [100, 100],
        "bid_price_2": [np.nan, np.nan], "ask_price_2": [np.nan, np.nan],
        "bid_size_2": [np.nan, np.nan], "ask_size_2": [np.nan, np.nan],
    })
    df = _prep(df, cfg)
    out = add_targets(df, cfg)
    # day-1 row cannot look into day 2 -> target NaN
    assert np.isnan(out["future_mid_change_ms1000"].iloc[0])


# --------------------------------------------------------------------------- #
# Event-time targets: later in event order, possibly not later in clock time
# --------------------------------------------------------------------------- #
def test_event_target_may_share_a_timestamp_but_is_strictly_later_in_order():
    """Several native events can arrive in one packet (P0.8).

    ``j = i + k`` is still future in event space, and the realized elapsed
    time is recorded rather than assumed positive — that column is what makes
    the non-executable fraction visible.
    """
    cfg = Config()
    cfg.targets.event_horizons = [1]
    cfg.targets.clock_horizons_ms = []
    ts = ["2024-01-01T09:00:00.000Z"] * 3 + ["2024-01-01T09:00:01.000Z"]
    out = add_targets(_prep(_frame(ts, [10.0, 11.0, 12.0, 13.0]), cfg), cfg)

    # target exists despite the timestamp tie
    assert out["future_mid_change_ev1"].iloc[0] == 1.0
    # and it is honestly reported as spanning zero clock time
    assert out["actual_horizon_ms_ev1"].iloc[0] == 0.0
    assert out["actual_horizon_ms_ev1"].iloc[2] == 1000.0


def test_event_target_order_key_is_strictly_greater():
    cfg = Config()
    cfg.targets.event_horizons = [2]
    cfg.targets.clock_horizons_ms = []
    ts = ["2024-01-01T09:00:00.000Z"] * 4
    out = add_targets(_prep(_frame(ts, [10.0, 11.0, 12.0, 13.0]), cfg), cfg)

    # +2 events ahead in a tied-timestamp batch: rows 0,1 resolve, 2,3 do not
    assert out["future_mid_change_ev2"].tolist()[:2] == [2.0, 2.0]
    assert np.isnan(out["future_mid_change_ev2"].iloc[2])


# --------------------------------------------------------------------------- #
# Clock targets: horizon tolerance
# --------------------------------------------------------------------------- #
def test_horizon_tolerance_rejects_fractional_overshoot():
    """A 1 s horizon realized 5 s later is not a 1 s observation."""
    cfg = Config()
    cfg.targets.event_horizons = []
    cfg.targets.clock_horizons_ms = [1000.0]
    cfg.targets.horizon_tolerance_frac = 0.5      # accept up to 1500 ms
    cfg.targets.horizon_abs_tolerance_ms = 50.0
    ts = ["2024-01-01T09:00:00.000Z", "2024-01-01T09:00:01.400Z",
          "2024-01-01T09:00:06.000Z"]
    out = add_targets(_prep(_frame(ts, [10.0, 12.0, 20.0]), cfg), cfg)

    assert out["future_mid_change_ms1000"].iloc[0] == 2.0      # 1400 ms: ok
    assert np.isnan(out["future_mid_change_ms1000"].iloc[1])   # 4600 ms: reject
    assert np.isnan(out["actual_horizon_ms_ms1000"].iloc[1])


def test_horizon_tolerance_rejects_absolute_overshoot_on_short_horizons():
    """At a 10 ms horizon the fractional band is tiny; the absolute one binds."""
    cfg = Config()
    cfg.targets.event_horizons = []
    cfg.targets.clock_horizons_ms = [10.0]
    cfg.targets.horizon_tolerance_frac = 0.5      # 5 ms
    cfg.targets.horizon_abs_tolerance_ms = 50.0   # 50 ms -> the binding one
    ts = ["2024-01-01T09:00:00.000Z", "2024-01-01T09:00:00.040Z",
          "2024-01-01T09:00:00.500Z"]
    out = add_targets(_prep(_frame(ts, [10.0, 12.0, 20.0]), cfg), cfg)

    assert out["future_mid_change_ms10"].iloc[0] == 2.0        # 40 ms <= 50
    assert np.isnan(out["future_mid_change_ms10"].iloc[1])     # 460 ms: reject


def test_causality_assertions_pass_on_well_formed_targets():
    cfg = Config()
    cfg.targets.event_horizons = [1]
    cfg.targets.clock_horizons_ms = [100.0]
    ts = pd.date_range("2024-01-01T09:00:00Z", periods=20, freq="50ms")
    out = add_targets(_prep(_frame(ts, np.arange(20.0)), cfg), cfg)
    assert_target_causality(out, cfg)   # raises on any violation


# --------------------------------------------------------------------------- #
# Execution-aligned targets: reachable after total latency
# --------------------------------------------------------------------------- #
def _latency_config(total_ms):
    cfg = Config()
    cfg.costs.latency_ms = 0.0
    cfg.costs.latency = LatencyConfig(capture_to_user_ms=0.0,
                                      decode_decide_ms=0.0,
                                      user_to_venue_ms=total_ms)
    return cfg


def test_execution_entry_respects_total_latency():
    """Entry is the first state at or after t + total latency, never earlier."""
    cfg = _latency_config(12.0)
    ts = pd.date_range("2024-01-01T09:00:00Z", periods=8, freq="5ms")
    df = _prep(_frame(ts, np.arange(8.0)), cfg)
    out = add_execution_aligned_targets(df, cfg, horizon_ms=10.0)

    # rows at 0,5,10,15,... ms; a signal at row 0 cannot fill before 12 ms
    assert out["exec_entry_index_ms10"].iloc[0] == 3      # t = 15 ms
    assert out["exec_entry_delay_ms_ms10"].iloc[0] == pytest.approx(15.0)
    delays = out["exec_entry_delay_ms_ms10"].dropna()
    assert (delays >= 12.0).all()


def test_higher_latency_pushes_the_entry_strictly_later():
    ts = pd.date_range("2024-01-01T09:00:00Z", periods=12, freq="5ms")
    fast = _latency_config(1.0)
    slow = _latency_config(26.0)
    e_fast = add_execution_aligned_targets(
        _prep(_frame(ts, np.arange(12.0)), fast), fast, 10.0)
    e_slow = add_execution_aligned_targets(
        _prep(_frame(ts, np.arange(12.0)), slow), slow, 10.0)

    assert e_fast["exec_entry_index_ms10"].iloc[0] == 1     # 5 ms
    assert e_slow["exec_entry_index_ms10"].iloc[0] == 6     # 30 ms


def test_execution_entry_never_lands_inside_the_signals_own_batch():
    """Records sharing a ts_recv became available at the same instant (P0.5)."""
    cfg = _latency_config(0.0)
    cfg.execution.no_fill_within_signal_batch = True
    ts = ["2024-01-01T09:00:00.000Z"] * 3 + ["2024-01-01T09:00:00.005Z",
                                             "2024-01-01T09:00:00.010Z",
                                             "2024-01-01T09:00:00.015Z"]
    df = _prep(_frame(ts, [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]), cfg)
    out = add_execution_aligned_targets(df, cfg, horizon_ms=5.0)

    # rows 0,1,2 share a timestamp: none of them may fill against another
    assert out["exec_entry_index_ms5"].iloc[0] == 3
    assert out["exec_entry_index_ms5"].iloc[2] == 3
    assert (out["exec_entry_delay_ms_ms5"].iloc[:3] > 0).all()


def test_execution_target_never_crosses_a_segment():
    cfg = _latency_config(1.0)
    ts = pd.date_range("2024-01-01T09:00:00Z", periods=10, freq="5ms")
    df = _frame(ts, np.arange(10.0))
    df["book_clear"] = [False] * 5 + [True] + [False] * 4
    df = _prep(df, cfg)
    out = add_execution_aligned_targets(df, cfg, horizon_ms=5.0)

    entry = out["exec_entry_index_ms5"].to_numpy()
    exit_ = out["exec_exit_index_ms5"].to_numpy()
    seg = out["segment_id"].to_numpy()
    ok = (entry >= 0) & (exit_ >= 0)
    assert (seg[entry[ok]] == seg[ok]).all()
    assert (seg[exit_[ok]] == seg[ok]).all()
