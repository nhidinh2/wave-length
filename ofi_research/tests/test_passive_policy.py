"""Queue simulator behaviour, pinned on hand-checkable books.

Every test here is a claim about money: whether an order fills, when, and at
what price. The arithmetic is small enough to verify by eye deliberately —
these are the assertions that keep the passive study from inventing an edge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ofi_research.config import Config
from ofi_research.passive_policy import (_episode_bounds, _gate_on_tape,
                                         _markout, simulate_side_fills)


def _book(n_events, bid=10.00, ask=10.02, bid_sz=100.0, vols=None,
          step_ms=1.0, seg=0):
    """A flat book where only the trade tape varies."""
    ts = (np.arange(n_events, dtype=np.int64)
          * int(step_ms * 1_000_000))
    vol = np.zeros(n_events) if vols is None else np.asarray(vols,
                                                             dtype=np.float64)
    return {
        "ts": ts,
        "px": np.full(n_events, bid, dtype=np.float64),
        "sz": np.full(n_events, bid_sz, dtype=np.float64),
        "vol": vol,
        "seg": np.full(n_events, seg, dtype=np.int64),
        "gate": np.ones(n_events, dtype=bool),
    }


def _sim(b, alpha=0.0, place_ns=0, cancel_ns=0, **kw):
    return simulate_side_fills(b["ts"], b["px"], b["sz"], b["vol"], b["seg"],
                               b["gate"], alpha, place_ns, cancel_ns, **kw)


# --- front of queue reproduces the Phase-0 assumption -------------------
def test_front_of_queue_fills_on_first_trade():
    """alpha=0 must degenerate to the screen, or the two are not comparable."""
    b = _book(5, vols=[0, 0, 30, 0, 40])
    out = _sim(b, alpha=0.0)
    assert out["fill_idx"].tolist() == [2]
    assert out["fill_price"].tolist() == [10.00]


def test_zero_volume_never_fills():
    b = _book(5)
    assert _sim(b, alpha=0.0)["fill_idx"].size == 0


# --- queue position is the whole point ---------------------------------
def test_queue_ahead_must_be_consumed_first():
    """100 shares ahead: 30+40 is not enough, the 50 that follows fills us."""
    b = _book(6, bid_sz=100.0, vols=[0, 30, 40, 0, 50, 10])
    out = _sim(b, alpha=1.0)
    assert out["fill_idx"].tolist() == [4]          # cum 30+40+50 = 120 > 100


def test_deeper_queue_fills_later_or_never():
    """The dial must actually bite: same tape, three positions, three answers."""
    b = _book(6, bid_sz=100.0, vols=[0, 30, 40, 0, 20, 0])
    assert _sim(b, alpha=0.0)["fill_idx"].tolist() == [1]
    assert _sim(b, alpha=0.5)["fill_idx"].tolist() == [2]   # 70 > 50
    assert _sim(b, alpha=1.0)["fill_idx"].size == 0         # 90 never > 100


def test_queue_measured_on_arrival_not_decision():
    """Depth at the moment we ARRIVE is what we queue behind."""
    b = _book(6, vols=[0, 0, 0, 0, 60, 0])
    b["sz"] = np.array([1000.0, 1000.0, 50.0, 50.0, 50.0, 50.0])
    # 2 ms of latency puts arrival at index 2, where only 50 sit ahead.
    out = _sim(b, alpha=1.0, place_ns=2_000_000)
    assert out["fill_idx"].tolist() == [4]
    # Arriving instantly would have queued behind 1000 and never filled.
    assert _sim(b, alpha=1.0, place_ns=0)["fill_idx"].size == 0


# --- price changes end the episode -------------------------------------
def test_price_change_starts_a_new_queue():
    """Our place in line does not survive the level moving."""
    b = _book(6, bid_sz=100.0, vols=[0, 60, 0, 60, 0, 0])
    b["px"] = np.array([10.0, 10.0, 10.0, 10.01, 10.01, 10.01])
    out = _sim(b, alpha=1.0)
    # 60 before the move plus 60 after would clear 100 in a single queue.
    # Split across the price change, neither side of it gets there alone.
    assert out["fill_idx"].size == 0
    starts, ends = _episode_bounds(b["px"], b["seg"])
    assert starts.tolist() == [0, 3] and ends.tolist() == [2, 5]


def test_segment_break_starts_a_new_queue():
    b = _book(4, bid_sz=10.0, vols=[0, 6, 6, 0])
    b["seg"] = np.array([0, 0, 1, 1])
    assert _sim(b, alpha=1.0)["fill_idx"].size == 0


def test_one_fill_per_episode():
    """We quote one lot; a second fill in the same queue would be free size."""
    b = _book(5, vols=[0, 500, 500, 500, 500])
    assert _sim(b, alpha=0.0)["fill_idx"].size == 1


# --- the gate ----------------------------------------------------------
def test_gate_off_means_no_quote():
    b = _book(4, vols=[0, 50, 50, 50])
    b["gate"] = np.zeros(4, dtype=bool)
    assert _sim(b)["fill_idx"].size == 0


def test_quote_starts_at_first_gated_instant():
    """Volume that traded before we decided to quote is not ours to consume."""
    b = _book(6, bid_sz=100.0, vols=[200, 0, 0, 60, 60, 0])
    b["gate"] = np.array([False, False, True, True, True, True])
    out = _sim(b, alpha=1.0)
    assert out["join_idx"].tolist() == [2]      # the early 200 does not count
    assert out["fill_idx"].tolist() == [4]      # 60+60 = 120 > 100


def test_cancel_latency_leaves_us_exposed():
    """Withdrawing is not instant, and the fill inside the window is real."""
    b = _book(6, vols=[0, 0, 0, 80, 0, 0])
    b["gate"] = np.array([True, True, False, False, False, False])
    # Cancel decided at index 2; with a 2 ms round trip the index-3 trade
    # (1 ms later) still fills us.
    assert _sim(b, alpha=0.0, cancel_ns=2_000_000)["fill_idx"].tolist() == [3]
    # With an instant cancel it does not.
    assert _sim(b, alpha=0.0, cancel_ns=0)["fill_idx"].size == 0


# --- cancels in the queue ahead ----------------------------------------
def test_cancels_ahead_do_not_help_by_default():
    """Depth evaporating must not promote us for free."""
    b = _book(5, vols=[0, 0, 0, 50, 0])
    b["sz"] = np.array([100.0, 100.0, 40.0, 40.0, 40.0])
    # Trades alone move 50 of the 100 ahead of us: not enough.
    assert _sim(b, alpha=1.0)["fill_idx"].size == 0
    # Crediting the 60 that cancelled as having been in front makes it 110.
    got = _sim(b, alpha=1.0, cancels_leave_from_behind=False)
    assert got["fill_idx"].tolist() == [3]


# --- markout -----------------------------------------------------------
def test_markout_signs_with_the_side():
    tape = pd.DataFrame({
        "timestamp": np.arange(4, dtype=np.int64) * 1_000_000,
        "midprice": [10.01, 10.01, 10.05, 10.05],
        "segment_id": [0, 0, 0, 0],
        "bid_price_1": [10.0, 10.0, 10.04, 10.04],
        "ask_price_1": [10.02, 10.02, 10.06, 10.06],
    })
    idx = np.array([0]); px = np.array([10.0])
    up = _markout(tape, idx, px, side=1, horizon_ms=2.0)
    assert up["markout"] == pytest.approx(0.05)     # bought at 10.00, mid 10.05
    down = _markout(tape, idx, np.array([10.02]), side=-1, horizon_ms=2.0)
    assert down["markout"] == pytest.approx(-0.03)  # sold at 10.02, mid rose


def test_markout_dropped_when_horizon_leaves_the_segment():
    tape = pd.DataFrame({
        "timestamp": np.arange(4, dtype=np.int64) * 1_000_000,
        "midprice": [10.0, 10.0, 10.0, 10.0],
        "segment_id": [0, 0, 1, 1],
        "bid_price_1": [10.0] * 4, "ask_price_1": [10.02] * 4,
    })
    out = _markout(tape, np.array([0]), np.array([10.0]), 1, horizon_ms=3.0)
    assert out["markout"].size == 0


# --- gate projection ---------------------------------------------------
def test_gate_holds_between_decisions():
    tape_ts = np.arange(0, 10, dtype=np.int64)
    dec_ts = np.array([2, 6], dtype=np.int64)
    got = _gate_on_tape(tape_ts, dec_ts, np.array([True, False]))
    # Nothing before the first decision; True from 2 until the 6 flips it off.
    assert got.tolist() == [False, False, True, True, True, True,
                            False, False, False, False]


def test_no_decisions_means_no_quoting():
    got = _gate_on_tape(np.arange(5, dtype=np.int64),
                        np.empty(0, dtype=np.int64), np.empty(0, dtype=bool))
    assert not got.any()


# --- config plumbing ---------------------------------------------------
def test_passive_config_round_trips_through_json():
    cfg = Config.from_dict({"passive": {"queue_ahead_fractions": [0.0, 0.75],
                                        "rebate_sweep": [0.0, 0.002],
                                        "gate_quantiles": [0.0, 0.8]}})
    assert cfg.passive.queue_ahead_fractions == [0.0, 0.75]
    assert cfg.passive.rebate_sweep == [0.0, 0.002]
    assert cfg.evaluation.run_taker_backtest is False
