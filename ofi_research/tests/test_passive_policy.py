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
    b = _book(6, bid_sz=100.0, vols=[0, 60, 0, 0, 60, 0])
    b["px"] = np.array([10.0, 10.0, 10.0, 10.01, 10.01, 10.01])
    out = _sim(b, alpha=1.0)
    # 60 before the move plus 60 after would clear 100 in a single queue.
    # Split across the price change, neither side of it gets there alone.
    assert out["fill_idx"].size == 0
    starts, ends = _episode_bounds(b["px"], b["seg"])
    assert starts.tolist() == [0, 3] and ends.tolist() == [2, 5]


def test_level_clearing_print_fills_at_the_level_it_cleared():
    """A row carries the book AFTER its event, so a print that sweeps the bid
    arrives on a row whose bid has already dropped. It traded at the OLD bid.

    Reading it against its own row credited the fill to 9.99 — a tick better
    than it printed — or, when the price change ended the episode first,
    dropped it entirely. Those are the most adverse fills in the book.
    """
    b = _book(5, bid_sz=100.0, vols=[0, 0, 150, 0, 0])
    b["px"] = np.array([10.0, 10.0, 9.99, 9.99, 9.99])
    b["sz"] = np.array([100.0, 100.0, 80.0, 80.0, 80.0])
    for alpha in (0.0, 1.0):          # 150 clears all 100 ahead either way
        out = _sim(b, alpha=alpha)
        assert out["fill_idx"].tolist() == [2]     # the print's own row/time
        assert out["fill_price"].tolist() == [10.0]


def test_print_on_the_join_row_predates_us():
    """Volume on the row we arrive at traded against the book BEFORE it."""
    b = _book(4, vols=[0, 0, 50, 0])
    b["gate"] = np.array([False, False, True, True])
    assert _sim(b, alpha=0.0)["fill_idx"].size == 0


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


# --- unwinding the inventory -------------------------------------------
from ofi_research.passive_policy import (_bonferroni_t, policy_gate,  # noqa: E402
                                         precompute_side, simulate_unwind)


def _unwind(ask, buys, bid, seg=None, alpha=0.0, fill_row=0,
            timeouts=(3.0,), lat_ns=0):
    """A long filled at 10.00 on ``fill_row``; unwind it on the ask."""
    n = len(ask)
    ts = np.arange(n, dtype=np.int64) * 1_000_000          # 1 ms per row
    seg = np.zeros(n, dtype=np.int64) if seg is None else np.asarray(seg)
    ask = np.asarray(ask, dtype=np.float64)
    sz = np.full(n, 100.0)
    pre = precompute_side(ask, sz, np.asarray(buys, dtype=np.float64), seg)
    return simulate_unwind(ts, seg, np.array([fill_row]), np.array([10.00]),
                           1, ask, sz, np.asarray(bid, dtype=np.float64), pre,
                           alpha, lat_ns, list(timeouts))


def test_passive_exit_earns_the_spread():
    out = _unwind(ask=[10.02] * 5, buys=[0, 0, 30, 0, 0], bid=[10.00] * 5)
    assert out["passive_3"].tolist() == [True]
    assert out["pnl_3"][0] == pytest.approx(0.02)


def test_unfilled_exit_crosses_at_the_deadline():
    """No buyer arrives: sell at the bid standing when the timeout bites."""
    out = _unwind(ask=[10.02] * 6, buys=[0] * 6,
                  bid=[10.00, 10.00, 10.00, 9.98, 9.98, 9.98])
    assert out["passive_3"].tolist() == [False]
    assert out["pnl_3"][0] == pytest.approx(-0.02)


def test_shorter_timeout_crosses_what_a_longer_one_rests_out():
    out = _unwind(ask=[10.02] * 6, buys=[0, 0, 0, 0, 30, 0],
                  bid=[10.00, 10.00, 9.99, 9.99, 9.99, 9.99],
                  timeouts=(2.0, 5.0))
    assert out["passive_2"].tolist() == [False]
    assert out["pnl_2"][0] == pytest.approx(-0.01)
    assert out["passive_5"].tolist() == [True]
    assert out["pnl_5"][0] == pytest.approx(0.02)


def test_exit_repegs_when_the_ask_moves():
    """The ask drops a tick; the exit follows it and fills there."""
    out = _unwind(ask=[10.02, 10.02, 10.01, 10.01, 10.01],
                  buys=[0, 0, 0, 30, 0], bid=[10.00] * 5)
    assert out["passive_3"].tolist() == [True]
    assert out["pnl_3"][0] == pytest.approx(0.01)


def test_exit_behind_the_queue_waits_for_it():
    """At alpha=1 the 100 displayed ahead must trade first."""
    out = _unwind(ask=[10.02] * 6, buys=[0, 60, 0, 0, 60, 0],
                  bid=[10.00] * 6, alpha=1.0, timeouts=(2.0, 5.0))
    assert out["passive_2"].tolist() == [False]    # only 60 by the deadline
    assert out["passive_5"].tolist() == [True]     # 120 > 100 by row 4


def test_exit_that_leaves_the_segment_is_dropped():
    out = _unwind(ask=[10.02] * 4, buys=[0] * 4, bid=[10.00] * 4,
                  seg=[0, 0, 1, 1])
    assert np.isnan(out["pnl_3"][0])


# --- the gate reads significance, not the biggest number ---------------
def test_bonferroni_is_stricter_than_two():
    # 20 cells, 8 days: far above the naive 2.0.
    assert _bonferroni_t(20, 8, 0.05) > 4.0
    assert _bonferroni_t(1, 8, 0.05) == pytest.approx(2.365, abs=1e-3)


def _grid_cell(gq, alpha, mk, t, unwind, unwind_t, n_fills=5000):
    return {"gate_quantile": gq, "queue_ahead_fraction": alpha,
            "maker_rebate_per_unit": 0.0, "markout_ticks": mk, "t_stat": t,
            "n_fills": n_fills, "n_days": 8, "below_min_fills": False,
            "markout_after_crossing_out_ticks": mk - 1.2,
            "unwind_1000ms_ticks": unwind, "unwind_1000ms_t": unwind_t,
            "unwind_1000ms_passive_exit_frac": 0.4}


def _gate(rows):
    cfg = Config()
    cfg.passive.unwind_timeouts_ms = [1000.0]
    return policy_gate(pd.DataFrame(rows), cfg).set_index("criterion")


def test_back_of_queue_cannot_pass_on_a_noisy_cell():
    """The last real run passed this on t = 0.75. It must not."""
    g = _gate([_grid_cell(0.0, 1.0, 0.19, 3.2, -0.5, -3.0),
               _grid_cell(0.9, 1.0, 0.25, 0.75, -0.4, -1.0),
               _grid_cell(0.3, 0.0, 0.27, 7.8, -0.3, -2.0)])
    assert g.loc["survives_back_of_queue", "status"] == "FAIL"


def test_positive_markout_with_a_losing_exit_fails_the_verdict():
    """Free exit at mid is not an exit."""
    g = _gate([_grid_cell(0.3, 0.0, 0.27, 7.8, -0.98, -9.0)])
    assert g.loc["markout_positive_significant", "status"] == "PASS"
    assert g.loc["positive_after_exit_costs", "status"] == "FAIL"
    assert g.loc["rebate_not_load_bearing", "status"] == "FAIL"
    # And the basis states the bracket, not one end of it.
    assert "Bracket" in g.loc["positive_after_exit_costs", "basis"]


def test_filter_comparison_without_days_is_not_evaluable():
    g = _gate([_grid_cell(0.3, 0.0, 0.27, 7.8, 0.1, 1.0)])
    assert g.loc["filter_beats_unconditional", "status"] == "NOT_EVALUABLE"
