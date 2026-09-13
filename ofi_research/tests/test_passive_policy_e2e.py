"""End-to-end: tape written at event resolution, policy read back from it.

The unit tests pin the queue arithmetic. This one pins the JOIN — that a tape
written during prep is found by the walk-forward, that the fold structure
survives the round trip, and that a rebate lands in the grid as a constant per
fill rather than being silently dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ofi_research.config import Config
from ofi_research.passive_policy import run_passive_walk_forward
from ofi_research.passive_tape import (available_tapes, read_tape, write_tape)


def _session(day, n=400, mid0=100.0, seed=0):
    """One session with informative flow: OFI leads the mid, so a gate can work."""
    rng = np.random.default_rng(seed)
    ofi = rng.normal(size=n)
    # Mid follows OFI with a lag, which is the structure the model must find.
    step = 0.002 * np.concatenate(([0.0], ofi[:-1])) + 0.0005 * rng.normal(size=n)
    mid = mid0 + np.cumsum(step)
    spread = 0.02
    return pd.DataFrame({
        "segment_id": 1,
        "session_date": day,
        "timestamp": pd.date_range(f"{day} 14:30", periods=n, freq="100ms"),
        "midprice": mid,
        "spread": spread,
        "bid_price_1": np.round(mid - spread / 2, 4),
        "ask_price_1": np.round(mid + spread / 2, 4),
        "bid_size_1": 100.0,
        "ask_size_1": 100.0,
        "signed_volume_increment": rng.normal(size=n) * 50.0,
        "signed_trade_volume": rng.normal(size=n) * 50.0,
        "OFI_L1_ref": ofi,
        "OFI1_ev50": ofi,
        "in_continuous_session": True,
        "integrity_break": False,
        "future_mid_change_ms1000": np.concatenate(
            (mid[10:] - mid[:-10], np.full(10, np.nan))),
    })


@pytest.fixture()
def prepared(tmp_path):
    """13 sessions — the minimum that yields one 10/2/1 walk-forward fold."""
    days = [f"2026-03-{d:02d}" for d in range(2, 15)]
    cfg = Config()
    cfg.output_dir = str(tmp_path)
    cfg.costs.tick_size = 0.01
    cfg.passive.queue_ahead_fractions = [0.0, 1.0]
    cfg.passive.gate_quantiles = [0.0, 0.5]
    cfg.passive.rebate_sweep = [0.0, 0.002]
    cfg.passive.min_fills_per_cell = 1
    frames = []
    for i, d in enumerate(days):
        s = _session(d, seed=i)
        write_tape(s, cfg)          # full resolution, as prep would
        frames.append(s.iloc[::2])  # a coarser "decision clock"
    return cfg, pd.concat(frames, ignore_index=True)


def test_tapes_land_on_disk_and_read_back(prepared):
    cfg, _ = prepared
    assert len(available_tapes(cfg)) == 13
    tape = read_tape(cfg, "2026-03-02")
    assert tape is not None and len(tape) == 400
    # int64 ns is what the simulator searches on.
    assert tape["timestamp"].dtype == np.int64


def test_policy_runs_and_produces_a_grid(prepared):
    cfg, feat = prepared
    out = run_passive_walk_forward(feat, cfg, "ms1000", model_name="M1_ofi")
    grid = out["23_passive_policy_grid"]
    assert not grid.empty
    # 2 queue positions x 2 gates x 2 rebates
    assert len(grid) == 8
    assert set(grid["queue_ahead_fraction"]) == {0.0, 1.0}
    assert set(grid["maker_rebate_per_unit"]) == {0.0, 0.002}


def test_rebate_shifts_markout_by_exactly_its_size(prepared):
    """A rebate is paid per fill, so it is addition — never a re-simulation."""
    cfg, feat = prepared
    grid = run_passive_walk_forward(feat, cfg, "ms1000",
                                    model_name="M1_ofi")["23_passive_policy_grid"]
    key = ["queue_ahead_fraction", "gate_quantile"]
    a = grid[grid["maker_rebate_per_unit"] == 0.0].set_index(key)
    b = grid[grid["maker_rebate_per_unit"] == 0.002].set_index(key)
    delta = (b["markout_ticks"] - a["markout_ticks"]).dropna()
    assert len(delta)
    assert np.allclose(delta.to_numpy(), 0.002 / cfg.costs.tick_size)


def test_deeper_queue_never_fills_more(prepared):
    """Being further back cannot get you more fills. If it does, we have a bug."""
    cfg, feat = prepared
    grid = run_passive_walk_forward(feat, cfg, "ms1000",
                                    model_name="M1_ofi")["23_passive_policy_grid"]
    z = grid[grid["maker_rebate_per_unit"] == 0.0]
    for gq, g in z.groupby("gate_quantile"):
        front = g[g["queue_ahead_fraction"] == 0.0]["n_fills"].iloc[0]
        back = g[g["queue_ahead_fraction"] == 1.0]["n_fills"].iloc[0]
        assert back <= front


def test_selective_gate_never_fills_more_than_no_gate(prepared):
    """Declining to quote cannot produce more fills than quoting always."""
    cfg, feat = prepared
    grid = run_passive_walk_forward(feat, cfg, "ms1000",
                                    model_name="M1_ofi")["23_passive_policy_grid"]
    z = grid[grid["maker_rebate_per_unit"] == 0.0]
    for alpha, g in z.groupby("queue_ahead_fraction"):
        always = g[g["gate_quantile"] == 0.0]["n_fills"].iloc[0]
        picky = g[g["gate_quantile"] == 0.5]["n_fills"].iloc[0]
        assert picky <= always


def test_missing_tape_is_survivable(prepared, tmp_path):
    """A fold whose day has no tape is skipped, not fatal."""
    cfg, feat = prepared
    (tmp_path / "passive_tapes" / "tape_2026-03-14.parquet").unlink()
    out = run_passive_walk_forward(feat, cfg, "ms1000", model_name="M1_ofi")
    # 13 days gives exactly one fold, whose test day we just deleted.
    assert out == {}
