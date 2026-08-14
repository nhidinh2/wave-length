"""Backtest engine, signal/position rules, and trade ledger (sections 19-20).

Contracts honored here:

* Predictions are used only at/after their signal time; execution happens at the
  first quote with timestamp >= signal_time + latency (no look-ahead, no
  impossible simultaneous fill).
* Exit happens at the first quote at/after entry_time + horizon (clock) or after
  ``k`` events (event horizon), within the same segment.
* Overlapping trades are prevented in v1 (no new entry while a position is open).
* Executable bid/ask P&L is computed explicitly via :mod:`costs`; mid-to-mid
  P&L is reported separately and labeled gross/theoretical.
* Positions reset at segment (session) end; no trade opens if it cannot close
  inside the same segment.

Queue position CANNOT be modeled from top-of-book data; this is reported, and
aggressive (taker) execution is assumed throughout.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import CostConfig, ExecutionConfig, SignalConfig
from .costs import (
    compute_trade_pnl, compute_trade_pnl_walked, estimate_round_trip_cost,
)

logger = logging.getLogger(__name__)

LEDGER_COLUMNS = [
    "signal_time", "execution_time", "instrument", "direction",
    "model_prediction", "forecast_volatility", "z_score",
    "entry_bid", "entry_ask", "entry_price",
    "exit_time", "exit_bid", "exit_ask", "exit_price",
    "size", "requested_size", "gross_mid_pnl", "spread_cost", "fees",
    "slippage_cost", "impact", "net_pnl", "holding_period_ms", "ofi_value",
    "entry_levels_used", "exit_levels_used", "depth_exhausted",
    "participation_capped", "execution_delay_ms",
    "regime_spread", "regime_vol", "regime_tod",
]


def _depth_arrays(df: pd.DataFrame, n_levels: int):
    """Stack available book levels into (n_rows, n_levels) float arrays.

    Returns ``None`` when no level-1 columns exist, in which case the caller
    falls back to single-level execution.
    """
    def stack(prefix: str):
        cols = [f"{prefix}_{k}" for k in range(1, n_levels + 1)]
        cols = [c for c in cols if c in df.columns]
        if not cols:
            return None
        return np.column_stack([df[c].to_numpy(dtype="float64") for c in cols])

    bid_px, ask_px = stack("bid_price"), stack("ask_price")
    bid_sz, ask_sz = stack("bid_size"), stack("ask_size")
    if any(a is None for a in (bid_px, ask_px, bid_sz, ask_sz)):
        return None
    return bid_px, bid_sz, ask_px, ask_sz


@dataclass
class BacktestResult:
    ledger: pd.DataFrame
    metrics: Dict[str, float]
    notes: List[str]


def _position_size(pred: float, sigma: float, cfg: SignalConfig,
                   eps: float = 1e-12) -> float:
    if cfg.position_mode == "fixed":
        return 1.0
    if cfg.position_mode == "vol_scaled":
        raw = pred / (sigma ** 2 + eps)
        return float(np.clip(raw, -cfg.max_position, cfg.max_position))
    if cfg.position_mode == "capped_continuous":
        raw = pred / (sigma + eps)
        return float(np.clip(raw, -cfg.max_position, cfg.max_position))
    return 1.0


def run_backtest(df: pd.DataFrame, predictions: np.ndarray,
                 sigma: np.ndarray, cost_cfg: CostConfig,
                 signal_cfg: SignalConfig, z_threshold: float,
                 horizon_ms: Optional[float] = None,
                 event_horizon: Optional[int] = None,
                 ofi_col: str = "OFI_L1_ref",
                 exec_cfg: Optional[ExecutionConfig] = None) -> BacktestResult:
    """Simulate the strategy on one contiguous test block.

    Exactly one of ``horizon_ms`` / ``event_horizon`` must be given.
    ``sigma`` may be a scalar array (per-row forecast noise).

    Execution uses the POST-LATENCY book, walking as many displayed levels as
    the order requires (P0.9). Total latency is
    ``cost_cfg.total_latency_ms()`` — capture-to-user plus decode/decide plus
    user-to-venue — not a bare ``latency_ms`` (P0.5).
    """
    if (horizon_ms is None) == (event_horizon is None):
        raise ValueError("provide exactly one of horizon_ms / event_horizon")

    df = df.reset_index(drop=True)
    n = len(df)
    sigma = np.asarray(sigma, dtype="float64")
    if sigma.ndim == 0:
        sigma = np.full(n, float(sigma))
    eps = 1e-12
    exec_cfg = exec_cfg or ExecutionConfig()
    depth = _depth_arrays(df, exec_cfg.depth_levels)
    notes = [
        "Queue position not modeled; aggressive TAKER execution assumed "
        "throughout.",
        f"Routing venue assumed: {exec_cfg.routing_venue}. XNAS.ITCH is a "
        "venue-LOCAL book; its mid/spread are not the national NBBO.",
    ]
    if depth is None:
        notes.append("Deep levels unavailable -> single-level execution; "
                     "orders larger than L1 are NOT price-walked.")
    else:
        notes.append(f"Depth walking enabled over {depth[0].shape[1]} level(s); "
                     f"size capped at {exec_cfg.max_participation:.0%} of "
                     "displayed liquidity.")

    t_ns = df["timestamp"].astype("int64").to_numpy()
    seg = df["segment_id"].to_numpy()
    bid = df["bid_price_1"].to_numpy(dtype="float64")
    ask = df["ask_price_1"].to_numpy(dtype="float64")
    mid = df["midprice"].to_numpy(dtype="float64")
    spread = df["spread"].to_numpy(dtype="float64")
    valid_book = (~np.isnan(bid)) & (~np.isnan(ask)) & (ask >= bid)

    # precompute per-segment sorted index arrays for searchsorted
    seg_idx = {s: np.sort(ix) for s, ix in
               pd.Series(range(n)).groupby(seg).groups.items()}
    seg_of = seg

    def first_at_or_after(i: int, target_ns: int) -> int:
        idx = seg_idx[seg_of[i]]
        pos = np.searchsorted(t_ns[idx], target_ns, side="left")
        if pos >= len(idx):
            return -1
        return int(idx[pos])

    def first_executable_after(i: int) -> int:
        """Earliest row at which a decision taken on row ``i`` could be filled.

        Two constraints that a plain "first timestamp >= t + latency" search
        does not enforce, and that only bite once the data has tied
        timestamps (P0.5):

        * **Never before the signal.** Several records can share a ``ts_recv``
          because they were captured in one packet. Searching by timestamp
          alone returns the FIRST member of that batch, which may sit earlier
          in native order than the record that produced the signal — a fill
          booked in the past.
        * **Never inside the signal's own batch.** Every record sharing that
          ``ts_recv`` became available at the same instant, so a decision made
          from one of them cannot be executed against another. The earliest
          honest fill is the first record with a strictly later ``ts_recv``.
        """
        idx = seg_idx[seg_of[i]]
        target = t_ns[i] + latency_ns
        if exec_cfg.no_fill_within_signal_batch:
            target = max(target, t_ns[i] + 1)
        pos = int(np.searchsorted(t_ns[idx], target, side="left"))
        # strictly after the signal row in native order, regardless of ties
        pos = max(pos, int(np.searchsorted(idx, i, side="right")))
        if pos >= len(idx):
            return -1
        return int(idx[pos])

    def exit_index(exec_i: int) -> int:
        if horizon_ms is not None:
            tgt = t_ns[exec_i] + int(horizon_ms * 1_000_000)
            return first_at_or_after(exec_i, tgt)
        # event horizon
        idx = seg_idx[seg_of[exec_i]]
        loc = int(np.searchsorted(idx, exec_i))
        j = loc + int(event_horizon)
        return int(idx[j]) if j < len(idx) else -1

    rows: List[dict] = []
    open_until_ns = -1  # overlap guard

    # Signal funnel (P0.11). An empty ledger looks identical whether the policy
    # never fired, could not be filled, or hit insufficient depth. These
    # counters make the drop-off point explicit.
    fun = dict(rows_total=n, rows_predictable=0, blocked_overlap=0,
               pass_cost_gate=0, pass_z_gate=0, signals=0, drop_no_execution=0,
               drop_no_exit=0, drop_no_depth=0, longs=0, shorts=0)
    abs_pred_sum = 0.0
    abs_pred_max = 0.0
    cost_thresh_sum = 0.0

    reg_spread = df["regime_spread"].to_numpy() if "regime_spread" in df else \
        np.array([None] * n, dtype=object)
    reg_vol = df["regime_vol"].to_numpy() if "regime_vol" in df else \
        np.array([None] * n, dtype=object)
    reg_tod = df["regime_tod"].to_numpy() if "regime_tod" in df else \
        np.array([None] * n, dtype=object)
    ofi_vals = df[ofi_col].to_numpy() if ofi_col in df else np.full(n, np.nan)
    instr = df["instrument"].to_numpy()

    total_latency_ms = cost_cfg.total_latency_ms()
    latency_ns = int(total_latency_ms * 1_000_000)
    notes.append(
        f"Total simulated latency {total_latency_ms:.3f} ms "
        f"(= latency_ms {cost_cfg.latency_ms:.3f} + capture->user "
        f"{cost_cfg.latency.capture_to_user_ms:.3f} + decode/decide "
        f"{cost_cfg.latency.decode_decide_ms:.3f} + user->venue "
        f"{cost_cfg.latency.user_to_venue_ms:.3f}). ts_recv is DATABENTO "
        "capture time, so zero total latency is an upper bound, not a "
        "reachable baseline.")
    notes.append(
        "Same-ts_recv batching: "
        + ("ON — a fill is never booked against a record sharing the signal's "
           "ts_recv, because every record in a captured packet became "
           "available at the same instant."
           if exec_cfg.no_fill_within_signal_batch else
           "OFF — intra-batch fills are permitted. NOT a defensible live "
           "assumption; use only to measure what the assumption costs."))

    for i in range(n):
        pred = predictions[i]
        if np.isnan(pred) or not valid_book[i]:
            continue
        if signal_cfg.prevent_overlap and t_ns[i] < open_until_ns:
            fun["blocked_overlap"] += 1
            continue

        sig = sigma[i] if sigma[i] > 0 else eps
        z = pred / (sig + eps)
        size = abs(_position_size(pred, sig, signal_cfg))
        if size <= 0:
            continue
        cost_thresh = estimate_round_trip_cost(mid[i], spread[i], size, cost_cfg)

        fun["rows_predictable"] += 1
        abs_pred_sum += abs(float(pred))
        abs_pred_max = max(abs_pred_max, abs(float(pred)))
        cost_thresh_sum += float(cost_thresh)
        passed_cost = abs(pred) > cost_thresh
        passed_z = abs(z) > z_threshold
        fun["pass_cost_gate"] += int(passed_cost)
        fun["pass_z_gate"] += int(passed_z)

        if pred > cost_thresh and z > z_threshold:
            direction = 1
        elif pred < -cost_thresh and z < -z_threshold:
            direction = -1
        else:
            continue
        fun["signals"] += 1

        exec_i = first_executable_after(i)
        if exec_i < 0 or not valid_book[exec_i]:
            fun["drop_no_execution"] += 1
            continue
        ex_i = exit_index(exec_i)
        if ex_i < 0 or not valid_book[ex_i]:
            fun["drop_no_exit"] += 1
            continue  # cannot close inside segment -> skip

        if depth is not None:
            bid_px, bid_sz, ask_px, ask_sz = depth
            pnl = compute_trade_pnl_walked(
                direction, size,
                entry_bid_px=bid_px[exec_i], entry_bid_sz=bid_sz[exec_i],
                entry_ask_px=ask_px[exec_i], entry_ask_sz=ask_sz[exec_i],
                exit_bid_px=bid_px[ex_i], exit_bid_sz=bid_sz[ex_i],
                exit_ask_px=ask_px[ex_i], exit_ask_sz=ask_sz[ex_i],
                cfg=cost_cfg, exec_cfg=exec_cfg)
            if pnl is None:
                fun["drop_no_depth"] += 1
                continue  # insufficient displayed depth -> no trade booked
        else:
            pnl = compute_trade_pnl(
                direction, size,
                entry_bid=bid[exec_i], entry_ask=ask[exec_i],
                exit_bid=bid[ex_i], exit_ask=ask[ex_i], cfg=cost_cfg)

        rows.append({
            "signal_time": df["timestamp"].iloc[i],
            "execution_time": df["timestamp"].iloc[exec_i],
            "instrument": instr[i],
            "direction": direction,
            "model_prediction": float(pred),
            "forecast_volatility": float(sig),
            "z_score": float(z),
            "entry_bid": bid[exec_i], "entry_ask": ask[exec_i],
            "entry_price": pnl.entry_price,
            "exit_time": df["timestamp"].iloc[ex_i],
            "exit_bid": bid[ex_i], "exit_ask": ask[ex_i],
            "exit_price": pnl.exit_price,
            "size": pnl.size,
            "requested_size": size,
            "entry_levels_used": pnl.entry_levels_used,
            "exit_levels_used": pnl.exit_levels_used,
            "depth_exhausted": pnl.depth_exhausted,
            "participation_capped": pnl.participation_capped,
            "execution_delay_ms": (t_ns[exec_i] - t_ns[i]) / 1e6,
            "gross_mid_pnl": pnl.gross_mid_pnl,
            "spread_cost": pnl.spread_cost,
            "fees": pnl.fees,
            "slippage_cost": pnl.slippage_cost,
            "impact": pnl.impact,
            "net_pnl": pnl.net_pnl,
            "holding_period_ms": (t_ns[ex_i] - t_ns[exec_i]) / 1e6,
            "ofi_value": float(ofi_vals[i]) if not np.isnan(ofi_vals[i])
            else np.nan,
            "regime_spread": reg_spread[i],
            "regime_vol": reg_vol[i],
            "regime_tod": reg_tod[i],
        })
        open_until_ns = t_ns[ex_i]
        fun["longs" if direction > 0 else "shorts"] += 1

    ledger = pd.DataFrame(rows, columns=LEDGER_COLUMNS)
    npred = max(fun["rows_predictable"], 1)
    span_ms = float(t_ns.max() - t_ns.min()) / 1e6 if n else 0.0
    fun["mean_abs_prediction"] = abs_pred_sum / npred
    fun["max_abs_prediction"] = abs_pred_max
    fun["mean_cost_threshold"] = cost_thresh_sum / npred
    # The number that explains an empty ledger: forecast size relative to the
    # edge it must clear before trading is rational.
    fun["mean_pred_over_cost"] = (
        fun["mean_abs_prediction"] / fun["mean_cost_threshold"]
        if fun["mean_cost_threshold"] > 0 else np.nan)
    fun["time_in_market_frac"] = (
        float(ledger["holding_period_ms"].sum()) / span_ms
        if span_ms > 0 and len(ledger) else 0.0)
    metrics = {**compute_metrics(ledger), **fun}
    metrics.update(_pnl_in_units(ledger, cost_cfg.tick_size))
    return BacktestResult(ledger=ledger, metrics=metrics, notes=notes)


def _pnl_in_units(ledger: pd.DataFrame, tick_size: float) -> Dict[str, float]:
    """Per-trade P&L in ticks and bps.

    Real per-trade edges live at a fraction of a tick, where price units round
    to 0.00000 and say nothing.
    """
    if ledger.empty:
        return {"net_per_trade": np.nan, "net_per_trade_ticks": np.nan,
                "net_per_trade_bps": np.nan, "gross_per_trade": np.nan,
                "gross_per_trade_ticks": np.nan, "cost_per_trade": np.nan,
                "cost_per_trade_ticks": np.nan}
    net = float(ledger["net_pnl"].mean())
    gross = float(ledger["gross_mid_pnl"].mean())
    cost = float((ledger["spread_cost"] + ledger["fees"]
                  + ledger["slippage_cost"] + ledger["impact"]).mean())
    ref_px = float(pd.to_numeric(ledger["entry_price"], errors="coerce").mean())
    tick = tick_size if tick_size > 0 else np.nan
    return {
        "net_per_trade": net,
        "net_per_trade_ticks": net / tick,
        "net_per_trade_bps": (net / ref_px * 1e4) if ref_px > 0 else np.nan,
        "gross_per_trade": gross,
        "gross_per_trade_ticks": gross / tick,
        "cost_per_trade": cost,
        "cost_per_trade_ticks": cost / tick,
    }


def compute_metrics(ledger: pd.DataFrame) -> Dict[str, float]:
    """Aggregate performance metrics from a trade ledger."""
    n = len(ledger)
    if n == 0:
        return {"trade_count": 0, "gross_pnl": 0.0, "net_pnl": 0.0,
                "total_costs": 0.0, "hit_rate": np.nan, "avg_pnl": np.nan,
                "sharpe_like": np.nan, "max_drawdown": 0.0, "turnover": 0.0}
    net = ledger["net_pnl"].to_numpy()
    gross = ledger["gross_mid_pnl"].to_numpy()
    costs = (ledger["spread_cost"] + ledger["fees"] + ledger["impact"]).sum()
    cum = np.cumsum(net)
    running_max = np.maximum.accumulate(cum)
    drawdown = cum - running_max
    turnover = float((ledger["size"] * ledger["entry_price"]).sum())
    sharpe = float(net.mean() / net.std(ddof=1)) if n > 1 and net.std() > 0 \
        else np.nan
    return {
        "trade_count": n,
        "gross_pnl": float(gross.sum()),
        "net_pnl": float(net.sum()),
        "total_costs": float(costs),
        "hit_rate": float(np.mean(net > 0)),
        "avg_pnl": float(net.mean()),
        "sharpe_like": sharpe,
        "max_drawdown": float(drawdown.min()),
        "turnover": turnover,
    }
