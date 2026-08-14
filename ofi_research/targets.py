"""Future-midprice targets (section 11).

Targets look STRICTLY into the future (timestamp > t) and never cross a segment
boundary (new day / halt / gap), matching the feature reset logic. For clock
horizons on irregular data we take the first quote at or after ``t + H`` within
the same segment and record the realized horizon; observations whose realized
horizon deviates from the request beyond tolerance are rejected (target = NaN).

Columns created per horizon ``H`` (tagged ``ms<H>`` or ``ev<k>``):
    future_mid_change_<tag>   = m_{t+H} - m_t
    future_log_return_<tag>   = log(m_{t+H}) - log(m_t)
    future_direction_<tag>    = sign(change)
    future_class_<tag>        = up / unchanged / down (half-tick band)
    actual_horizon_ms_<tag>   = realized horizon in ms (clock horizons)
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

from .config import Config

logger = logging.getLogger(__name__)


def _clock_future_index(t_ns: np.ndarray, horizon_ms: float) -> np.ndarray:
    """For each i, index j = first k with t[k] >= t[i] + H (within array).

    Returns -1 where no such future observation exists in the segment.
    """
    n = len(t_ns)
    target = t_ns + int(horizon_ms * 1_000_000)  # ms -> ns
    # timestamps are sorted within a segment
    j = np.searchsorted(t_ns, target, side="left")
    j = np.where(j >= n, -1, j)
    return j


def _event_future_index(n: int, k: int) -> np.ndarray:
    j = np.arange(n) + k
    j = np.where(j >= n, -1, j)
    return j


def add_targets(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Add all clock- and event-horizon targets. Requires ``segment_id`` and
    ``midprice`` (from :func:`features.build_features`)."""
    tcfg = config.targets
    eps = config.data.epsilon
    df = df.copy()
    mid = df["midprice"].to_numpy(dtype="float64")
    t_ns = df["timestamp"].astype("int64").to_numpy()
    tick = config.features.tick_size
    band = tcfg.unchanged_band_ticks * tick

    seg_indices = {seg: np.sort(idx)
                   for seg, idx in df.groupby("segment_id").indices.items()}

    def finalize(tag: str, fut_idx_global: np.ndarray,
                 realized_ms: np.ndarray = None) -> None:
        valid = fut_idx_global >= 0
        fm = np.full(len(df), np.nan)
        fm[valid] = mid[fut_idx_global[valid]]
        change = fm - mid
        df[f"future_mid_change_{tag}"] = change
        with np.errstate(divide="ignore", invalid="ignore"):
            df[f"future_log_return_{tag}"] = np.log(
                np.clip(fm, eps, None)) - np.log(np.clip(mid, eps, None))
        df[f"future_direction_{tag}"] = np.sign(change)
        cls = np.where(change > band, "up",
                       np.where(change < -band, "down", "unchanged"))
        cls = np.where(np.isnan(change), None, cls)
        df[f"future_class_{tag}"] = cls
        if realized_ms is not None:
            df[f"actual_horizon_ms_{tag}"] = realized_ms

    # clock horizons
    for H in tcfg.clock_horizons_ms:
        tag = f"ms{int(H)}"
        fut = np.full(len(df), -1, dtype="int64")
        realized = np.full(len(df), np.nan)
        for seg, idx in seg_indices.items():
            local = _clock_future_index(t_ns[idx], H)
            has = local >= 0
            g = np.full(len(idx), -1, dtype="int64")
            g[has] = idx[local[has]]
            fut[idx] = g
            r = np.full(len(idx), np.nan)
            r[has] = (t_ns[idx[local[has]]] - t_ns[idx[has]]) / 1e6
            realized[idx] = r
        # tolerance rejection
        tol = max(tcfg.horizon_abs_tolerance_ms, tcfg.horizon_tolerance_frac * H)
        reject = np.abs(realized - H) > tol
        fut[reject] = -1
        realized[reject] = np.nan
        finalize(tag, fut, realized)
        n_ok = int((fut >= 0).sum())
        logger.info("Target %s: %d usable (%.1f%%)", tag, n_ok,
                    100.0 * n_ok / max(len(df), 1))

    # event horizons — RESEARCH DIAGNOSTIC ONLY (P0.8).
    # `j = i + k` is strictly later in canonical-event order, but the elapsed
    # clock time may be zero: several native events can share a ts_recv when
    # they arrive in one packet. Such a target is legitimate for measuring
    # information content in event time; it is NOT executable, and it must not
    # be fed to the backtest. `actual_horizon_ms_ev{k}` records the realized
    # elapsed time so the non-executable fraction is visible rather than
    # assumed away.
    for k in tcfg.event_horizons:
        tag = f"ev{k}"
        fut = np.full(len(df), -1, dtype="int64")
        realized = np.full(len(df), np.nan)
        for seg, idx in seg_indices.items():
            local = _event_future_index(len(idx), k)
            has = local >= 0
            g = np.full(len(idx), -1, dtype="int64")
            g[has] = idx[local[has]]
            fut[idx] = g
            r = np.full(len(idx), np.nan)
            r[has] = (t_ns[idx[local[has]]] - t_ns[idx[has]]) / 1e6
            realized[idx] = r
        finalize(tag, fut, realized)
        ok = realized[~np.isnan(realized)]
        if len(ok):
            n_zero = int((ok <= 0).sum())
            logger.info(
                "Target %s (event-time DIAGNOSTIC): median elapsed %.3f ms; "
                "%d/%d (%.2f%%) span zero clock time (not executable).",
                tag, float(np.median(ok)), n_zero, len(ok),
                100.0 * n_zero / len(ok))

    return df


# --- Execution-aligned targets (P0.8, family 3) ---
def add_execution_aligned_targets(df: pd.DataFrame, config: Config,
                                  horizon_ms: float,
                                  copy: bool = True) -> pd.DataFrame:
    """Targets measured from the first state a taker could actually reach.

    Entry is the first observation at or after ``t + total_latency``; the exit
    is the first observation at or after ``entry + horizon``. Both stay inside
    the segment. This is the only target family whose P&L interpretation is
    honest, because it never assumes action on information that had not yet
    arrived and been processed.

    Two ordering rules matter once ``ts_recv`` has ties (P0.5), and they are
    what keep "honest" from being merely a claim:

    * The entry index is always strictly later in native order than the row
      the signal came from. Searching by timestamp alone would return the
      first member of a tied batch, which can sit *earlier* than the signal.
    * With ``execution.no_fill_within_signal_batch`` the entry must carry a
      strictly later ``ts_recv``: records captured in one packet all became
      available at the same instant, so none of them can execute another.
    """
    # ``copy=False`` is for callers that own the frame outright and are adding
    # every horizon in a loop: on a real session this function is invoked once
    # per clock horizon, and copying a multi-gigabyte frame eight times over
    # dominates both runtime and peak memory while producing nothing.
    df = df.copy() if copy else df
    tag = f"ms{int(horizon_ms)}"
    total_latency_ms = config.costs.total_latency_ms()
    lat_ns = int(total_latency_ms * 1_000_000)
    batch_guard = config.execution.no_fill_within_signal_batch
    t_ns = df["timestamp"].astype("int64").to_numpy()
    mid = df["midprice"].to_numpy(dtype="float64")

    entry = np.full(len(df), -1, dtype="int64")
    exit_ = np.full(len(df), -1, dtype="int64")

    for _, idx in df.groupby("segment_id").indices.items():
        idx = np.sort(idx)
        ts = t_ns[idx]
        target = ts + lat_ns
        if batch_guard:
            target = np.maximum(target, ts + 1)
        e_local = np.searchsorted(ts, target, side="left")
        # never at or before the signal record itself, whatever the ties
        e_local = np.maximum(e_local, np.arange(len(idx)) + 1)
        e_ok = e_local < len(idx)
        e_glob = np.full(len(idx), -1, dtype="int64")
        e_glob[e_ok] = idx[e_local[e_ok]]
        entry[idx] = e_glob

        x_local = np.full(len(idx), -1, dtype="int64")
        src = e_local[e_ok]
        tgt = ts[src] + int(horizon_ms * 1_000_000)
        x = np.searchsorted(ts, tgt, side="left")
        valid = x < len(idx)
        tmp = np.full(len(src), -1, dtype="int64")
        tmp[valid] = idx[x[valid]]
        x_local[e_ok] = tmp
        exit_[idx] = x_local

    ok = (entry >= 0) & (exit_ >= 0)
    change = np.full(len(df), np.nan)
    change[ok] = mid[exit_[ok]] - mid[entry[ok]]

    df[f"exec_entry_index_{tag}"] = entry
    df[f"exec_exit_index_{tag}"] = exit_
    delay = np.full(len(df), np.nan)
    delay[entry >= 0] = (t_ns[entry[entry >= 0]] - t_ns[entry >= 0]) / 1e6
    df[f"exec_entry_delay_ms_{tag}"] = delay
    df[f"future_mid_change_exec_{tag}"] = change

    logger.info("Execution-aligned target %s: total latency %.3f ms, "
                "%d/%d rows executable (%.1f%%).", tag, total_latency_ms,
                int(ok.sum()), len(df), 100.0 * ok.sum() / max(len(df), 1))
    return df


# --- Causality assertions (P0.8) ---
def assert_target_causality(df: pd.DataFrame, config: Config) -> None:
    """Raise if any target violates its stated ordering contract.

    Cheap enough to run on every real-data load, and it converts a class of
    silent corruption into an immediate failure.
    """
    t_ns = df["timestamp"].astype("int64").to_numpy()
    seg = df["segment_id"].to_numpy()

    # clock targets: realized horizon strictly positive and at least H
    for H in config.targets.clock_horizons_ms:
        col = f"actual_horizon_ms_ms{int(H)}"
        if col not in df.columns:
            continue
        v = df[col].dropna().to_numpy()
        if len(v) and (v <= 0).any():
            raise AssertionError(
                f"{col}: clock target with non-positive realized horizon.")
        if len(v) and (v < H - 1e-9).any():
            raise AssertionError(
                f"{col}: clock target closer than the requested horizon {H} ms.")

    # event targets: strictly later in event order, same segment. Equal
    # timestamps are ALLOWED here and only here — that is the whole point of an
    # event-time diagnostic — but the event order key must still advance.
    for k in config.targets.event_horizons:
        col = f"future_mid_change_ev{k}"
        if col not in df.columns:
            continue
        if int(k) < 1:
            raise AssertionError(
                f"event horizon {k} does not advance the event order key; "
                "an event target must look at least one canonical event ahead.")
        valid = df[col].notna().to_numpy()
        i = np.flatnonzero(valid)
        j = i + k
        if len(i) and (j >= len(df)).any():
            raise AssertionError(f"ev{k}: target index past end of frame.")
        if len(i) and (seg[j] != seg[i]).any():
            raise AssertionError(f"ev{k}: target crosses a segment boundary.")
        if len(i) and (t_ns[j] < t_ns[i]).any():
            raise AssertionError(f"ev{k}: target timestamp precedes feature.")

    # execution-aligned targets: entry never before the latency deadline, never
    # before the signal row, and never inside the signal's own capture batch
    lat_ms = config.costs.total_latency_ms()
    batch_guard = config.execution.no_fill_within_signal_batch
    row_index = np.arange(len(df))
    for H in config.targets.clock_horizons_ms:
        col = f"exec_entry_index_{f'ms{int(H)}'}"
        if col not in df.columns:
            continue
        e = df[col].to_numpy()
        ok = e >= 0
        if ok.any():
            delay_ms = (t_ns[e[ok]] - t_ns[ok]) / 1e6
            if (delay_ms < lat_ms - 1e-6).any():
                raise AssertionError(
                    f"{col}: simulated entry earlier than total latency "
                    f"{lat_ms} ms.")
            if (e[ok] <= row_index[ok]).any():
                raise AssertionError(
                    f"{col}: simulated entry at or before the signal record "
                    "in native order (tied ts_recv mis-handled).")
            if batch_guard and (delay_ms <= 0).any():
                raise AssertionError(
                    f"{col}: simulated entry shares the signal's ts_recv; a "
                    "record cannot execute another record of its own packet.")
            if (seg[e[ok]] != seg[ok]).any():
                raise AssertionError(f"{col}: entry crosses a segment.")

    logger.info("Target causality assertions passed (clock, event, execution).")


def target_columns(config: Config, kind: str = "mid_change") -> List[str]:
    """Return the list of target column names for a given kind."""
    tags = ([f"ms{int(H)}" for H in config.targets.clock_horizons_ms]
            + [f"ev{k}" for k in config.targets.event_horizons])
    prefix = {
        "mid_change": "future_mid_change_",
        "log_return": "future_log_return_",
        "direction": "future_direction_",
        "class": "future_class_",
    }[kind]
    return [prefix + t for t in tags]


def all_horizon_tags(config: Config) -> List[str]:
    return ([f"ms{int(H)}" for H in config.targets.clock_horizons_ms]
            + [f"ev{k}" for k in config.targets.event_horizons])
