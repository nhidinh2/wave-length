"""Make a real multi-day sample fit in memory, without changing what it means.

One session of a busy Nasdaq name is ~3.5M events and ~5 GB once features
exist, and the walk-forward holds train + validation + test resident at once.
Three levers, cheapest first:

1. :func:`compact_dtypes` — representation only. No column disappears and no
   value changes beyond float32 rounding, which price levels are spared.
2. :func:`prune_columns` — drops intermediates nothing reads. The keep-list is
   derived from the model feature groups, so adding a feature cannot silently
   prune it away.
3. :func:`sample_decision_rows` — the only lever that removes observations, and
   the only one that is a research decision rather than an implementation
   detail. It defines a decision clock and must be reported as such.

Order matters: sampling runs AFTER targets, so every retained row's target was
resolved against the complete event stream, not the sampled grid.
"""

from __future__ import annotations

import logging
from typing import Dict, Set

import numpy as np
import pandas as pd

from .config import Config
from .dbn import INT_PRICE_SUFFIX

logger = logging.getLogger(__name__)

#: Columns whose ABSOLUTE magnitude is a price and which therefore stay
#: float64. At ~$100 a float32 ulp is ~7.6e-6, which is 0.08% of a penny tick —
#: tolerable once, corrosive after a day of accumulated P&L arithmetic.
#: Differences (spreads, mid changes) are small in magnitude and safe.
PRICE_LEVEL_PREFIXES = ("bid_price_", "ask_price_", "trade_price", "midprice",
                        "log_midprice", "vwap")

#: Always retained: identity, causal clock, integrity, and the book the taker
#: execution simulator walks.
ESSENTIAL_COLUMNS = (
    "instrument", "timestamp", "ts_recv", "ts_event", "session_date",
    "sequence", "canonical_event_id", "raw_row_id", "segment_id",
    "is_segment_start", "integrity_break", "gap_ms", "long_quiet_gap",
    "event_window_elapsed_ms", "midprice", "spread", "relative_spread",
    "book_clear", "halt_boundary", "F_SNAPSHOT", "F_MAYBE_BAD_BOOK",
    "F_BAD_TS_RECV", "in_continuous_session", "trade_size",
    "signed_trade_volume", "n_trades",
    # Per-event signed flow and aggressor side, for the passive study. The
    # rolling signedvol_* windows cannot substitute: a trailing sum says how
    # much net flow arrived recently, not whether THIS event was a buy or sell.
    "signed_volume_increment", "trade_direction",
)


def _is_price_level(col: str) -> bool:
    return col.startswith(PRICE_LEVEL_PREFIXES)


def compact_dtypes(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Shrink representation without dropping information.

    The fixed-point ``*_int`` price copies are dropped here rather than never
    created: they exist so :func:`ofi_research.dbn.assert_prices_converted_once`
    can prove at load time that each price column was divided exactly once
    (P0.6). That proof is complete before features are built, and nothing
    downstream reads them.
    """
    scfg = config.sampling
    out = df
    before = out.memory_usage(deep=True).sum()

    if scfg.drop_fixed_price_integers:
        int_cols = [c for c in out.columns if c.endswith(INT_PRICE_SUFFIX)]
        if int_cols:
            out = out.drop(columns=int_cols)
            logger.info("Released %d fixed-point integer price column(s); the "
                        "converted-exactly-once assertion already ran at load.",
                        len(int_cols))

    if scfg.categorical_strings:
        obj = [c for c in out.columns if out[c].dtype == object]
        for c in obj:
            # only worth it when the column is genuinely low-cardinality
            nunique = out[c].nunique(dropna=False)
            if nunique <= max(1024, int(0.01 * len(out))):
                out[c] = out[c].astype("category")
        if obj:
            logger.info("Converted %d object column(s) to category.", len(obj))

    if scfg.downcast_float32:
        f64 = [c for c in out.columns
               if out[c].dtype == "float64" and not _is_price_level(c)]
        for c in f64:
            out[c] = out[c].astype("float32")
        if f64:
            logger.info("Downcast %d float64 column(s) to float32; %d price-"
                        "level column(s) kept at float64.", len(f64),
                        sum(1 for c in out.columns if _is_price_level(c)))

    after = out.memory_usage(deep=True).sum()
    logger.info("Dtype compaction: %.2f GB -> %.2f GB (%.0f%% saved).",
                before / 1e9, after / 1e9,
                100.0 * (1 - after / max(before, 1)))
    return out


def required_columns(df: pd.DataFrame, config: Config) -> Set[str]:
    """Everything anything downstream reads, derived not hardcoded.

    Built from the model feature groups, the OFI variants the diagnostics
    iterate, the depth ladder the execution simulator walks, and every target
    column. A feature added to a group is therefore retained automatically.
    """
    from .models import model_feature_sets
    from .targets import all_horizon_tags

    keep: Set[str] = {c for c in ESSENTIAL_COLUMNS if c in df.columns}

    # model inputs, across the whole ladder / ablations / L2 variants
    for feats in model_feature_sets(config).values():
        keep.update(f for f in feats if f in df.columns)

    # depth ladder for taker execution (P0.9)
    n = max(config.columns.n_depth_levels, config.execution.depth_levels)
    for k in range(1, n + 1):
        for pat in (f"bid_price_{k}", f"ask_price_{k}",
                    f"bid_size_{k}", f"ask_size_{k}"):
            if pat in df.columns:
                keep.add(pat)

    # targets, realized horizons, and execution-aligned indices
    for tag in all_horizon_tags(config):
        for pat in (f"future_mid_change_{tag}", f"future_log_return_{tag}",
                    f"future_direction_{tag}", f"future_class_{tag}",
                    f"actual_horizon_ms_{tag}",
                    f"future_mid_change_exec_{tag}",
                    f"exec_entry_index_{tag}", f"exec_exit_index_{tag}",
                    f"exec_entry_delay_ms_{tag}"):
            if pat in df.columns:
                keep.add(pat)

    # OFI variants, regime inputs, and the per-window elapsed-time columns the
    # horizon/window realization table reports (P0.3 — an event window is not a
    # fixed amount of clock time, and that has to stay visible).
    for c in df.columns:
        if (c.startswith(("OFI", "nOFI", "zOFI", "signedvol", "regime_",
                          "trailing_", "event_window_elapsed_ms"))
                or c in ("depth_imbalance_L1", "delta_spread",
                         "total_L1_depth", "total_L2_depth",
                         "aggressive_trade_intensity_imbalance",
                         "normalized_aggressive_trade_intensity_imbalance")):
            keep.add(c)
    return keep


def prune_columns(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Drop intermediates nothing downstream reads."""
    if not config.sampling.prune_columns:
        return df
    keep = required_columns(df, config)
    drop = [c for c in df.columns if c not in keep]
    if not drop:
        return df
    before = df.memory_usage(deep=True).sum()
    out = df.drop(columns=drop)
    after = out.memory_usage(deep=True).sum()
    logger.info("Pruned %d/%d column(s), %.2f GB -> %.2f GB. Dropped: %s",
                len(drop), df.shape[1], before / 1e9, after / 1e9,
                ",".join(sorted(drop))[:300])
    return out


def sample_decision_rows(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Retain one row per decision interval — a DECISION CLOCK, not a shortcut.

    This does NOT subsample the data features are built from: because it runs
    last, every trailing window, OFI increment and target on a retained row was
    computed over the complete event stream.

    It restricts when the strategy may ACT. At 100 ms the model sees the book as
    of each bucket's first event and can enter only there — a real, reportable
    property of a 100 ms decision cycle, not an approximation of one acting on
    every event. Report it alongside any P&L: a shorter interval is a different
    and more demanding strategy.

    The first row of every segment is always retained, so no OFI chain is
    re-based by sampling.
    """
    interval = config.sampling.decision_interval_ms
    if not interval or interval <= 0:
        return df
    if df.empty:
        return df

    n0 = len(df)
    t_ns = df["timestamp"].astype("int64").to_numpy()
    bucket = t_ns // int(interval * 1_000_000)

    seg = df["segment_id"].to_numpy()
    # first row of each (segment, bucket) pair, in existing row order
    is_first = np.empty(len(df), dtype=bool)
    is_first[0] = True
    is_first[1:] = (bucket[1:] != bucket[:-1]) | (seg[1:] != seg[:-1])

    out = df.loc[is_first].reset_index(drop=True)
    logger.info(
        "Decision clock %.0f ms: retained %d/%d rows (%.2f%%). Features and "
        "targets were computed on ALL %d events; the strategy may only act at "
        "these instants.", interval, len(out), n0, 100.0 * len(out) / n0, n0)
    return out


def memory_report(df: pd.DataFrame, label: str = "") -> Dict[str, float]:
    """Per-dtype footprint, for the run log and the report."""
    mem = df.memory_usage(deep=True)
    total = float(mem.sum())
    logger.info("Frame%s: %d rows x %d cols, %.2f GB",
                f" ({label})" if label else "", len(df), df.shape[1],
                total / 1e9)
    return {"rows": len(df), "columns": df.shape[1], "bytes": total}
