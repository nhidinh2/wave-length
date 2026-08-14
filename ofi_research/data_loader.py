"""Data loading, column canonicalization, and a labeled synthetic generator.

Deliberately conservative: it renames mapped columns to canonical names but
never guesses a missing required field (it raises with a mapping template),
parses timestamps per :class:`DataConfig`, sorts by (instrument, timestamp,
sequence), and NEVER back-fills quotes — forward state is carried only where a
real book would carry it, and only from the past.

``make_synthetic_data`` plants an adjustable predictive OFI relationship in a
structurally realistic book so the pipeline can be exercised end to end. Any
Config built from it must set ``synthetic_data=True`` so every artifact is
stamped SYNTHETIC.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from .config import ColumnMapping, Config, DataConfig

logger = logging.getLogger(__name__)

# canonical column names the rest of the pipeline relies on
CANONICAL_COLUMNS = [
    "timestamp", "instrument", "sequence",
    "bid_price_1", "ask_price_1", "bid_size_1", "ask_size_1",
    "bid_price_2", "ask_price_2", "bid_size_2", "ask_size_2",
    "trade_price", "trade_size", "trade_side", "event_type",
]


class MissingColumnsError(ValueError):
    """Raised when required source columns cannot be located."""


def _mapping_template(mapping: ColumnMapping) -> str:
    lines = ["", "Column mapping template (canonical <- source):"]
    for canon in CANONICAL_COLUMNS:
        src = getattr(mapping, canon, None)
        req = " [REQUIRED]" if canon in ColumnMapping.REQUIRED_FIELDS else ""
        lines.append(f"    {canon:14s} <- {src!r}{req}")
    return "\n".join(lines)


def _read_raw(cfg: DataConfig) -> pd.DataFrame:
    path = Path(cfg.data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"data_path does not exist: {cfg.data_path!r}. "
            "Set DataConfig.data_path to your dataset, or use "
            "make_synthetic_data() for a labeled synthetic run."
        )
    fmt = cfg.file_format
    if fmt == "auto":
        fmt = "parquet" if path.suffix.lower() in (".parquet", ".pq") else "csv"
    logger.info("Reading %s as %s", path, fmt)
    if fmt == "parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _parse_timestamp(s: pd.Series, cfg: DataConfig) -> pd.Series:
    if cfg.timestamp_is_epoch:
        unit = cfg.timestamp_unit or "ns"
        ts = pd.to_datetime(s, unit=unit, utc=True)
    else:
        ts = pd.to_datetime(s, utc=True)
    return ts


def canonicalize(df: pd.DataFrame, mapping: ColumnMapping,
                 cfg: DataConfig) -> pd.DataFrame:
    """Rename mapped columns to canonical names and validate required fields.

    Missing optional columns are created as all-NA so downstream code can rely
    on the columns existing while ``validation``/``features`` decide whether the
    corresponding feature family is available.
    """
    missing = [src for src in mapping.required_source_columns()
               if src not in df.columns]
    if missing:
        raise MissingColumnsError(
            "Required source columns not found: "
            f"{missing}. Present columns: {list(df.columns)}."
            + _mapping_template(mapping)
        )

    out = pd.DataFrame(index=df.index)
    # stamped from native file order, before any sort can permute ties (P0.1)
    out["raw_row_id"] = np.arange(len(df), dtype="int64")
    for canon in CANONICAL_COLUMNS:
        src = getattr(mapping, canon, None)
        if src is not None and src in df.columns:
            out[canon] = df[src].values
        else:
            out[canon] = np.nan

    out["timestamp"] = _parse_timestamp(out["timestamp"], cfg)

    if out["instrument"].isna().all():
        out["instrument"] = "INSTRUMENT_0"

    # session date used for day-boundary logic
    out["session_date"] = out["timestamp"].dt.tz_convert(cfg.timezone).dt.date

    return out


def sort_observations(df: pd.DataFrame) -> pd.DataFrame:
    """Stable sort by instrument, timestamp, sequence, then source order.

    ``raw_row_id`` is the final tie-breaker so that records sharing a timestamp
    (and even a sequence) keep the order they arrived in. That order is what
    makes the book state at each row well defined; a sort that leaves ties
    unbroken would let an equal-timestamp group be permuted, silently changing
    every OFI increment computed across it (P0.1).
    """
    df = df.copy()
    if "raw_row_id" not in df.columns:
        df["raw_row_id"] = np.arange(len(df), dtype="int64")
    keys = ["instrument", "timestamp"]
    if df["sequence"].notna().any():
        keys.append("sequence")
    keys.append("raw_row_id")
    return df.sort_values(keys, kind="mergesort").reset_index(drop=True)


def normalize_absolute_price_book(
    df: pd.DataFrame, n_levels: int = 2
) -> pd.DataFrame:
    """Convert an absolute-price book to rank-based L1/L2 columns.

    Some feeds represent the book by absolute price rather than rank: e.g. many
    ``(price, size, side)`` rows per timestamp. This helper expects such rows to
    already be widened; it is documented here as the extension point and is a
    no-op when the data is already rank-based (the common case). See README
    "Absolute-price books" for the full contract. The important microstructure
    consequence, spelled out in §7 of the brief: OFI must be computed on the
    *rank* series (best bid, 2nd-best bid, ...), so any absolute->rank mapping
    must happen HERE, before feature construction, and must never look ahead.
    """
    return df


def load_data(config: Config) -> pd.DataFrame:
    """Full load path: read -> canonicalize -> (abs->rank) -> sort."""
    raw = _read_raw(config.data)
    df = canonicalize(raw, config.columns, config.data)
    df = normalize_absolute_price_book(df)
    df = sort_observations(df)
    logger.info("Loaded %d rows, %d instrument(s), %d day(s)",
                len(df), df["instrument"].nunique(),
                df["session_date"].nunique())
    return df


# --- Synthetic data (clearly labeled) ---
def make_synthetic_data(
    n_days: int = 15,
    events_per_day: int = 4000,
    seed: int = 7,
    signal_strength: float = 0.35,
    tick_size: float = 0.01,
    start_price: float = 100.0,
) -> Tuple[pd.DataFrame, Config]:
    """Generate a labeled synthetic L2 book + trade stream.

    The midprice evolves as a random walk PLUS a component driven by a latent
    order-flow state; the same latent state drives book-size changes so that
    OFI carries genuine (tunable) predictive information about the NEXT few
    midprice moves. This is a *test fixture*, not market data.

    Returns (dataframe_in_canonical_schema, config_with_synthetic_flag_set).
    """
    rng = np.random.default_rng(seed)
    rows = []
    base_day = pd.Timestamp("2024-01-01", tz="UTC")

    for d in range(n_days):
        day_start = base_day + pd.Timedelta(days=d)
        # intraday timestamps: ~ every 100ms with jitter (Poisson-like arrivals).
        # Gaps stay well under gap_reset_ms so events form contiguous segments;
        # the session length grows naturally with the number of events.
        dt_ms = rng.exponential(100.0, size=events_per_day).clip(1, 800)
        t_ms = np.cumsum(dt_ms)
        timestamps = day_start + pd.to_timedelta(t_ms, unit="ms")

        mid = start_price
        latent = 0.0
        for i in range(events_per_day):
            # latent order-flow state: mean-reverting
            latent = 0.9 * latent + rng.normal(0, 1.0)
            # planted predictive OFI: expected next move ~ signal_strength*latent
            drift = signal_strength * latent * tick_size
            mid = mid + drift + rng.normal(0, 0.5 * tick_size)
            mid = max(mid, tick_size * 10)

            half_spread = tick_size * rng.integers(1, 4)
            bid1 = np.round((mid - half_spread) / tick_size) * tick_size
            ask1 = np.round((mid + half_spread) / tick_size) * tick_size
            if ask1 <= bid1:
                ask1 = bid1 + tick_size
            bid2 = bid1 - tick_size * rng.integers(1, 3)
            ask2 = ask1 + tick_size * rng.integers(1, 3)

            # sizes correlated with latent so OFI reflects the latent state
            base = 200.0
            bsz1 = max(1.0, base + 60 * latent + rng.normal(0, 40))
            asz1 = max(1.0, base - 60 * latent + rng.normal(0, 40))
            bsz2 = max(1.0, base + 30 * latent + rng.normal(0, 40))
            asz2 = max(1.0, base - 30 * latent + rng.normal(0, 40))

            # a trade fires sometimes; aggressor side leans with latent
            has_trade = rng.random() < 0.35
            if has_trade:
                p_buy = 1.0 / (1.0 + np.exp(-1.2 * latent))
                is_buy = rng.random() < p_buy
                trade_side = "B" if is_buy else "S"
                trade_price = ask1 if is_buy else bid1
                trade_size = float(rng.integers(1, 50))
                event_type = "market_buy" if is_buy else "market_sell"
            else:
                trade_side = np.nan
                trade_price = np.nan
                trade_size = np.nan
                event_type = rng.choice(
                    ["limit_bid", "limit_ask", "cancel_bid", "cancel_ask"]
                )

            rows.append((
                timestamps[i], "SYN", i + d * events_per_day,
                bid1, ask1, bsz1, asz1,
                bid2, ask2, bsz2, asz2,
                trade_price, trade_size, trade_side, event_type,
            ))

    df = pd.DataFrame(rows, columns=[
        "timestamp", "instrument", "sequence",
        "bid_price_1", "ask_price_1", "bid_size_1", "ask_size_1",
        "bid_price_2", "ask_price_2", "bid_size_2", "ask_size_2",
        "trade_price", "trade_size", "trade_side", "event_type",
    ])
    df["session_date"] = df["timestamp"].dt.date

    cfg = Config()
    cfg.data.data_path = "<SYNTHETIC>"
    cfg.synthetic_data = True
    cfg.label = "synthetic"
    cfg.features.tick_size = tick_size
    cfg.costs.tick_size = tick_size
    logger.warning("Generated SYNTHETIC data: %d rows across %d days. "
                   "Results from this data are NOT real.", len(df), n_days)
    return df, cfg
