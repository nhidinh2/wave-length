"""Schema inspection (section 3) and data cleaning (section 4).

``inspect_schema`` produces the report the brief asks for BEFORE any modeling:
columns, dtypes, counts, timestamp range/resolution, missing values, duplicate
timestamps, sortedness, instrument count, time-gap distribution, crossed/locked
books, negative sizes and possible sequence problems. It prints and returns a
structured report; it does not guess or repair anything.

``clean_data`` applies the configurable cleaning rules and returns
``(clean_df, removal_report)`` where the report accounts for every removed row
by reason, percentage, and affected dates/instruments.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import Config

logger = logging.getLogger(__name__)


@dataclass
class SchemaReport:
    n_rows: int
    columns: List[str]
    dtypes: Dict[str, str]
    ts_min: Optional[str]
    ts_max: Optional[str]
    ts_resolution_ms: Optional[float]
    missing_counts: Dict[str, int]
    n_duplicate_rows: int
    n_duplicate_timestamps: int
    is_sorted: bool
    n_instruments: int
    n_days: int
    gap_ms_describe: Dict[str, float]
    n_crossed: int
    n_locked: int
    n_nonpositive_sizes: int
    has_level2: bool
    has_trades: bool
    has_trade_side: bool
    has_event_type: bool
    notes: List[str] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        flat = {}
        for k, v in self.__dict__.items():
            if isinstance(v, (dict, list)):
                flat[k] = repr(v)
            else:
                flat[k] = v
        return pd.DataFrame({"field": list(flat.keys()),
                             "value": list(flat.values())})


def _timestamp_resolution_ms(ts: pd.Series) -> Optional[float]:
    if len(ts) < 2:
        return None
    diffs = ts.sort_values().diff().dropna()
    pos = diffs[diffs > pd.Timedelta(0)]
    if pos.empty:
        return 0.0
    return float(pos.min() / pd.Timedelta(milliseconds=1))


def inspect_schema(df: pd.DataFrame, config: Config) -> SchemaReport:
    """Compute and log the section-3 data inspection report."""
    cols = list(df.columns)
    dtypes = {c: str(df[c].dtype) for c in cols}
    notes: List[str] = []

    ts = df["timestamp"]
    ts_valid = ts.dropna()
    ts_min = str(ts_valid.min()) if not ts_valid.empty else None
    ts_max = str(ts_valid.max()) if not ts_valid.empty else None

    # per-instrument gaps (avoid cross-instrument diffs)
    gap_ms = (
        df.groupby("instrument")["timestamp"].diff()
        / pd.Timedelta(milliseconds=1)
    ).dropna()
    if gap_ms.empty:
        gap_desc = {}
    else:
        q = gap_ms.quantile([0.5, 0.9, 0.99]).to_dict()
        gap_desc = {
            "min": float(gap_ms.min()),
            "median": float(q.get(0.5, np.nan)),
            "p90": float(q.get(0.9, np.nan)),
            "p99": float(q.get(0.99, np.nan)),
            "max": float(gap_ms.max()),
            "n_negative": int((gap_ms < 0).sum()),
        }
        if gap_desc["n_negative"] > 0:
            notes.append(
                f"{gap_desc['n_negative']} negative intra-instrument time gaps "
                "-> possible out-of-order events or sequence problems."
            )

    crossed = int((df["bid_price_1"] > df["ask_price_1"]).sum())
    locked = int((df["bid_price_1"] == df["ask_price_1"]).sum())
    size_cols = [c for c in ["bid_size_1", "ask_size_1", "bid_size_2",
                             "ask_size_2", "trade_size"] if c in df.columns]
    nonpos = int(
        sum((df[c] <= 0).sum() for c in size_cols
            if df[c].notna().any())
    )

    is_sorted = bool(
        df.groupby("instrument")["timestamp"]
        .apply(lambda s: s.is_monotonic_increasing).all()
    )
    if not is_sorted:
        notes.append("Data is NOT sorted by timestamp within instrument.")

    dup_rows = int(df.duplicated().sum())
    dup_ts = int(df.duplicated(subset=["instrument", "timestamp"]).sum())

    report = SchemaReport(
        n_rows=len(df),
        columns=cols,
        dtypes=dtypes,
        ts_min=ts_min,
        ts_max=ts_max,
        ts_resolution_ms=_timestamp_resolution_ms(ts_valid),
        missing_counts={c: int(df[c].isna().sum()) for c in cols},
        n_duplicate_rows=dup_rows,
        n_duplicate_timestamps=dup_ts,
        is_sorted=is_sorted,
        n_instruments=int(df["instrument"].nunique()),
        n_days=int(df["session_date"].nunique()),
        gap_ms_describe=gap_desc,
        n_crossed=crossed,
        n_locked=locked,
        n_nonpositive_sizes=nonpos,
        has_level2=config.columns.has_level2()
        and df["bid_price_2"].notna().any(),
        has_trades=config.columns.has_trades() and df["trade_size"].notna().any(),
        has_trade_side=config.columns.has_trade_side()
        and df["trade_side"].notna().any(),
        has_event_type=config.columns.has_event_type()
        and df["event_type"].notna().any(),
        notes=notes,
    )

    logger.info("Schema: %d rows, %d instruments, %d days, ts %s..%s",
                report.n_rows, report.n_instruments, report.n_days,
                report.ts_min, report.ts_max)
    logger.info("L2=%s trades=%s side=%s events=%s | crossed=%d locked=%d "
                "nonpos_sizes=%d dup_rows=%d dup_ts=%d",
                report.has_level2, report.has_trades, report.has_trade_side,
                report.has_event_type, report.n_crossed, report.n_locked,
                report.n_nonpositive_sizes, report.n_duplicate_rows,
                report.n_duplicate_timestamps)
    for n in notes:
        logger.warning("SCHEMA NOTE: %s", n)
    return report


#: Reasons that are pure ingestion artifacts. Removing one of these cannot have
#: skipped a real book-state transition, so the OFI chain continues across it.
#: EVERY other reason is treated as potentially state-changing (P0.4).
BENIGN_REMOVAL_REASONS = frozenset({"duplicate_row"})

#: Bookkeeping columns that are unique per row by construction and therefore
#: say nothing about whether two records describe the same market event. They
#: are excluded from the duplicate identity: leaving ``raw_row_id`` in would
#: make every row unique and silently disable the benign-duplicate rule
#: altogether.
#: A day must span at least this multiple of the total trim window before the
#: open/close trim is applied to it. Guards fixtures and partial extracts.
TRIM_MIN_SPAN_FACTOR = 4.0

IDENTITY_EXCLUDED_COLUMNS = (
    "raw_row_id", "canonical_event_id", "src_row_first", "src_row_last",
    "_source_file", "_source_index", "_drop_reason",
)


def clean_data(df: pd.DataFrame, config: Config
               ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply configurable cleaning; return (clean_df, removal_report).

    Rows are dropped for exactly one *first-matching* reason so counts sum
    correctly. Locked books are kept by default (flagged, not removed) unless
    ``drop_locked_books`` is set.

    Two rules that matter more than they look (P0.1 / P0.4):

    * **Timestamp ties are never a removal reason.** Genuine book updates can
      share a ``ts_recv`` (one packet). Dropping the second would make the NEXT
      OFI increment difference across a skipped book state — silently wrong.
      Only full-row duplicates (ingestion artifacts) are removed.
    * **A non-benign removal breaks the chain.** The next retained row is marked
      ``integrity_break`` so :func:`features.add_segments` starts a new segment
      rather than differencing across the gap.
    """
    dcfg = config.data
    n0 = len(df)
    df = df.copy()
    df["_drop_reason"] = ""

    def mark(mask: pd.Series, reason: str) -> None:
        newly = mask & (df["_drop_reason"] == "")
        df.loc[newly, "_drop_reason"] = reason

    # Exact duplicate rows: identical across every column that describes the
    # market (row-unique bookkeeping columns excluded), i.e. a re-ingest.
    # This is the only benign removal.
    identity = [c for c in df.columns if c not in IDENTITY_EXCLUDED_COLUMNS]
    mark(df.duplicated(subset=identity), "duplicate_row")
    # NOTE: there is deliberately NO `duplicate_timestamp` rule here.
    # invalid book values
    mark(df["bid_price_1"].isna() | df["ask_price_1"].isna(), "missing_l1_price")
    if dcfg.drop_crossed_books:
        mark(df["bid_price_1"] > df["ask_price_1"], "crossed_book")
    if dcfg.drop_locked_books:
        mark(df["bid_price_1"] == df["ask_price_1"], "locked_book")
    if dcfg.drop_nonpositive_sizes:
        mark((df["bid_size_1"] <= 0) | (df["ask_size_1"] <= 0),
             "nonpositive_size")

    # stale-quote runs (optional)
    if dcfg.stale_quote_max_repeat is not None:
        key = (df["bid_price_1"].astype(str) + "|" + df["ask_price_1"].astype(str)
               + "|" + df["bid_size_1"].astype(str) + "|"
               + df["ask_size_1"].astype(str))
        grp = (key != key.shift()).cumsum()
        run_len = df.groupby(["instrument", grp]).cumcount()
        mark(run_len > dcfg.stale_quote_max_repeat, "stale_quote")

    # Open/close trimming — excludes the auction prints that mbp-10 carries
    # inline at the session edges.
    #
    # Applied only to days long enough for the rule to mean something. The trim
    # says "cut the auction minutes off a trading session"; a fixture or a
    # partial extract spanning less than TRIM_MIN_SPAN_FACTOR x the trim window
    # is not such a session, and trimming it would remove most or all of the
    # data. Skipping loudly beats returning an empty frame that fails several
    # functions later with an unrelated error.
    trim_total = (dcfg.trim_open_minutes or 0.0) + (dcfg.trim_close_minutes or 0.0)
    if trim_total > 0:
        by_day = df.groupby(["instrument", "session_date"])["timestamp"]
        day_start = by_day.transform("min")
        day_end = by_day.transform("max")
        span_min = (day_end - day_start) / pd.Timedelta(minutes=1)
        long_enough = span_min >= TRIM_MIN_SPAN_FACTOR * trim_total
        n_short = int((~long_enough).sum())
        if n_short:
            short_days = sorted({str(d) for d in
                                 df.loc[~long_enough, "session_date"]})
            logger.warning(
                "Session trimming SKIPPED for %d row(s) on %s: the span is "
                "shorter than %.0fx the %.1f-minute trim window, so these are "
                "not full trading sessions. Auction prints, if any, are NOT "
                "excluded there.", n_short, ",".join(short_days)[:120],
                TRIM_MIN_SPAN_FACTOR, trim_total)
        if dcfg.trim_open_minutes:
            mark(long_enough & (df["timestamp"] < day_start
                 + pd.Timedelta(minutes=dcfg.trim_open_minutes)), "open_trim")
        if dcfg.trim_close_minutes:
            mark(long_enough & (df["timestamp"] > day_end
                 - pd.Timedelta(minutes=dcfg.trim_close_minutes)), "close_trim")

    # ---- P0.4: break the OFI chain after any non-benign removal ----
    reason = df["_drop_reason"]
    dropped = reason != ""
    nonbenign = dropped & ~reason.isin(BENIGN_REMOVAL_REASONS)
    retained_mask = ~dropped

    removed = df[dropped]
    clean = df[retained_mask].drop(columns=["_drop_reason"]).copy()

    # For each retained row: did a non-benign removal occur since the previous
    # retained row? If so this row begins a new segment.
    cum_nonbenign = nonbenign.cumsum()[retained_mask]
    if len(cum_nonbenign):
        prior = cum_nonbenign.shift(1).fillna(0)
        integrity_break = (cum_nonbenign - prior) > 0
    else:
        integrity_break = pd.Series([], dtype=bool)
    existing = (clean["integrity_break"].fillna(False).astype(bool).to_numpy()
                if "integrity_break" in clean.columns
                else np.zeros(len(clean), dtype=bool))
    clean["integrity_break"] = existing | integrity_break.to_numpy().astype(bool)
    clean = clean.reset_index(drop=True)

    n_breaks = int(clean["integrity_break"].sum())
    if n_breaks:
        logger.info("%d retained row(s) marked integrity_break after a "
                    "non-benign removal; their OFI increment will be 0.",
                    n_breaks)

    # build removal report
    if removed.empty:
        report = pd.DataFrame(
            columns=["reason", "n_removed", "pct_removed",
                     "dates", "instruments"])
    else:
        recs = []
        for reason, g in removed.groupby("_drop_reason"):
            recs.append({
                "reason": reason,
                "n_removed": len(g),
                "pct_removed": 100.0 * len(g) / n0,
                "dates": ",".join(sorted({str(d) for d in g["session_date"]}))[:200],
                "instruments": ",".join(sorted(map(str, g["instrument"].unique())))[:200],
            })
        report = pd.DataFrame(recs).sort_values("n_removed", ascending=False)

    total_removed = n0 - len(clean)
    logger.info("Cleaning removed %d/%d rows (%.3f%%)", total_removed, n0,
                100.0 * total_removed / max(n0, 1))
    if n0 and clean.empty:
        # Returning an empty frame lets the failure surface far downstream as
        # something unrelated (an .iloc[0] on an empty index, say). Name the
        # cause here instead, with the reason breakdown that explains it.
        breakdown = ", ".join(f"{r.reason}={r.n_removed}"
                              for r in report.itertuples())
        raise ValueError(
            f"Cleaning removed ALL {n0} rows ({breakdown}). Check the cleaning "
            "toggles in DataConfig against this dataset before rerunning.")
    return clean, report
