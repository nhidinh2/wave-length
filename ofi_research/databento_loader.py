"""Databento ``mbp-10`` ingestion: raw records -> canonical native events.

Separate from :mod:`data_loader` on purpose. All vendor-specific semantics —
flags, actions, fixed-point prices, normalized-record grouping — live here, so
the OFI math never imports them and can be tested against hand-built books.

Pipeline:

1. :func:`read_raw`        — read in NATIVE FILE ORDER, stamp ``raw_row_id``.
2. :func:`prepare_raw`     — decode flags, convert prices once, derive the
                             exchange-local session date.
3. :func:`drop_ingestion_duplicates` — remove only byte-identical re-ingests,
                             using the full raw identity. A timestamp tie is
                             NEVER sufficient (P0.1).
4. :func:`build_canonical_events` — collapse consecutive normalized records
                             into native events on ``F_LAST`` (P0.2).
5. :func:`to_canonical_frame` — emit the column names the OFI pipeline expects.

Two facts drive the design:

* Databento may normalize ONE native Nasdaq message into SEVERAL records that
  share a ``sequence``. ``F_LAST`` marks the end of the normalized event.
* A ``Trade`` record does not itself change the book; the book update arrives
  on a sibling record of the same event. Summing trade size and taking the
  final book state must therefore happen at EVENT level, not record level.

Every assumption encoded here is listed in ``PILOT_ASSUMPTIONS`` and must be
checked against a one-day sample before a multi-day run.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import Config
from .dbn import (
    ACTION_CLEAR, ACTION_TRADE, F_BAD_TS_RECV, F_LAST, F_MAYBE_BAD_BOOK,
    F_SNAPSHOT, INT_PRICE_SUFFIX, SIDE_ASK, SIDE_BID,
    assert_plausible_prices, assert_prices_converted_once, audit_flags,
    decode_flags, infer_tick_size, to_dollars, to_fixed_int,
)

logger = logging.getLogger(__name__)

# Default Databento mbp-10 column names.
TS_RECV = "ts_recv"
TS_EVENT = "ts_event"
ACTION = "action"
SIDE = "side"
PRICE = "price"
SIZE = "size"
FLAGS = "flags"
SEQUENCE = "sequence"
SYMBOL = "symbol"
INSTRUMENT_ID = "instrument_id"

#: Assumptions that a one-day sample must confirm before scaling up.
PILOT_ASSUMPTIONS = [
    "F_LAST (128) marks the final record of one captured ts_recv PACKET, which "
    "may span several `sequence` values. CONFIRMED on XNAS.ITCH INTC "
    "2026-08-05: max records/event (16) equals max ts_recv group size (16), no "
    "two canonical events share a ts_recv, and 36,703 events span >1 sequence "
    "while sequence groups never exceed 2 records. Re-confirm per dataset.",
    "Records of one native event are CONSECUTIVE in native file order.",
    "action=='T' records carry the AGGRESSOR side in `side` ('B' buy, 'A' sell).",
    "action=='T' does not itself change the displayed book state. Only PART of "
    "the assumed Trade->Fill/Cancel pairing was observed on INTC 2026-08-05: "
    "69,079 'TC' groups vs 99,464 standalone 'T' records, so 59% of trades "
    "carry no same-sequence book update. Aggregating at packet level is what "
    "makes this harmless — do not narrow it back to sequence level.",
    "The last record of an event carries the post-event book for all levels.",
    "Prices are fixed-point int64 nanodollars unless price_mode='float'.",
    "UNDEF_PRICE (INT64_MAX) marks an absent price level.",
    "action=='R' clears the book and invalidates cross-event differencing.",
    "Records sharing a ts_recv were captured in ONE packet, so none of them "
    "can be executed against another (execution.no_fill_within_signal_batch).",
    "A halt/auction reopen is INFERRED from a one-sided book lasting at least "
    "integrity.halt_min_outage_ms; mbp-10 carries no halt action. Verify the "
    "inferred count against the day's known halts.",
    "Auction/cross prints are not separately identifiable in mbp-10; they are "
    "handled only by the continuous-session filter. Verify against the day's "
    "opening/closing cross.",
]


def depth_cols(prefix: str, n_levels: int) -> List[str]:
    """e.g. ('bid_px', 10) -> ['bid_px_00', ..., 'bid_px_09']."""
    return [f"{prefix}_{i:02d}" for i in range(n_levels)]


# --- 1. Raw read — native order preserved ---
def _index_to_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Promote a NAMED index back to an ordinary column.

    Databento's own exporters index the frame by ``ts_recv``
    (``DBNStore.to_df`` does, and ``to_parquet`` writes that index into the
    file's pandas metadata). Reading such a file back leaves the causal clock
    sitting in the index, where every downstream ``df[TS_RECV]`` lookup misses
    it. Promoting it costs nothing when the index is the default RangeIndex.

    Order is untouched — ``reset_index`` preserves row order, which is what
    ``raw_row_id`` is about to be stamped from.
    """
    named = [n for n in (df.index.names or []) if n is not None]
    if not named:
        return df
    keep = [n for n in named if n not in df.columns]
    if not keep:
        return df.reset_index(drop=True)
    logger.info("Promoted index column(s) %s to ordinary columns.", keep)
    return df.reset_index()


def read_raw(paths, file_format: str = "auto") -> pd.DataFrame:
    """Read one or more Databento files, preserving native order.

    ``raw_row_id`` is stamped immediately and monotonically across the
    concatenation, so it can serve as the final tie-breaker in every later
    sort. Files are concatenated in the order given — deterministic by
    construction rather than by filesystem listing order.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    frames = []
    for src_idx, p in enumerate(paths):
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"Databento file not found: {path}")
        fmt = file_format
        if fmt == "auto":
            suffix = path.suffix.lower()
            fmt = "parquet" if suffix in (".parquet", ".pq") else "csv"
        logger.info("Reading Databento %s as %s", path, fmt)
        df = pd.read_parquet(path) if fmt == "parquet" else pd.read_csv(path)
        df = _index_to_columns(df)
        df["_source_file"] = str(path)
        df["_source_index"] = src_idx
        frames.append(df)

    out = pd.concat(frames, ignore_index=True, sort=False)
    out["raw_row_id"] = np.arange(len(out), dtype="int64")
    logger.info("Read %d raw records from %d file(s)", len(out), len(frames))
    return out


# --- 2. Prepare — flags, prices, session ---
def _parse_ts(s: pd.Series) -> pd.Series:
    """Parse a Databento timestamp column (epoch ns or ISO string) to UTC."""
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_datetime(s, unit="ns", utc=True)
    return pd.to_datetime(s, utc=True)


def prepare_raw(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Decode flags, convert prices exactly once, derive session_date."""
    dcfg = config.data
    out = df.copy()

    if TS_RECV not in out.columns:
        raise KeyError(f"{TS_RECV!r} missing — required as the causal clock.")
    out[TS_RECV] = _parse_ts(out[TS_RECV])
    if TS_EVENT in out.columns:
        out[TS_EVENT] = _parse_ts(out[TS_EVENT])
        out["ts_recv_minus_ts_event_ms"] = (
            (out[TS_RECV] - out[TS_EVENT]) / pd.Timedelta(milliseconds=1))
    else:
        out[TS_EVENT] = pd.NaT
        out["ts_recv_minus_ts_event_ms"] = np.nan

    out = decode_flags(out, FLAGS)

    # ---- prices: converted exactly once, per explicit mode ----
    n_levels = min(config.columns.n_depth_levels, config.execution.depth_levels)
    price_cols = ([PRICE] + depth_cols("bid_px", n_levels)
                  + depth_cols("ask_px", n_levels))
    present = [c for c in price_cols if c in out.columns]
    keep_int = dcfg.price_mode == "fixed" and dcfg.retain_fixed_price_integers
    for c in present:
        if keep_int:
            # snapshot the untouched integers BEFORE converting, so quote
            # comparisons never have to trust a float round-trip
            out[c + INT_PRICE_SUFFIX] = to_fixed_int(
                out[c], dcfg.undef_price_sentinel)
        out[c] = to_dollars(out[c], dcfg.price_mode, dcfg.price_scale,
                            dcfg.undef_price_sentinel)
    if keep_int:
        for c in present:
            assert_prices_converted_once(
                out[c], out[c + INT_PRICE_SUFFIX], dcfg.price_scale,
                context=f"raw DBN load ({c})")
    assert_plausible_prices(out, present, dcfg.plausible_price_min,
                            dcfg.plausible_price_max, context="raw DBN load")
    logger.info("Converted %d price column(s) using price_mode=%r (once)%s",
                len(present), dcfg.price_mode,
                "; fixed-point integers retained" if keep_int else "")

    # ---- exchange-local session date (never the UTC calendar date) ----
    local = out[TS_RECV].dt.tz_convert(dcfg.session_timezone)
    out["exchange_local_time"] = local
    out["session_date"] = local.dt.date
    out["in_continuous_session"] = _continuous_mask(local, dcfg)

    # ---- instrument ----
    if SYMBOL in out.columns and out[SYMBOL].notna().any():
        out["instrument"] = out[SYMBOL].astype(str)
    elif INSTRUMENT_ID in out.columns:
        out["instrument"] = out[INSTRUMENT_ID].astype(str)
    else:
        out["instrument"] = "INSTRUMENT_0"

    if SEQUENCE not in out.columns:
        out[SEQUENCE] = np.nan
        logger.warning("No 'sequence' column; event grouping relies on F_LAST "
                       "alone and cannot be cross-checked.")
    if ACTION not in out.columns:
        out[ACTION] = ""
        logger.warning("No 'action' column; trades and book clears cannot be "
                       "identified. Signed volume will be zero.")
    if SIDE not in out.columns:
        out[SIDE] = ""

    return out


def _continuous_mask(local: pd.Series, dcfg) -> pd.Series:
    """Boolean mask for the exchange-local continuous-trading window."""
    if not dcfg.continuous_session_start or not dcfg.continuous_session_end:
        return pd.Series(True, index=local.index)
    start = pd.to_datetime(dcfg.continuous_session_start).time()
    end = pd.to_datetime(dcfg.continuous_session_end).time()
    tod = local.dt.time
    return (tod >= start) & (tod <= end)


# --- 3. Ingestion duplicates only (P0.1) ---
#: Columns forming a record's immutable raw identity. Timestamps alone are
#: NEVER enough: two genuine book updates can legitimately share ts_recv.
RAW_IDENTITY_COLS = [
    TS_RECV, TS_EVENT, SEQUENCE, ACTION, SIDE, PRICE, SIZE, FLAGS,
    "instrument", "depth", "rtype", "publisher_id", "instrument_id",
]


def drop_ingestion_duplicates(df: pd.DataFrame,
                              identity_cols: Optional[List[str]] = None
                              ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Remove only records that are identical across their FULL raw identity.

    Overlapping downloads of the same day produce byte-identical records; those
    are safe to collapse. Two distinct book updates sharing ``ts_recv`` (or
    even ``sequence``) are NOT duplicates and are always retained.

    Returns ``(deduped, report)``.
    """
    cols = identity_cols or RAW_IDENTITY_COLS
    present = [c for c in cols if c in df.columns]
    # Require the full book state to match too — a same-identity record with a
    # different resulting book is a real event, not a re-ingest.
    book_like = [c for c in df.columns
                 if c.startswith(("bid_px_", "ask_px_", "bid_sz_", "ask_sz_"))]
    key = present + book_like
    if not key:
        return df.copy(), pd.DataFrame(columns=["reason", "n_removed"])

    dup = df.duplicated(subset=key, keep="first")
    n = int(dup.sum())
    if n:
        logger.warning("Removed %d confirmed ingestion duplicate(s) "
                       "(identical across %d identity columns).", n, len(key))
    report = pd.DataFrame([{
        "reason": "confirmed_ingestion_duplicate",
        "n_removed": n,
        "pct_removed": 100.0 * n / max(len(df), 1),
        "identity_columns": ",".join(key)[:400],
    }])
    return df.loc[~dup].copy(), report


# --- 4. Canonical native events (P0.2) ---
def assign_event_ids(df: pd.DataFrame) -> pd.Series:
    """Group CONSECUTIVE records into events using ``F_LAST``.

    An event runs from just after the previous ``F_LAST`` through the next one,
    grouped within ``(instrument, session_date)`` over consecutive rows in
    native order — equal ``sequence`` values are never pooled globally.

    In real records ``F_LAST`` delimits one captured ``ts_recv`` packet rather
    than one native Nasdaq message, and a packet can carry several ``sequence``
    values (see :data:`PILOT_ASSUMPTIONS`). That is the more useful boundary,
    not a defect: every record in a packet became available at the same instant,
    so one post-packet book state is exactly the granularity at which a decision
    can be taken and P0.5's batching rule enforced. Splitting at ``sequence``
    would manufacture states no strategy could have acted on separately.

    Falls back to runs of equal ``sequence`` when no ``F_LAST`` is present.
    """
    n = len(df)
    if n == 0:
        return pd.Series([], dtype="int64")

    has_last = "F_LAST" in df.columns and bool(df["F_LAST"].any())
    grp_change = (
        (df["instrument"] != df["instrument"].shift())
        | (df["session_date"] != df["session_date"].shift())
    ).to_numpy()
    grp_change[0] = True

    if has_last:
        # a new event starts on the row AFTER an F_LAST, or at a group change
        prev_last = df["F_LAST"].shift(1, fill_value=False).to_numpy()
        starts = prev_last | grp_change
    else:
        seq = df[SEQUENCE]
        seq_change = (seq != seq.shift()).to_numpy()
        starts = seq_change | grp_change
        logger.warning("No F_LAST flags present — falling back to consecutive "
                       "equal-`sequence` runs for event boundaries. Confirm "
                       "against a real sample before trusting.")
    starts[0] = True
    return pd.Series(np.cumsum(starts) - 1, index=df.index, dtype="int64")


def _side_sign(side: pd.Series) -> pd.Series:
    """Aggressor side -> {+1, -1, 0}. Databento: 'B' buy, 'A' sell, 'N' none."""
    s = side.astype(str).str.upper().str.strip()
    return pd.Series(
        np.where(s == SIDE_BID, 1.0, np.where(s == SIDE_ASK, -1.0, 0.0)),
        index=side.index)


def build_canonical_events(df: pd.DataFrame, config: Config
                           ) -> Tuple[pd.DataFrame, Dict]:
    """Collapse normalized records into one row per native event.

    For each event: the final post-event book state, the summed signed trade
    volume across its ``T`` records, OR-ed flags, action diagnostics, both
    timestamps, and the source row range.
    """
    n_levels = min(config.columns.n_depth_levels, config.execution.depth_levels)
    work = df.copy()
    work["event_id"] = assign_event_ids(work)

    sign = _side_sign(work[SIDE])
    is_trade = work[ACTION].astype(str).str.upper().str.strip() == ACTION_TRADE
    size_num = pd.to_numeric(work[SIZE], errors="coerce").fillna(0.0)
    work["_signed_trade_volume"] = np.where(is_trade, sign * size_num, 0.0)
    work["_trade_size"] = np.where(is_trade, size_num, 0.0)
    work["_trade_notional"] = np.where(
        is_trade, size_num * pd.to_numeric(work[PRICE], errors="coerce")
        .fillna(0.0), 0.0)
    work["_is_trade"] = is_trade
    work["_unspecified_side_trade"] = is_trade & (sign == 0.0)
    work["_is_clear"] = (
        work[ACTION].astype(str).str.upper().str.strip() == ACTION_CLEAR)

    g = work.groupby("event_id", sort=True)

    # ---- final post-event book state = LAST record of the event ----
    book_cols = (depth_cols("bid_px", n_levels) + depth_cols("ask_px", n_levels)
                 + depth_cols("bid_sz", n_levels) + depth_cols("ask_sz", n_levels))
    # the retained fixed-point integers travel with their dollar counterparts
    book_cols += [c + INT_PRICE_SUFFIX for c in
                  depth_cols("bid_px", n_levels) + depth_cols("ask_px", n_levels)]
    present_book = [c for c in book_cols if c in work.columns]
    ev = g[present_book].last() if present_book else pd.DataFrame(
        index=g.size().index)

    ev["ts_recv"] = g[TS_RECV].last()
    ev["ts_recv_first"] = g[TS_RECV].first()
    ev["ts_event"] = g[TS_EVENT].last()
    ev["ts_recv_minus_ts_event_ms"] = g["ts_recv_minus_ts_event_ms"].last()
    ev["instrument"] = g["instrument"].last()
    ev["session_date"] = g["session_date"].last()
    ev["in_continuous_session"] = g["in_continuous_session"].last()
    ev["sequence"] = g[SEQUENCE].last()
    ev["n_distinct_sequence"] = g[SEQUENCE].nunique(dropna=False)
    ev["n_records"] = g.size()
    ev["src_row_first"] = g["raw_row_id"].min()
    ev["src_row_last"] = g["raw_row_id"].max()

    ev["signed_trade_volume"] = g["_signed_trade_volume"].sum()
    ev["trade_size"] = g["_trade_size"].sum()
    notional = g["_trade_notional"].sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        ev["trade_price"] = np.where(ev["trade_size"] > 0,
                                     notional / ev["trade_size"].replace(0, np.nan),
                                     np.nan)
    ev["n_trades"] = g["_is_trade"].sum()
    ev["n_unspecified_side_trades"] = g["_unspecified_side_trade"].sum()
    ev["book_clear"] = g["_is_clear"].any()
    ev["actions"] = g[ACTION].apply(lambda s: "".join(sorted(set(
        s.astype(str).str.upper().str.strip()))))

    # flags OR-ed across the event
    for name in ("F_LAST", "F_SNAPSHOT", "F_BAD_TS_RECV", "F_MAYBE_BAD_BOOK"):
        ev[name] = g[name].any() if name in work.columns else False

    # a chain break anywhere inside the event breaks the event (P0.4)
    ev["integrity_break"] = (g["integrity_break"].any()
                             if "integrity_break" in work.columns else False)

    ev = ev.reset_index(drop=True)
    ev["canonical_event_id"] = np.arange(len(ev), dtype="int64")

    diagnostics = {
        "n_raw_records": int(len(work)),
        "n_canonical_events": int(len(ev)),
        "records_per_event_mean": float(ev["n_records"].mean())
        if len(ev) else np.nan,
        "records_per_event_max": int(ev["n_records"].max()) if len(ev) else 0,
        "n_events_mixed_sequence": int((ev["n_distinct_sequence"] > 1).sum()),
        "n_events_with_trade": int((ev["n_trades"] > 0).sum()),
        "n_book_clears": int(ev["book_clear"].sum()),
        "n_unspecified_side_trades": int(ev["n_unspecified_side_trades"].sum()),
    }
    if diagnostics["n_events_mixed_sequence"]:
        # Expected, not alarming: an F_LAST-delimited packet legitimately spans
        # several sequences (see build_canonical_events). Reported at INFO with
        # the share, because a sudden change in that share between datasets IS
        # worth noticing.
        logger.info(
            "%d canonical event(s) (%.2f%%) span more than one `sequence` "
            "value — expected, since F_LAST bounds a ts_recv packet rather "
            "than a single native message.",
            diagnostics["n_events_mixed_sequence"],
            100.0 * diagnostics["n_events_mixed_sequence"] / max(len(ev), 1))
    logger.info("Canonical events: %d records -> %d events (mean %.2f rec/event)",
                diagnostics["n_raw_records"], diagnostics["n_canonical_events"],
                diagnostics["records_per_event_mean"])
    return ev, diagnostics


# --- 5. Canonical schema for the OFI pipeline ---
def to_canonical_frame(ev: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Rename event columns to the canonical names the OFI pipeline expects.

    Levels 1-2 feed the signal; levels 3..N are carried through under
    ``bid_price_k`` / ``ask_size_k`` for taker execution only.
    """
    n_levels = min(config.columns.n_depth_levels, config.execution.depth_levels)
    out = pd.DataFrame(index=ev.index)

    # `timestamp` is the CAUSAL clock: ts_recv (capture-server availability).
    out["timestamp"] = ev["ts_recv"]
    out["ts_recv"] = ev["ts_recv"]
    out["ts_event"] = ev["ts_event"]
    out["ts_recv_minus_ts_event_ms"] = ev["ts_recv_minus_ts_event_ms"]
    out["instrument"] = ev["instrument"]
    out["sequence"] = ev["sequence"]
    out["session_date"] = ev["session_date"]
    out["in_continuous_session"] = ev["in_continuous_session"]
    out["canonical_event_id"] = ev["canonical_event_id"]
    out["raw_row_id"] = ev["src_row_first"]
    out["src_row_first"] = ev["src_row_first"]
    out["src_row_last"] = ev["src_row_last"]
    out["n_records"] = ev["n_records"]
    out["actions"] = ev["actions"]
    out["n_trades"] = ev["n_trades"]

    for k in range(1, n_levels + 1):
        i = k - 1
        for canon, src in (
            (f"bid_price_{k}", f"bid_px_{i:02d}"),
            (f"ask_price_{k}", f"ask_px_{i:02d}"),
            (f"bid_size_{k}", f"bid_sz_{i:02d}"),
            (f"ask_size_{k}", f"ask_sz_{i:02d}"),
        ):
            out[canon] = ev[src] if src in ev.columns else np.nan
        # exact integer quote prices, where they were retained
        for canon, src in (
            (f"bid_price_{k}{INT_PRICE_SUFFIX}", f"bid_px_{i:02d}{INT_PRICE_SUFFIX}"),
            (f"ask_price_{k}{INT_PRICE_SUFFIX}", f"ask_px_{i:02d}{INT_PRICE_SUFFIX}"),
        ):
            if src in ev.columns:
                out[canon] = ev[src]

    out["trade_price"] = ev["trade_price"]
    out["trade_size"] = ev["trade_size"].replace(0.0, np.nan)
    # Signed volume is carried EXPLICITLY: an event may contain both buy and
    # sell executions, so direction * size would not reproduce the net.
    out["signed_trade_volume"] = ev["signed_trade_volume"]
    out["trade_side"] = np.where(
        ev["signed_trade_volume"] > 0, "B",
        np.where(ev["signed_trade_volume"] < 0, "S", np.nan))
    out["event_type"] = np.where(
        ev["n_trades"] > 0,
        np.where(ev["signed_trade_volume"] >= 0, "market_buy", "market_sell"),
        "book_update")

    # integrity signals consumed by features.add_segments
    out["book_clear"] = ev["book_clear"]
    out["F_SNAPSHOT"] = ev["F_SNAPSHOT"]
    out["F_MAYBE_BAD_BOOK"] = ev["F_MAYBE_BAD_BOOK"]
    out["F_BAD_TS_RECV"] = ev["F_BAD_TS_RECV"]
    if "integrity_break" in ev.columns:
        out["integrity_break"] = ev["integrity_break"]

    return out.reset_index(drop=True)


def sort_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Stable sort with ``raw_row_id`` as the FINAL tie-breaker (P0.1).

    Records sharing ``ts_recv`` keep their native relative order, which is what
    makes the book state at each row well defined.
    """
    keys = [k for k in ("instrument", "timestamp", "sequence", "raw_row_id")
            if k in df.columns]
    return df.sort_values(keys, kind="mergesort").reset_index(drop=True)


# --- Orchestration + audit ---
def load_databento(paths, config: Config
                   ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict]:
    """Full path: read -> prepare -> dedupe -> events -> canonical frame.

    Returns ``(canonical_df, audit_tables, diagnostics)``.
    """
    raw = read_raw(paths, config.data.file_format)
    prepared = prepare_raw(raw, config)
    deduped, dup_report = drop_ingestion_duplicates(prepared)

    audits = {
        "raw_multiplicity": audit_multiplicity(deduped),
        "flags": audit_flags(deduped),
        "actions": audit_actions(deduped),
        "consecutive_actions": audit_consecutive_actions(deduped),
        "bad_ts_recv": audit_bad_ts_recv(deduped),
        "session_boundaries": audit_session_boundaries(deduped),
        "ingestion_duplicates": dup_report,
    }

    diagnostics_pre: Dict = {}
    if config.data.exclude_bad_ts_recv and "F_BAD_TS_RECV" in deduped.columns:
        bad = deduped["F_BAD_TS_RECV"].fillna(False).astype(bool)
        n_bad = int(bad.sum())
        if n_bad:
            # Dropping these removes real book transitions, so the survivors
            # must not be differenced across the hole (P0.4).
            deduped = deduped.loc[~bad].copy()
            deduped["integrity_break"] = False
            deduped.iloc[0, deduped.columns.get_loc("integrity_break")] = True
            logger.warning(
                "Excluded %d record(s) flagged F_BAD_TS_RECV; the OFI chain is "
                "broken at the exclusion point rather than differenced across "
                "it.", n_bad)
        diagnostics_pre["n_excluded_bad_ts_recv"] = n_bad

    if config.data.restrict_to_continuous_session:
        before = len(deduped)
        deduped = deduped.loc[deduped["in_continuous_session"]].copy()
        logger.info("Continuous-session filter kept %d/%d records (%s-%s %s)",
                    len(deduped), before, config.data.continuous_session_start,
                    config.data.continuous_session_end,
                    config.data.session_timezone)

    ev, diagnostics = build_canonical_events(deduped, config)
    diagnostics.update(diagnostics_pre)
    canonical = sort_canonical(to_canonical_frame(ev, config))

    # halt / auction reopen, inferred from quote outages (HEURISTIC — verify)
    canonical["halt_boundary"] = mark_halt_boundaries(canonical, config)
    diagnostics["n_inferred_halt_boundaries"] = int(
        canonical["halt_boundary"].sum())

    # tick-size check against the configured value
    if "bid_price_1" in canonical:
        inferred = infer_tick_size(canonical["bid_price_1"],
                                   config.features.tick_size)
        diagnostics["inferred_tick_size"] = inferred
        if not np.isclose(inferred, config.features.tick_size, rtol=0.5):
            logger.warning("Inferred quote increment %.6g differs from "
                           "configured tick_size %.6g — verify before costing.",
                           inferred, config.features.tick_size)

    diagnostics["n_canonical_rows"] = int(len(canonical))
    return canonical, audits, diagnostics


def audit_multiplicity(df: pd.DataFrame) -> pd.DataFrame:
    """Timestamp / sequence multiplicity — items 3 and 4 of the pilot audit."""
    rows = []
    if TS_RECV in df.columns:
        vc = df.groupby([TS_RECV]).size()
        rows.append({"key": "ts_recv", "n_groups": int(len(vc)),
                     "n_groups_gt1": int((vc > 1).sum()),
                     "max_group_size": int(vc.max()) if len(vc) else 0,
                     "mean_group_size": float(vc.mean()) if len(vc) else np.nan,
                     "pct_rows_in_shared_group":
                         100.0 * float(vc[vc > 1].sum()) / max(len(df), 1)})
    if SEQUENCE in df.columns and df[SEQUENCE].notna().any():
        vs = df.groupby([SEQUENCE]).size()
        rows.append({"key": "sequence", "n_groups": int(len(vs)),
                     "n_groups_gt1": int((vs > 1).sum()),
                     "max_group_size": int(vs.max()) if len(vs) else 0,
                     "mean_group_size": float(vs.mean()) if len(vs) else np.nan,
                     "pct_rows_in_shared_group":
                         100.0 * float(vs[vs > 1].sum()) / max(len(df), 1)})
    return pd.DataFrame(rows)


def audit_consecutive_actions(df: pd.DataFrame, max_patterns: int = 50
                              ) -> pd.DataFrame:
    """Action patterns inside repeated-``sequence`` groups (P0.1, pilot item 4).

    The question this answers is the one that decides whether the event-boundary
    logic is right: when Databento splits one native Nasdaq message into several
    records, *what* does that group of records look like? A ``T`` followed by an
    ``F``/``C`` that carries the book update is the pattern the canonical-event
    builder assumes. Anything else showing up in volume here means
    :func:`build_canonical_events` is aggregating something other than what was
    assumed, and the assumption — not the data — has to change.

    One row per distinct consecutive action pattern, most frequent first.
    """
    if ACTION not in df.columns or SEQUENCE not in df.columns:
        return pd.DataFrame(columns=["pattern", "n_groups", "group_size",
                                     "pct_of_groups"])
    a = df[ACTION].astype(str).str.upper().str.strip()
    seq = df[SEQUENCE]
    grp_change = (
        (seq != seq.shift())
        | (df["instrument"] != df["instrument"].shift())
        if "instrument" in df.columns else (seq != seq.shift())
    ).to_numpy()
    grp_change[0] = True
    gid = np.cumsum(grp_change) - 1

    work = pd.DataFrame({"gid": gid, "action": a.to_numpy()})
    grouped = work.groupby("gid")["action"]
    # ORDER matters — 'TF' and 'FT' are different claims about the feed
    patterns = grouped.apply(lambda s: "".join(s.tolist()))
    sizes = grouped.size()
    tab = (pd.DataFrame({"pattern": patterns, "group_size": sizes})
           .groupby(["pattern", "group_size"]).size()
           .reset_index(name="n_groups"))
    n_groups_total = int(len(patterns))
    tab["pct_of_groups"] = 100.0 * tab["n_groups"] / max(n_groups_total, 1)
    tab = tab.sort_values("n_groups", ascending=False).reset_index(drop=True)
    return tab.head(max_patterns)


def audit_bad_ts_recv(df: pd.DataFrame) -> pd.DataFrame:
    """Report ``F_BAD_TS_RECV`` incidence (P0.5).

    Every clock-latency and clock-horizon statement in the pipeline rests on
    ``ts_recv``. Records where the vendor itself flags that timestamp as
    unreliable have to be counted before any of those statements is believed —
    silently averaging them in would corrupt exactly the quantity being
    measured.
    """
    n = max(len(df), 1)
    flagged = (df["F_BAD_TS_RECV"].fillna(False).astype(bool)
               if "F_BAD_TS_RECV" in df.columns
               else pd.Series(False, index=df.index))
    row = {
        "n_records": int(len(df)),
        "n_bad_ts_recv": int(flagged.sum()),
        "pct_bad_ts_recv": 100.0 * float(flagged.sum()) / n,
    }
    if "ts_recv_minus_ts_event_ms" in df.columns:
        d = pd.to_numeric(df["ts_recv_minus_ts_event_ms"],
                          errors="coerce").dropna()
        if len(d):
            row.update({
                "capture_lag_median_ms": float(d.median()),
                "capture_lag_p99_ms": float(d.quantile(0.99)),
                "capture_lag_negative_count": int((d < 0).sum()),
            })
    row["interpretation"] = (
        "ts_recv - ts_event is a CAPTURE/FEED-latency proxy (venue send -> "
        "Databento capture server), not network congestion and not your own "
        "receive time.")
    return pd.DataFrame([row])


def mark_halt_boundaries(df: pd.DataFrame, config: Config) -> pd.Series:
    """Infer halt/auction-reopen boundaries from quote outages (HEURISTIC).

    mbp-10 carries no halt action, so this is inferred, not read: a one-sided
    or empty book that persists for at least
    ``IntegrityConfig.halt_min_outage_ms`` and is then restored marks the
    restoring record as a boundary. Differencing the reopening book against the
    pre-halt book would otherwise manufacture an enormous fictitious OFI
    increment out of an auction.

    Conservative by construction: it only ever adds segment breaks. Confirm the
    incidence against a one-day sample before trusting the count.
    """
    bid = pd.to_numeric(df.get("bid_price_1"), errors="coerce") \
        if "bid_price_1" in df.columns else pd.Series(np.nan, index=df.index)
    ask = pd.to_numeric(df.get("ask_price_1"), errors="coerce") \
        if "ask_price_1" in df.columns else pd.Series(np.nan, index=df.index)
    two_sided = bid.notna() & ask.notna()
    if two_sided.all() or not two_sided.any():
        return pd.Series(False, index=df.index)

    ts = pd.to_datetime(df["timestamp"] if "timestamp" in df.columns
                        else df[TS_RECV])
    t_ms = ts.astype("int64").to_numpy() / 1e6
    ok = two_sided.to_numpy()

    out = np.zeros(len(df), dtype=bool)
    outage_start: Optional[float] = None
    for i in range(len(df)):
        if not ok[i]:
            if outage_start is None:
                outage_start = t_ms[i]
        else:
            if outage_start is not None:
                if t_ms[i] - outage_start >= config.integrity.halt_min_outage_ms:
                    out[i] = True
                outage_start = None
    return pd.Series(out, index=df.index)


def audit_session_boundaries(df: pd.DataFrame) -> pd.DataFrame:
    """Per-session record counts and first/last exchange-local timestamps.

    Early closes and DST shifts show up here as a short session or a shifted
    local open, which is the point: ``session_date`` is derived in exchange-
    local time precisely so those days do not silently merge or split.
    """
    if "session_date" not in df.columns:
        return pd.DataFrame()
    local_col = "exchange_local_time" if "exchange_local_time" in df.columns \
        else TS_RECV
    g = df.groupby("session_date")
    tab = pd.DataFrame({
        "n_records": g.size(),
        "local_first": g[local_col].min(),
        "local_last": g[local_col].max(),
    }).reset_index()
    tab["session_span_hours"] = (
        (pd.to_datetime(tab["local_last"], utc=True)
         - pd.to_datetime(tab["local_first"], utc=True))
        / pd.Timedelta(hours=1))
    if "in_continuous_session" in df.columns:
        tab["n_in_continuous"] = g["in_continuous_session"].sum().to_numpy()
        tab["n_outside_continuous"] = (
            tab["n_records"].to_numpy() - tab["n_in_continuous"].to_numpy())
    return tab


def audit_actions(df: pd.DataFrame) -> pd.DataFrame:
    """Counts by action and side — item 2 / item 8 of the pilot audit."""
    if ACTION not in df.columns:
        return pd.DataFrame(columns=["action", "side", "n", "pct"])
    a = df[ACTION].astype(str).str.upper().str.strip()
    s = df[SIDE].astype(str).str.upper().str.strip() if SIDE in df.columns \
        else pd.Series("", index=df.index)
    t = (pd.DataFrame({"action": a, "side": s})
         .groupby(["action", "side"]).size().reset_index(name="n"))
    t["pct"] = 100.0 * t["n"] / max(len(df), 1)
    return t.sort_values("n", ascending=False).reset_index(drop=True)


def trace_events(raw_prepared: pd.DataFrame, canonical: pd.DataFrame,
                 n_events: int = 100, start_event: int = 0) -> pd.DataFrame:
    """Manual raw->event trace — items 9 and 10 of the pilot audit.

    Emits one row per RAW record for the selected canonical events, alongside
    the event's aggregated outputs, so a human can verify by eye that record
    grouping, book state, and signed volume line up.
    """
    work = raw_prepared.copy()
    work["event_id"] = assign_event_ids(work)
    sel = work[(work["event_id"] >= start_event)
               & (work["event_id"] < start_event + n_events)].copy()
    keep = [c for c in ["event_id", "raw_row_id", TS_RECV, TS_EVENT, SEQUENCE,
                        ACTION, SIDE, PRICE, SIZE, "F_LAST", "F_SNAPSHOT",
                        "F_MAYBE_BAD_BOOK", "bid_px_00", "ask_px_00",
                        "bid_sz_00", "ask_sz_00"] if c in sel.columns]
    sel = sel[keep]
    ev_cols = [c for c in ["canonical_event_id", "signed_trade_volume",
                           "trade_size", "n_records", "actions",
                           "bid_price_1", "ask_price_1", "bid_size_1",
                           "ask_size_1"] if c in canonical.columns]
    ev = canonical[ev_cols].iloc[start_event:start_event + n_events]
    return sel.merge(ev, left_on="event_id", right_on="canonical_event_id",
                     how="left", suffixes=("_raw", "_event"))
