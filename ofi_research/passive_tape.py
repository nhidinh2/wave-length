"""Per-session event-resolution tape for the passive queue simulator.

The decision clock (:func:`ofi_research.sampling.sample_decision_rows`) keeps
the FIRST row of each interval, so on the sampled frame a row's traded volume
is one event's worth, not the interval's. That is harmless for features and
targets — both were computed on the complete stream before sampling — but it is
fatal for a queue model, which fills an order by CONSUMING shares: given only
1/150th of the flow, every order but a front-of-queue one would look unfillable.

Measured on INTC 2026-07-29, the opposite error is just as bad. Summing volume
into 200 ms buckets makes 66% of buckets trade more than the entire displayed
L1 depth, so a back-of-queue order and a front-of-queue order would fill in the
same bucket two times in three and the queue-position dial would read as "no
effect" — a statement about the sampling grid wearing the costume of a result.

Hence this tape: the handful of columns a queue simulation actually consumes,
kept at full event resolution and written to disk one session at a time. It is
~10 columns rather than the ~250 of the feature frame, which is what makes
event resolution affordable at all (~0.2 GB in memory per session, and only one
session is ever resident).

The gate stays on the decision clock — deciding whether to quote every 200 ms
is a real strategy, and a defensible one. Only the FILL is resolved per event.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .config import Config

logger = logging.getLogger(__name__)

#: Directory under the run's output dir where tapes are written.
TAPE_SUBDIR = "passive_tapes"

#: Everything the simulator reads, and nothing else.
TAPE_COLUMNS = (
    "timestamp", "session_date", "segment_id", "midprice",
    "bid_price_1", "ask_price_1", "bid_size_1", "ask_size_1",
    "signed_volume_increment",
)

#: Optional guards; copied when present so the simulator can exclude auction
#: and broken-book regions exactly as the Phase-0 screen does.
TAPE_OPTIONAL = ("in_continuous_session", "integrity_break")


def tape_dir(config: Config) -> Path:
    return Path(config.output_dir) / TAPE_SUBDIR


def tape_path(config: Config, session_date: str) -> Path:
    return tape_dir(config) / f"tape_{session_date}.parquet"


def write_tape(feat: pd.DataFrame, config: Config) -> Optional[Path]:
    """Write one session's event-resolution tape. Call BEFORE decision sampling.

    Returns the path written, or None when the frame lacks what a queue model
    needs. A missing column is logged and skipped rather than raised: the tape
    is an input to an optional study, and a synthetic-data run that never had a
    depth ladder should not fail the whole experiment.
    """
    missing = [c for c in TAPE_COLUMNS if c not in feat.columns]
    if missing:
        logger.warning("Passive tape: session lacks %s; not written", missing)
        return None
    if feat.empty:
        return None

    cols = list(TAPE_COLUMNS) + [c for c in TAPE_OPTIONAL if c in feat.columns]
    tape = feat.loc[:, cols].copy()

    # session_date is the file identity; a tape spanning two dates would mean
    # break_on_session_change stopped holding, which the whole streaming design
    # relies on.
    dates = pd.unique(tape["session_date"].astype(str))
    if len(dates) != 1:
        logger.warning("Passive tape: session spans %d dates %s; not written",
                       len(dates), list(dates)[:4])
        return None
    day = str(dates[0])

    # int64 ns is what searchsorted wants; doing it once here keeps the
    # simulator free of dtype branching.
    tape["timestamp"] = pd.to_datetime(tape["timestamp"]).astype("int64")
    for c in ("midprice", "bid_price_1", "ask_price_1"):
        tape[c] = tape[c].astype("float64")     # prices stay exact
    for c in ("bid_size_1", "ask_size_1", "signed_volume_increment"):
        tape[c] = tape[c].astype("float32")

    out = tape_path(config, day)
    out.parent.mkdir(parents=True, exist_ok=True)
    tape.to_parquet(out, index=False)
    logger.info("Passive tape: wrote %d events for %s (%.1f MB) -> %s",
                len(tape), day, out.stat().st_size / 1e6, out.name)
    return out


def available_tapes(config: Config) -> List[str]:
    """Session dates that have a tape on disk, sorted."""
    d = tape_dir(config)
    if not d.exists():
        return []
    return sorted(p.name[len("tape_"):-len(".parquet")]
                  for p in d.glob("tape_*.parquet"))


def read_tape(config: Config, session_date: str) -> Optional[pd.DataFrame]:
    """Load one session's tape, or None when it was never written."""
    p = tape_path(config, str(session_date))
    if not p.exists():
        return None
    tape = pd.read_parquet(p)
    # Restrict to the tradable book exactly as the screen does: no auction, no
    # crossed/broken book, no region flagged as an integrity break.
    ok = np.ones(len(tape), dtype=bool)
    if "in_continuous_session" in tape.columns:
        ok &= tape["in_continuous_session"].to_numpy(dtype=bool)
    if "integrity_break" in tape.columns:
        ok &= ~tape["integrity_break"].to_numpy(dtype=bool)
    ok &= (tape["ask_price_1"].to_numpy() >= tape["bid_price_1"].to_numpy())
    ok &= np.isfinite(tape["midprice"].to_numpy())
    out = tape.loc[ok].reset_index(drop=True)
    logger.info("Passive tape %s: %d/%d events usable", session_date,
                len(out), len(tape))
    return out
