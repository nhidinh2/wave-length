"""Data-integrity segmentation vs. quiet markets (P0.3 / P0.4).

A segment break asserts "the observable book-state chain is discontinuous".
These tests hold that line from both sides: genuine discontinuities must break
the chain, and ordinary market inactivity must not.
"""

import numpy as np
import pandas as pd
import pytest

from ofi_research.config import Config
from ofi_research.features import add_segments, compute_ofi_level
from ofi_research.validation import clean_data

DAY = pd.Timestamp("2024-03-05").date()
T0 = pd.Timestamp("2024-03-05T14:30:00Z")


def frame(gaps_ms, bid_sz, *, bid_px=99.99, ask_px=100.01, ask_sz=100.0,
          instrument=None, session_date=None, **extra):
    """Build a canonical-shaped frame. ``gaps_ms[i]`` precedes row ``i``."""
    n = len(bid_sz)
    ts = [T0]
    for g in gaps_ms:
        ts.append(ts[-1] + pd.Timedelta(milliseconds=g))
    df = pd.DataFrame({
        "instrument": instrument or ["TEST"] * n,
        "timestamp": pd.to_datetime(ts[:n], utc=True),
        "session_date": session_date or [DAY] * n,
        "sequence": np.arange(n),
        "bid_price_1": _fill(bid_px, n),
        "ask_price_1": _fill(ask_px, n),
        "bid_size_1": list(bid_sz),
        "ask_size_1": _fill(ask_sz, n),
    })
    for k, v in extra.items():
        df[k] = v
    return df


def _fill(v, n):
    return list(v) if isinstance(v, (list, tuple)) else [v] * n


def cleaning_config():
    """Config for the cleaning tests, with session trimming disabled.

    ``trim_open_minutes``/``trim_close_minutes`` default to one minute so the
    opening and closing auctions are excluded (see ``test_pilot_findings``).
    The fixtures below span milliseconds, so the default would remove every
    row and mask the removal rule actually under test.
    """
    cfg = Config()
    cfg.data.trim_open_minutes = None
    cfg.data.trim_close_minutes = None
    return cfg


def segments(df):
    """``segment_id`` renumbered from 0 in order of appearance.

    ``add_segments`` counts breaks cumulatively and always breaks on the first
    row, so the absolute ids start at 1. Only the grouping matters here.
    """
    return pd.factorize(df["segment_id"])[0].tolist()


# --------------------------------------------------------------------------- #
# P0.3: elapsed time is data, not a fault
# --------------------------------------------------------------------------- #
def test_quiet_gap_longer_than_one_second_does_not_reset():
    """A silent market is a market state, not a feed failure."""
    cfg = Config()
    df = add_segments(frame([5000.0], [100.0, 150.0]), cfg)

    assert segments(df) == [0, 0]
    assert df["long_quiet_gap"].tolist() == [False, True]
    assert df["gap_ms"].iloc[1] == pytest.approx(5000.0)
    # the chain continues across the quiet period
    assert compute_ofi_level(df, 1).iloc[1] == 50.0


def test_gap_ms_and_quiet_flag_are_retained_as_features():
    cfg = Config()
    df = add_segments(frame([10.0, 2000.0], [100.0, 110.0, 120.0]), cfg)
    assert np.isnan(df["gap_ms"].iloc[0])
    assert df["gap_ms"].tolist()[1:] == pytest.approx([10.0, 2000.0])
    assert df["long_quiet_gap"].tolist() == [False, False, True]


def test_gap_breaks_the_chain_only_on_explicit_opt_in():
    """The old ``gap_ms > 1000`` rule survives only as a deliberate choice."""
    cfg = Config()
    cfg.data.use_gap_as_integrity_break = True
    cfg.data.gap_reset_ms = 1000.0
    df = add_segments(frame([5000.0], [100.0, 150.0]), cfg)

    assert segments(df) == [0, 1]
    assert compute_ofi_level(df, 1).iloc[1] == 0.0


# --------------------------------------------------------------------------- #
# Genuine integrity breaks
# --------------------------------------------------------------------------- #
def test_instrument_change_resets():
    cfg = Config()
    df = add_segments(frame([10.0], [100.0, 150.0],
                            instrument=["AAA", "BBB"]), cfg)
    assert segments(df) == [0, 1]
    assert compute_ofi_level(df, 1).iloc[1] == 0.0


def test_session_change_resets():
    cfg = Config()
    df = add_segments(
        frame([10.0], [100.0, 150.0],
              session_date=[DAY, pd.Timestamp("2024-03-06").date()]), cfg)
    assert segments(df) == [0, 1]


def test_book_clear_resets_into_and_out_of_the_clearing_row():
    """Action ``R`` invalidates differencing on both sides of itself."""
    cfg = Config()
    df = add_segments(
        frame([10.0, 10.0], [100.0, 150.0, 200.0],
              book_clear=[False, True, False]), cfg)

    assert segments(df) == [0, 1, 2]
    ofi = compute_ofi_level(df, 1)
    assert ofi.iloc[1] == 0.0
    assert ofi.iloc[2] == 0.0


def test_maybe_bad_book_resets():
    cfg = Config()
    df = add_segments(
        frame([10.0, 10.0], [100.0, 150.0, 200.0],
              F_MAYBE_BAD_BOOK=[False, True, False]), cfg)
    assert segments(df) == [0, 1, 2]


def test_snapshot_resets():
    """``F_SNAPSHOT`` restates the book; it is not a delta."""
    cfg = Config()
    df = add_segments(
        frame([10.0, 10.0], [100.0, 150.0, 200.0],
              F_SNAPSHOT=[False, True, False]), cfg)
    assert segments(df) == [0, 1, 1]
    assert compute_ofi_level(df, 1).iloc[1] == 0.0


def test_halt_boundary_resets():
    """A reopening auction book has no differencing relation to the pre-halt one."""
    cfg = Config()
    df = add_segments(
        frame([10.0, 10.0], [100.0, 150.0, 200.0],
              halt_boundary=[False, True, False]), cfg)
    assert segments(df) == [0, 1, 2]
    assert compute_ofi_level(df, 1).iloc[1] == 0.0


def test_missing_book_state_resets():
    cfg = Config()
    df = add_segments(
        frame([10.0, 10.0], [100.0, 150.0, 200.0],
              bid_px=[99.99, np.nan, 99.99]), cfg)
    assert segments(df) == [0, 1, 2]


# --------------------------------------------------------------------------- #
# P0.4: removals and the OFI chain
# --------------------------------------------------------------------------- #
def test_nonbenign_removal_breaks_the_next_increment():
    """A dropped crossed book may have hidden a real transition."""
    cfg = cleaning_config()
    df = frame([10.0, 10.0], [100.0, 130.0, 150.0],
               bid_px=[99.99, 100.5, 99.99])  # row 1 is crossed
    clean, report = clean_data(df, cfg)

    assert len(clean) == 2
    assert report["reason"].tolist() == ["crossed_book"]
    assert clean["integrity_break"].tolist() == [False, True]

    seg = add_segments(clean, cfg)
    assert segments(seg) == [0, 1]
    # without the break this would have been 150 - 100 = +50 across a hole
    assert compute_ofi_level(seg, 1).iloc[1] == 0.0


def test_benign_ingestion_duplicate_does_not_create_a_reset():
    """A byte-identical re-ingest skipped no state, so the chain continues."""
    cfg = cleaning_config()
    df = frame([10.0, 0.0], [100.0, 150.0, 150.0])
    # make row 2 an exact re-ingest of row 1
    df.loc[2] = df.loc[1]
    df["raw_row_id"] = [0, 1, 2]   # row-unique bookkeeping, not an identity

    clean, report = clean_data(df, cfg)

    assert len(clean) == 2
    assert report["reason"].tolist() == ["duplicate_row"]
    assert not clean["integrity_break"].any()

    seg = add_segments(clean, cfg)
    assert segments(seg) == [0, 0]
    assert compute_ofi_level(seg, 1).iloc[1] == 50.0


def test_timestamp_ties_are_never_a_removal_reason():
    """Distinct updates sharing a timestamp must both survive cleaning."""
    cfg = cleaning_config()
    df = frame([0.0, 0.0], [100.0, 150.0, 175.0])
    assert df["timestamp"].nunique() == 1

    clean, report = clean_data(df, cfg)
    assert len(clean) == 3
    assert report.empty
    assert not clean["integrity_break"].any()


def test_removal_reasons_are_first_matching_and_sum_exactly():
    cfg = cleaning_config()
    n0 = 4
    df = frame([10.0] * 3, [100.0, 130.0, 150.0, 160.0],
               bid_px=[99.99, 100.5, 99.99, 99.99])
    df.loc[3, "bid_size_1"] = 0.0     # nonpositive size
    clean, report = clean_data(df, cfg)

    assert int(report["n_removed"].sum()) == n0 - len(clean)
    assert sorted(report["reason"]) == ["crossed_book", "nonpositive_size"]


def test_no_rolling_window_or_target_crosses_a_segment():
    """End-to-end: features and targets both respect the segment boundary."""
    from ofi_research.features import build_features
    from ofi_research.targets import add_targets, assert_target_causality

    cfg = Config()
    cfg.targets.clock_horizons_ms = [100.0]
    cfg.targets.event_horizons = [2]
    n = 40
    df = frame([50.0] * (n - 1), [100.0 + 5 * i for i in range(n)])
    df["book_clear"] = [False] * 20 + [True] + [False] * (n - 21)
    df["trade_price"] = np.nan
    df["trade_size"] = np.nan
    df["trade_side"] = np.nan
    df["event_type"] = "book_update"
    df["bid_price_2"] = np.nan
    df["ask_price_2"] = np.nan
    df["bid_size_2"] = np.nan
    df["ask_size_2"] = np.nan

    feat, _ = build_features(df, cfg)
    feat = add_targets(feat, cfg)
    assert_target_causality(feat, cfg)   # raises if any target spans a segment

    # the first row of every segment has a zero OFI increment by construction
    starts = feat.groupby("segment_id").head(1).index
    assert (feat.loc[starts, "OFI_level_1_increment"] == 0.0).all()
