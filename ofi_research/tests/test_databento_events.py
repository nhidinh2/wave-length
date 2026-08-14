"""Vendor-record ordering and canonical native-event boundaries (P0.1 / P0.2).

The claims under test are all about *what the vendor's records mean*, and every
one of them is an assumption listed in
:data:`ofi_research.databento_loader.PILOT_ASSUMPTIONS`. These tests pin the
implementation to the documented semantics; only a one-day real sample can
confirm the semantics themselves.
"""

import numpy as np
import pandas as pd
import pytest

from ofi_research.databento_loader import (
    RAW_IDENTITY_COLS, assign_event_ids, build_canonical_events,
    drop_ingestion_duplicates, prepare_raw, read_raw, sort_canonical,
    to_canonical_frame,
)
from ofi_research.dbn import F_LAST
from ofi_research.tests.dbn_fixtures import (
    MS, SEC, SESSION_OPEN_NS, databento_config, raw_frame, record, write_csv,
)


def _prepared(records):
    cfg = databento_config()
    return prepare_raw(raw_frame(records), cfg), cfg


# --------------------------------------------------------------------------- #
# Timestamp and sequence multiplicity: nothing is dropped for a tie
# --------------------------------------------------------------------------- #
def test_distinct_records_sharing_ts_recv_are_all_retained():
    """Three genuine updates captured in one packet must all survive.

    Dropping the second would not merely lose a row: the next OFI increment
    would then difference across a book state that was never observed.
    """
    ts = SESSION_OPEN_NS
    recs = [
        record(ts_recv=ts, sequence=1, size=10, bid=[(99.99, 100.0)]),
        record(ts_recv=ts, sequence=2, size=20, bid=[(99.99, 120.0)]),
        record(ts_recv=ts, sequence=3, size=30, bid=[(99.99, 150.0)]),
    ]
    prep, _ = _prepared(recs)
    kept, report = drop_ingestion_duplicates(prep)

    assert len(kept) == 3
    assert int(report["n_removed"].iloc[0]) == 0
    assert kept["ts_recv"].nunique() == 1


def test_records_sharing_sequence_are_retained_in_source_order():
    """Equal ``sequence`` is a normalization artifact, not a duplicate."""
    ts = SESSION_OPEN_NS
    recs = [
        record(ts_recv=ts, sequence=42, action="T", size=25, flags=0),
        record(ts_recv=ts, sequence=42, action="F", size=25, flags=0),
        record(ts_recv=ts, sequence=42, action="C", size=25, flags=F_LAST),
    ]
    prep, _ = _prepared(recs)
    kept, _ = drop_ingestion_duplicates(prep)

    assert len(kept) == 3
    assert kept["raw_row_id"].tolist() == [0, 1, 2]
    assert kept["action"].tolist() == ["T", "F", "C"]


def test_sort_is_stable_with_raw_row_id_as_final_tiebreaker():
    """A sort over tied keys must not permute native order (P0.1)."""
    ts = SESSION_OPEN_NS
    df = pd.DataFrame({
        "instrument": ["TEST"] * 4,
        "timestamp": pd.to_datetime([ts, ts, ts, ts + SEC], utc=True),
        "sequence": [5, 5, 5, 6],
        "raw_row_id": [0, 1, 2, 3],
        "bid_size_1": [100.0, 110.0, 120.0, 130.0],
    })
    out = sort_canonical(df.sample(frac=1.0, random_state=3))
    assert out["raw_row_id"].tolist() == [0, 1, 2, 3]
    assert out["bid_size_1"].tolist() == [100.0, 110.0, 120.0, 130.0]


def test_identical_reingested_record_is_removed():
    """A byte-identical re-ingest is the ONLY safe removal."""
    ts = SESSION_OPEN_NS
    r = record(ts_recv=ts, sequence=1, size=10)
    prep, _ = _prepared([r, dict(r), record(ts_recv=ts + SEC, sequence=2)])
    kept, report = drop_ingestion_duplicates(prep)

    assert len(kept) == 2
    assert int(report["n_removed"].iloc[0]) == 1


def test_same_identity_but_different_book_is_not_a_duplicate():
    """Identity columns alone are not enough — the resulting book must match."""
    ts = SESSION_OPEN_NS
    a = record(ts_recv=ts, sequence=1, size=10, bid=[(99.99, 100.0)])
    b = record(ts_recv=ts, sequence=1, size=10, bid=[(99.99, 140.0)])
    prep, _ = _prepared([a, b])
    kept, _ = drop_ingestion_duplicates(prep)
    assert len(kept) == 2


def test_ts_recv_is_never_part_of_a_dedup_key_on_its_own():
    """Guard the identity list itself: a timestamp-only key would be wrong."""
    assert "ts_recv" in RAW_IDENTITY_COLS
    assert len(RAW_IDENTITY_COLS) > 1
    for key in ("action", "side", "price", "size", "sequence"):
        assert key in RAW_IDENTITY_COLS


# --------------------------------------------------------------------------- #
# Canonical native events
# --------------------------------------------------------------------------- #
def test_trade_then_fill_with_f_last_becomes_one_canonical_event():
    """T -> F(F_LAST) is one native message normalized into two records."""
    ts = SESSION_OPEN_NS
    recs = [
        record(ts_recv=ts, action="T", side="B", price=100.01, size=25,
               flags=0, sequence=7,
               bid=[(99.99, 100.0)], ask=[(100.01, 100.0)]),
        record(ts_recv=ts, action="F", side="A", price=100.01, size=25,
               flags=F_LAST, sequence=7,
               bid=[(99.99, 100.0)], ask=[(100.01, 75.0)]),
    ]
    prep, cfg = _prepared(recs)
    assert assign_event_ids(prep).tolist() == [0, 0]

    ev, diag = build_canonical_events(prep, cfg)
    assert diag["n_raw_records"] == 2
    assert diag["n_canonical_events"] == 1
    assert int(ev["n_records"].iloc[0]) == 2
    assert diag["n_events_mixed_sequence"] == 0


def test_signed_volume_from_trades_but_book_from_the_last_record():
    """A ``T`` record does not carry the book; its sibling does."""
    ts = SESSION_OPEN_NS
    recs = [
        record(ts_recv=ts, action="T", side="B", price=100.01, size=25,
               flags=0, sequence=7,
               bid=[(99.99, 100.0)], ask=[(100.01, 100.0)]),
        record(ts_recv=ts, action="F", side="A", price=100.01, size=25,
               flags=F_LAST, sequence=7,
               bid=[(99.99, 100.0)], ask=[(100.01, 75.0)]),
    ]
    prep, cfg = _prepared(recs)
    ev, _ = build_canonical_events(prep, cfg)
    can = to_canonical_frame(ev, cfg)

    # +25 from the buy-aggressor trade, counted once, not twice
    assert can["signed_trade_volume"].iloc[0] == 25.0
    assert can["trade_size"].iloc[0] == 25.0
    # post-event book comes from the LAST record of the event
    assert can["ask_size_1"].iloc[0] == 75.0
    assert can["ask_price_1"].iloc[0] == pytest.approx(100.01)


def test_buy_and_sell_aggressor_trades_net_within_one_event():
    """Signed volume is carried explicitly because it can net to anything."""
    ts = SESSION_OPEN_NS
    recs = [
        record(ts_recv=ts, action="T", side="B", size=30, flags=0, sequence=9),
        record(ts_recv=ts, action="T", side="A", size=10, flags=0, sequence=9),
        record(ts_recv=ts, action="C", side="A", size=10, flags=F_LAST,
               sequence=9, ask=[(100.01, 60.0)]),
    ]
    prep, cfg = _prepared(recs)
    ev, _ = build_canonical_events(prep, cfg)
    assert ev["signed_trade_volume"].iloc[0] == 20.0   # +30 - 10
    assert ev["trade_size"].iloc[0] == 40.0            # gross volume
    assert int(ev["n_trades"].iloc[0]) == 2


def test_unspecified_aggressor_side_is_counted_not_guessed():
    """``side == 'N'`` contributes zero signed volume and is reported."""
    ts = SESSION_OPEN_NS
    recs = [
        record(ts_recv=ts, action="T", side="N", size=15, flags=F_LAST,
               sequence=3),
    ]
    prep, cfg = _prepared(recs)
    ev, diag = build_canonical_events(prep, cfg)
    assert ev["signed_trade_volume"].iloc[0] == 0.0
    assert diag["n_unspecified_side_trades"] == 1


def test_equal_sequence_is_never_pooled_across_non_consecutive_rows():
    """Grouping is over CONSECUTIVE rows only — no global sequence groupby."""
    ts = SESSION_OPEN_NS
    recs = [
        record(ts_recv=ts, sequence=5, flags=F_LAST),
        record(ts_recv=ts + SEC, sequence=6, flags=F_LAST),
        record(ts_recv=ts + 2 * SEC, sequence=5, flags=F_LAST),  # reused
    ]
    prep, _ = _prepared(recs)
    assert assign_event_ids(prep).tolist() == [0, 1, 2]


def test_event_boundaries_fall_back_to_sequence_runs_without_f_last():
    ts = SESSION_OPEN_NS
    recs = [
        record(ts_recv=ts, sequence=1, flags=0),
        record(ts_recv=ts, sequence=1, flags=0),
        record(ts_recv=ts + SEC, sequence=2, flags=0),
    ]
    prep, _ = _prepared(recs)
    assert assign_event_ids(prep).tolist() == [0, 0, 1]


def test_events_never_span_an_instrument_or_session_change():
    ts = SESSION_OPEN_NS
    next_day = ts + 24 * 3600 * 1_000_000_000
    recs = [
        record(ts_recv=ts, sequence=1, flags=0, symbol="AAA"),
        record(ts_recv=ts, sequence=1, flags=0, symbol="BBB"),
        record(ts_recv=next_day, sequence=1, flags=0, symbol="BBB"),
    ]
    prep, _ = _prepared(recs)
    # no F_LAST anywhere and identical sequences, so ONLY the group change
    # can separate these; it must.
    assert assign_event_ids(prep).tolist() == [0, 1, 2]


def test_ofi_is_computed_between_consecutive_final_event_states():
    """Two events, one book change: the increment uses post-event states."""
    from ofi_research.features import add_segments, compute_ofi_level

    ts = SESSION_OPEN_NS
    recs = [
        # event 0: intermediate record has a DIFFERENT book than the final one
        record(ts_recv=ts, action="A", flags=0, sequence=1,
               bid=[(99.99, 999.0)], ask=[(100.01, 100.0)]),
        record(ts_recv=ts, action="A", flags=F_LAST, sequence=1,
               bid=[(99.99, 100.0)], ask=[(100.01, 100.0)]),
        # event 1: bid size 100 -> 150 at an unchanged price => OFI +50
        record(ts_recv=ts + SEC, action="A", flags=F_LAST, sequence=2,
               bid=[(99.99, 150.0)], ask=[(100.01, 100.0)]),
    ]
    prep, cfg = _prepared(recs)
    ev, _ = build_canonical_events(prep, cfg)
    can = sort_canonical(to_canonical_frame(ev, cfg))
    can = add_segments(can, cfg)
    ofi = compute_ofi_level(can, 1)

    assert len(can) == 2
    assert ofi.iloc[0] == 0.0        # first state of the segment
    assert ofi.iloc[1] == 50.0       # not 999-derived: intermediates ignored


def test_deep_levels_map_to_canonical_columns():
    ts = SESSION_OPEN_NS
    recs = [record(ts_recv=ts, flags=F_LAST, sequence=1,
                   bid=[(99.99, 100.0), (99.98, 200.0), (99.97, 300.0)],
                   ask=[(100.01, 110.0), (100.02, 210.0), (100.03, 310.0)])]
    prep, cfg = _prepared(recs)
    ev, _ = build_canonical_events(prep, cfg)
    can = to_canonical_frame(ev, cfg)

    assert can["bid_price_1"].iloc[0] == pytest.approx(99.99)
    assert can["bid_price_3"].iloc[0] == pytest.approx(99.97)
    assert can["ask_size_2"].iloc[0] == 210.0
    # levels the vendor did not send stay missing rather than being invented
    assert np.isnan(can["bid_price_4"].iloc[0])


# --------------------------------------------------------------------------- #
# Multi-file ingestion
# --------------------------------------------------------------------------- #
def test_concatenating_files_preserves_deterministic_order(tmp_path):
    """Order follows the argument list, not the filesystem."""
    a = write_csv([record(ts_recv=SESSION_OPEN_NS + i * SEC, sequence=i,
                          size=i + 1) for i in range(3)], tmp_path / "a.csv")
    b = write_csv([record(ts_recv=SESSION_OPEN_NS + (10 + i) * SEC,
                          sequence=10 + i, size=100 + i) for i in range(3)],
                  tmp_path / "b.csv")

    ab = read_raw([a, b])
    assert ab["raw_row_id"].tolist() == list(range(6))
    assert ab["raw_row_id"].is_monotonic_increasing
    assert ab["_source_index"].tolist() == [0, 0, 0, 1, 1, 1]
    assert ab["sequence"].tolist() == [0, 1, 2, 10, 11, 12]

    ba = read_raw([b, a])
    assert ba["sequence"].tolist() == [10, 11, 12, 0, 1, 2]
    # re-reading the same argument list is reproducible
    assert read_raw([a, b])["sequence"].tolist() == ab["sequence"].tolist()


def test_parquet_indexed_by_ts_recv_is_read_correctly(tmp_path):
    """Databento's exporters index by ``ts_recv``; it must come back as a column.

    ``DBNStore.to_parquet`` writes the frame's ``ts_recv`` index into the
    file's pandas metadata, so a naive read leaves the causal clock in the
    index where every ``df['ts_recv']`` lookup misses it.
    """
    recs = [record(ts_recv=SESSION_OPEN_NS + i * SEC, sequence=i, size=i + 1)
            for i in range(4)]
    path = tmp_path / "indexed.parquet"
    pd.DataFrame(recs).set_index("ts_recv").to_parquet(path)

    raw = read_raw([path])
    assert "ts_recv" in raw.columns
    assert raw["raw_row_id"].tolist() == [0, 1, 2, 3]
    assert raw["sequence"].tolist() == [0, 1, 2, 3]   # order preserved

    prep = prepare_raw(raw, databento_config())
    assert prep["ts_recv"].is_monotonic_increasing


def test_reading_a_plain_parquet_is_unaffected(tmp_path):
    recs = [record(ts_recv=SESSION_OPEN_NS + i * SEC, sequence=i)
            for i in range(3)]
    path = tmp_path / "plain.parquet"
    pd.DataFrame(recs).to_parquet(path, index=False)

    raw = read_raw([path])
    assert "ts_recv" in raw.columns
    assert raw["raw_row_id"].tolist() == [0, 1, 2]


def test_overlapping_downloads_do_not_duplicate_records(tmp_path):
    """The same file twice must collapse back to the original record set."""
    recs = [record(ts_recv=SESSION_OPEN_NS + i * SEC, sequence=i, size=i + 1)
            for i in range(4)]
    path = write_csv(recs, tmp_path / "day.csv")

    cfg = databento_config()
    doubled = prepare_raw(read_raw([path, path]), cfg)
    assert len(doubled) == 8

    kept, report = drop_ingestion_duplicates(doubled)
    assert len(kept) == 4
    assert int(report["n_removed"].iloc[0]) == 4
    assert kept["sequence"].tolist() == [0, 1, 2, 3]
