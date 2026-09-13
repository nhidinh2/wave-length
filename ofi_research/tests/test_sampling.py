"""Memory levers: representation, pruning, and the decision clock.

The two things that must not happen: a column something downstream reads gets
pruned away silently, and sampling changes what a retained row's target means.
"""

import numpy as np
import pandas as pd
import pytest

from ofi_research.config import Config
from ofi_research.data_loader import make_synthetic_data
from ofi_research.dbn import INT_PRICE_SUFFIX
from ofi_research.features import build_features
from ofi_research.sampling import (
    compact_dtypes, prune_columns, required_columns, sample_decision_rows,
)
from ofi_research.targets import add_execution_aligned_targets, add_targets
from ofi_research.validation import clean_data


def _pipeline_to_targets(cfg, n_days=2, events=400):
    df, cfg2 = make_synthetic_data(n_days=n_days, events_per_day=events, seed=3)
    # keep the caller's sampling/target settings, take the generator's columns
    cfg2.sampling = cfg.sampling
    cfg2.targets = cfg.targets
    clean, _ = clean_data(df, cfg2)
    feat, _ = build_features(clean, cfg2)
    feat = compact_dtypes(feat, cfg2)
    feat = add_targets(feat, cfg2)
    for H in cfg2.targets.clock_horizons_ms:
        feat = add_execution_aligned_targets(feat, cfg2, float(H))
    return feat, cfg2


# --------------------------------------------------------------------------- #
# compact_dtypes — representation only
# --------------------------------------------------------------------------- #
def test_fixed_point_integer_copies_are_released():
    cfg = Config()
    df = pd.DataFrame({
        "bid_price_1": [99.99, 100.0],
        f"bid_price_1{INT_PRICE_SUFFIX}": pd.array([99_990_000_000,
                                                    100_000_000_000],
                                                   dtype="Int64"),
    })
    out = compact_dtypes(df, cfg)
    assert f"bid_price_1{INT_PRICE_SUFFIX}" not in out.columns
    assert "bid_price_1" in out.columns


def test_price_levels_stay_float64_and_other_floats_shrink():
    """float32 at ~$100 is ~0.08% of a tick — fine for OFI, not for prices."""
    cfg = Config()
    df = pd.DataFrame({
        "bid_price_1": np.array([99.99, 100.01], dtype="float64"),
        "midprice": np.array([100.0, 100.02], dtype="float64"),
        "OFI_L1_ref": np.array([1500.0, -220.0], dtype="float64"),
        "depth_imbalance_L1": np.array([0.3, -0.1], dtype="float64"),
    })
    out = compact_dtypes(df, cfg)

    assert out["bid_price_1"].dtype == "float64"
    assert out["midprice"].dtype == "float64"
    assert out["OFI_L1_ref"].dtype == "float32"
    assert out["depth_imbalance_L1"].dtype == "float32"


def test_price_values_are_bit_identical_after_compaction():
    cfg = Config()
    px = np.array([99.99, 100.01, 100.07, 250.13], dtype="float64")
    out = compact_dtypes(pd.DataFrame({"bid_price_1": px.copy()}), cfg)
    assert (out["bid_price_1"].to_numpy() == px).all()


def test_repeated_strings_become_categories():
    cfg = Config()
    df = pd.DataFrame({"instrument": ["INTC"] * 500,
                       "event_type": ["book_update"] * 500})
    before = df.memory_usage(deep=True).sum()
    out = compact_dtypes(df, cfg)

    assert str(out["instrument"].dtype) == "category"
    assert out.memory_usage(deep=True).sum() < before
    assert out["instrument"].tolist() == ["INTC"] * 500


def test_compaction_can_be_switched_off():
    cfg = Config()
    cfg.sampling.downcast_float32 = False
    cfg.sampling.categorical_strings = False
    cfg.sampling.drop_fixed_price_integers = False
    df = pd.DataFrame({
        "OFI_L1_ref": np.array([1.0, 2.0], dtype="float64"),
        f"bid_price_1{INT_PRICE_SUFFIX}": pd.array([1, 2], dtype="Int64"),
        "instrument": ["A", "A"],
    })
    out = compact_dtypes(df, cfg)
    assert out["OFI_L1_ref"].dtype == "float64"
    assert f"bid_price_1{INT_PRICE_SUFFIX}" in out.columns
    assert out["instrument"].dtype == object


# --------------------------------------------------------------------------- #
# prune_columns — nothing downstream reads may disappear
# --------------------------------------------------------------------------- #
def test_every_model_feature_survives_pruning():
    from ofi_research.models import model_feature_sets

    cfg = Config()
    feat, cfg2 = _pipeline_to_targets(cfg)
    pruned = prune_columns(feat, cfg2)

    for name, feats in model_feature_sets(cfg2).items():
        for f in feats:
            if f in feat.columns:
                assert f in pruned.columns, f"{name} lost {f}"


def test_depth_ladder_survives_for_taker_execution():
    cfg = Config()
    feat, cfg2 = _pipeline_to_targets(cfg)
    pruned = prune_columns(feat, cfg2)
    for k in (1, 2):
        for pat in (f"bid_price_{k}", f"ask_price_{k}",
                    f"bid_size_{k}", f"ask_size_{k}"):
            if pat in feat.columns:
                assert pat in pruned.columns


def test_target_columns_survive_pruning():
    cfg = Config()
    cfg.targets.clock_horizons_ms = [1000.0]
    cfg.targets.event_horizons = [10]
    feat, cfg2 = _pipeline_to_targets(cfg)
    pruned = prune_columns(feat, cfg2)

    for col in ("future_mid_change_ms1000", "actual_horizon_ms_ms1000",
                "future_mid_change_ev10", "exec_entry_index_ms1000",
                "future_mid_change_exec_ms1000"):
        assert col in pruned.columns, col


def test_event_window_elapsed_columns_survive_pruning():
    """Regression: these feed the horizon/window realization table (P0.3).

    An event window is not a fixed amount of clock time. Pruning these emptied
    the table silently rather than failing.
    """
    cfg = Config()
    feat, cfg2 = _pipeline_to_targets(cfg)
    pruned = prune_columns(feat, cfg2)

    cols = [c for c in feat.columns if c.startswith("event_window_elapsed_ms")]
    assert cols, "fixture produced no event-window columns"
    for c in cols:
        assert c in pruned.columns


def test_pruning_actually_drops_something_and_can_be_disabled():
    cfg = Config()
    feat, cfg2 = _pipeline_to_targets(cfg)
    assert prune_columns(feat, cfg2).shape[1] < feat.shape[1]

    cfg2.sampling.prune_columns = False
    assert prune_columns(feat, cfg2).shape[1] == feat.shape[1]


# --------------------------------------------------------------------------- #
# sample_decision_rows — a decision clock
# --------------------------------------------------------------------------- #
def _grid(n, step_ms, segment=1):
    ts = pd.date_range("2026-08-05T13:30:00Z", periods=n,
                       freq=f"{step_ms}ms")
    return pd.DataFrame({
        "timestamp": ts,
        "segment_id": [segment] * n,
        "midprice": np.arange(float(n)),
    })


def test_sampling_is_a_no_op_when_no_interval_is_set():
    cfg = Config()
    assert cfg.sampling.decision_interval_ms is None
    df = _grid(50, 10)
    assert len(sample_decision_rows(df, cfg)) == 50


def test_one_row_is_retained_per_decision_interval():
    cfg = Config()
    cfg.sampling.decision_interval_ms = 100.0
    df = _grid(100, 10)          # 100 rows spanning 1000 ms
    out = sample_decision_rows(df, cfg)

    assert len(out) == 10
    # the first event of each bucket, so 10 ms apart x 10
    assert out["midprice"].tolist() == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]


def test_sampling_preserves_order_and_never_invents_rows():
    cfg = Config()
    cfg.sampling.decision_interval_ms = 50.0
    df = _grid(200, 7)
    out = sample_decision_rows(df, cfg)

    assert out["timestamp"].is_monotonic_increasing
    assert set(out["midprice"]).issubset(set(df["midprice"]))


def test_every_segment_start_is_retained():
    """A segment must never begin mid-bucket."""
    cfg = Config()
    cfg.sampling.decision_interval_ms = 100.0
    a = _grid(30, 10, segment=1)
    b = _grid(30, 10, segment=2)
    b["timestamp"] = a["timestamp"] + pd.Timedelta(milliseconds=5)
    df = pd.concat([a, b]).sort_values("timestamp").reset_index(drop=True)

    out = sample_decision_rows(df, cfg)
    for seg, g in df.groupby("segment_id"):
        first_ts = g["timestamp"].iloc[0]
        kept = out[out["segment_id"] == seg]
        assert kept["timestamp"].iloc[0] == first_ts


def test_targets_on_retained_rows_are_unchanged_by_sampling():
    """The point of sampling last: a retained row's target is still exact.

    If sampling ran before the targets, a 1 s target would resolve to the next
    *sampled* row rather than the next event, quantizing the realized horizon.
    """
    cfg = Config()
    cfg.targets.clock_horizons_ms = [1000.0]
    cfg.targets.event_horizons = []
    cfg.sampling.decision_interval_ms = 500.0
    feat, cfg2 = _pipeline_to_targets(cfg, n_days=1, events=2000)

    out = sample_decision_rows(feat, cfg2)
    assert 0 < len(out) < len(feat)

    merged = out[["timestamp", "future_mid_change_ms1000",
                  "actual_horizon_ms_ms1000"]].merge(
        feat[["timestamp", "future_mid_change_ms1000",
              "actual_horizon_ms_ms1000"]],
        on="timestamp", suffixes=("_sampled", "_full"))
    a = merged["future_mid_change_ms1000_sampled"]
    b = merged["future_mid_change_ms1000_full"]
    assert ((a == b) | (a.isna() & b.isna())).all()

    # realized horizons stay off the sampling grid, i.e. genuinely unquantized
    realized = out["actual_horizon_ms_ms1000"].dropna()
    assert len(realized)
    assert not np.allclose(realized % 500.0, 0.0)


# --------------------------------------------------------------------------- #
# copy=False must not change results, only memory
# --------------------------------------------------------------------------- #
def test_execution_targets_identical_with_and_without_copying():
    """``copy=False`` is a memory switch, never a semantic one."""
    cfg = Config()
    cfg.targets.clock_horizons_ms = [500.0, 1000.0]
    cfg.targets.event_horizons = []
    base, cfg2 = _pipeline_to_targets(cfg, n_days=1, events=800)

    copied = base.copy()
    for H in cfg2.targets.clock_horizons_ms:
        copied = add_execution_aligned_targets(copied, cfg2, float(H),
                                               copy=True)
    inplace = base.copy()
    for H in cfg2.targets.clock_horizons_ms:
        inplace = add_execution_aligned_targets(inplace, cfg2, float(H),
                                                copy=False)

    for tag in ("ms500", "ms1000"):
        for col in (f"exec_entry_index_{tag}", f"exec_exit_index_{tag}",
                    f"future_mid_change_exec_{tag}"):
            a, b = copied[col], inplace[col]
            assert ((a == b) | (a.isna() & b.isna())).all(), col


def test_copy_true_leaves_the_caller_frame_untouched():
    cfg = Config()
    cfg.targets.clock_horizons_ms = [1000.0]
    cfg.targets.event_horizons = []
    base, cfg2 = _pipeline_to_targets(cfg, n_days=1, events=400)
    before = set(base.columns)

    add_execution_aligned_targets(base, cfg2, 1000.0, copy=True)
    assert set(base.columns) == before
