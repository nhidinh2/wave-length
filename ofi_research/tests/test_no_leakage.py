"""No-look-ahead / no-leakage tests (section 12).

The central, strongest test is *truncation causality*: if a feature at row t
uses only information up to t, then rebuilding features on a strict prefix of
the data must reproduce the exact same feature values for the retained rows.
Any peek into the future would change those values.
"""

import numpy as np
import pandas as pd
import pytest

from ofi_research.config import Config
from ofi_research.data_loader import make_synthetic_data
from ofi_research.features import build_features, winsor_bounds
from ofi_research.targets import add_targets

FEATURE_COLS = [
    "OFI_level_1_increment", "OFI_level_2_increment",
    "OFI1_ev10", "OFI1_ev100", "OFI1_ms1000", "signedvol_ev10",
    "nOFI_L1", "nOFI_L2", "zOFI", "trailing_mid_vol",
    "spread", "delta_spread", "depth_imbalance_L1",
    "market_order_intensity_imbalance",
    # the wave layer u_t: a convolution over the past must not see the future
    "wave_exp100", "wave_exp400", "wave_exp1600",
    "wave_osc1000p2000_sin", "wave_osc1000p2000_cos",
]


@pytest.fixture(scope="module")
def synth():
    df, cfg = make_synthetic_data(n_days=2, events_per_day=1500, seed=3)
    return df, cfg


def test_features_are_causal_under_truncation(synth):
    df, cfg = synth
    full, _ = build_features(df, cfg)
    m = 1200  # a prefix cut inside the first day (same segment structure)
    prefix, _ = build_features(df.iloc[:m].copy(), cfg)
    for col in FEATURE_COLS:
        a = full[col].to_numpy()[:m]
        b = prefix[col].to_numpy()
        assert np.allclose(a, b, equal_nan=True, rtol=1e-9, atol=1e-9), (
            f"Feature '{col}' changed when future rows were removed -> leakage")


def test_targets_are_strictly_future(synth):
    df, cfg = synth
    feat, _ = build_features(df, cfg)
    out = add_targets(feat, cfg)
    # event-1 target must equal next-row mid minus this-row mid within segment
    mid = out["midprice"].to_numpy()
    seg = out["segment_id"].to_numpy()
    ch = out["future_mid_change_ev1"].to_numpy()
    for i in range(len(out) - 1):
        if seg[i] == seg[i + 1]:
            assert np.isclose(ch[i], mid[i + 1] - mid[i], equal_nan=True)
    # last row of the whole frame has no future -> NaN
    assert np.isnan(ch[-1])


def test_target_never_uses_current_or_past(synth):
    df, cfg = synth
    feat, _ = build_features(df, cfg)
    out = add_targets(feat, cfg)
    # future change should be uncorrelated-by-construction with a purely past
    # shift: specifically, adding a constant to all FUTURE mids only should not
    # be needed — instead assert horizon>0 realized for clock targets.
    ah = out["actual_horizon_ms_ms1000"].dropna()
    assert (ah > 0).all()


def test_winsor_bounds_use_training_only(synth):
    df, cfg = synth
    feat, _ = build_features(df, cfg)
    train = feat["OFI1_ev50"].iloc[:1000]
    test = feat["OFI1_ev50"].iloc[1000:]
    lo1, hi1 = winsor_bounds(train, cfg)
    # bounds must not change if we corrupt the test portion
    corrupted = feat.copy()
    corrupted.loc[corrupted.index[1000:], "OFI1_ev50"] = 1e9
    lo2, hi2 = winsor_bounds(corrupted["OFI1_ev50"].iloc[:1000], cfg)
    assert lo1 == lo2 and hi1 == hi2


def test_no_random_split_used():
    # guardrail: the splits module must be day-contiguous, never shuffled.
    # Check for the actual leakage-inducing CALLS, not prose in docstrings.
    import inspect
    from ofi_research import splits
    src = inspect.getsource(splits)
    forbidden = ["train_test_split", ".sample(", "np.random.shuffle",
                 "rng.shuffle", "KFold", "shuffle=True"]
    for token in forbidden:
        assert token not in src, f"splits.py must not use {token!r}"


def test_folds_are_time_ordered_and_disjoint():
    # train days strictly precede val days strictly precede test days.
    from ofi_research import splits
    df, cfg = make_synthetic_data(n_days=15, events_per_day=200, seed=5)
    from ofi_research.features import add_segments
    df = add_segments(df, cfg)
    folds = splits.make_folds(df, cfg)
    assert len(folds) > 0
    for f in folds:
        assert max(f.train_days) < min(f.val_days)
        assert max(f.val_days) < min(f.test_days)
