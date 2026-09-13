"""Unit tests for the Level-1 OFI increment (section 6 / 28).

Every case uses a hand-constructed two-row book and states the expected OFI
value explicitly, derived from:

    bid_contrib = I(bp>=bp0)*bs - I(bp<=bp0)*bs0
    ask_contrib = -I(ap<=ap0)*as + I(ap>=ap0)*as0
    increment   = bid_contrib + ask_contrib
"""

import numpy as np
import pandas as pd
import pytest

from ofi_research.config import Config
from ofi_research.features import (
    _ofi_increment_arrays, add_segments, compute_ofi_level, build_features,
)


def _incr(prev, now):
    """prev/now = (bid_p, bid_s, ask_p, ask_s). Return increment at row 1."""
    bid_p = np.array([prev[0], now[0]], dtype=float)
    bid_s = np.array([prev[1], now[1]], dtype=float)
    ask_p = np.array([prev[2], now[2]], dtype=float)
    ask_s = np.array([prev[3], now[3]], dtype=float)
    out = _ofi_increment_arrays(bid_p, bid_s, ask_p, ask_s)
    assert out[0] == 0.0  # first observation in a segment has no increment
    return out[1]


def test_bid_price_increase():
    # bid 10->10.5 (bs 50), ask 11 unchanged (100->80)
    assert _incr((10, 100, 11, 100), (10.5, 50, 11, 80)) == 70.0


def test_bid_price_decrease():
    # bid 10->9.5 (loses old 100), ask unchanged same size -> 0
    assert _incr((10, 100, 11, 100), (9.5, 50, 11, 100)) == -100.0


def test_bid_unchanged_size_change():
    # bid price flat, size 100->150 => +50; ask flat same size => 0
    assert _incr((10, 100, 11, 100), (10, 150, 11, 100)) == 50.0


def test_ask_price_increase():
    # ask lifts 11->11.5 => +old ask size 100; bid flat same size => 0
    assert _incr((10, 100, 11, 100), (10, 100, 11.5, 80)) == 100.0


def test_ask_price_decrease():
    # ask drops 11->10.5 => -new ask size 80; bid flat => 0
    assert _incr((10, 100, 11, 100), (10, 100, 10.5, 80)) == -80.0


def test_ask_unchanged_size_change():
    # ask flat, size 100->140 => -40; bid flat => 0
    assert _incr((10, 100, 11, 100), (10, 100, 11, 140)) == -40.0


def test_both_sides_change():
    # bid up (+60), ask down (-70) => -10
    assert _incr((10, 100, 11, 100), (10.5, 60, 10.8, 70)) == -10.0


def test_new_segment_resets_increment():
    # two segments; first row of each must have increment 0
    cfg = Config()
    df = pd.DataFrame({
        "instrument": ["A", "A", "B", "B"],
        "timestamp": pd.to_datetime(
            ["2024-01-01T09:00:00Z", "2024-01-01T09:00:01Z",
             "2024-01-01T09:00:02Z", "2024-01-01T09:00:03Z"]),
        "session_date": [pd.Timestamp("2024-01-01").date()] * 4,
        "sequence": [np.nan] * 4,
        "bid_price_1": [10, 10.5, 20, 20.5],
        "ask_price_1": [11, 11, 21, 21],
        "bid_size_1": [100, 50, 100, 50],
        "ask_size_1": [100, 80, 100, 80],
    })
    df = add_segments(df, cfg)
    ofi = compute_ofi_level(df, 1)
    # instrument change => B's first row resets
    assert ofi.iloc[0] == 0.0
    assert ofi.iloc[2] == 0.0
    assert ofi.iloc[1] == 70.0  # same as bid-increase case
    assert ofi.iloc[3] == 70.0


def test_new_day_resets_increment():
    cfg = Config()
    df = pd.DataFrame({
        "instrument": ["A"] * 3,
        "timestamp": pd.to_datetime(
            ["2024-01-01T09:00:00Z", "2024-01-02T09:00:00Z",
             "2024-01-02T09:00:01Z"]),
        "session_date": [pd.Timestamp("2024-01-01").date(),
                         pd.Timestamp("2024-01-02").date(),
                         pd.Timestamp("2024-01-02").date()],
        "sequence": [np.nan] * 3,
        "bid_price_1": [10, 10, 10.5],
        "ask_price_1": [11, 11, 11],
        "bid_size_1": [100, 100, 50],
        "ask_size_1": [100, 100, 80],
    })
    df = add_segments(df, cfg)
    ofi = compute_ofi_level(df, 1)
    assert ofi.iloc[1] == 0.0  # new day resets, no increment vs prior day
    assert ofi.iloc[2] == 70.0


def test_bid_contribution_telescopes_when_prices_are_constant():
    """With a flat bid price, the summed contribution is a pure size delta.

    Every increment reduces to ``bs - bs0``, so the sum over a segment must
    collapse to ``final - initial``. If the indicator convention were wrong on
    the equality case, this identity would fail while every single-step test
    still passed.
    """
    rng = np.random.default_rng(11)
    sizes = rng.integers(10, 500, size=200).astype(float)
    n = len(sizes)
    flat = np.full(n, 10.0)
    ones = np.ones(n)

    bid_only = _ofi_increment_arrays(flat, sizes, np.full(n, 11.0), ones)
    # the ask side contributes -(1) + (1) = 0 at every step with flat price
    assert bid_only.sum() == pytest.approx(sizes[-1] - sizes[0])


def test_ask_contribution_telescopes_with_reversed_sign():
    rng = np.random.default_rng(12)
    sizes = rng.integers(10, 500, size=200).astype(float)
    n = len(sizes)
    ones = np.ones(n)

    ask_only = _ofi_increment_arrays(np.full(n, 10.0), ones,
                                     np.full(n, 11.0), sizes)
    assert ask_only.sum() == pytest.approx(-(sizes[-1] - sizes[0]))


def test_levels_are_computed_independently_from_their_own_columns():
    """L2 must read L2 columns; a change at one level must not leak to the other."""
    cfg = Config()
    df = pd.DataFrame({
        "instrument": ["A"] * 3,
        "timestamp": pd.to_datetime(
            ["2024-01-01T09:00:00Z", "2024-01-01T09:00:01Z",
             "2024-01-01T09:00:02Z"]),
        "session_date": [pd.Timestamp("2024-01-01").date()] * 3,
        "sequence": [np.nan] * 3,
        # row 1 changes L1 only; row 2 changes L2 only
        "bid_price_1": [10.0, 10.0, 10.0],
        "ask_price_1": [11.0, 11.0, 11.0],
        "bid_size_1": [100.0, 160.0, 160.0],
        "ask_size_1": [100.0, 100.0, 100.0],
        "bid_price_2": [9.0, 9.0, 9.0],
        "ask_price_2": [12.0, 12.0, 12.0],
        "bid_size_2": [200.0, 200.0, 275.0],
        "ask_size_2": [200.0, 200.0, 200.0],
    })
    df = add_segments(df, cfg)
    l1 = compute_ofi_level(df, 1)
    l2 = compute_ofi_level(df, 2)

    assert l1.tolist() == [0.0, 60.0, 0.0]
    assert l2.tolist() == [0.0, 0.0, 75.0]


def test_missing_l2_disables_l2_ofi():
    from ofi_research.data_loader import make_synthetic_data
    df, cfg = make_synthetic_data(n_days=1, events_per_day=200, seed=1)
    df[["bid_price_2", "ask_price_2", "bid_size_2", "ask_size_2"]] = np.nan
    feat, notes = build_features(df, cfg)
    assert (feat["OFI_level_2_increment"] == 0.0).all()
    assert any("Level-2" in n for n in notes)
