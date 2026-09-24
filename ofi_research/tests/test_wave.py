"""The wave / Duhamel layer: an exact, causal, segment-reset convolution."""

from __future__ import annotations

import numpy as np
import pytest

from ofi_research.config import Config
from ofi_research.features import _causal_kernel_filter, wave_feature_names
from ofi_research.models import LADDER_ORDER, model_feature_sets


def _naive(t_ns, seg, f, tau_ms, period_ms=None):
    """O(n^2) definition, to pin the chunked evaluation against."""
    lam = complex(1000.0 / tau_ms,
                  -(2 * np.pi / (period_ms / 1000.0)) if period_ms else 0.0)
    out = np.zeros(len(f), dtype=complex)
    for k in range(len(f)):
        for j in range(k + 1):
            if seg[j] == seg[k]:
                out[k] += np.exp(-lam * (t_ns[k] - t_ns[j]) / 1e9) * f[j]
    return out


@pytest.fixture()
def tape():
    rng = np.random.default_rng(0)
    n = 300
    # bursty clock: sub-ms clusters and multi-second lulls, plus ties
    dt = rng.choice([0, 50_000, 2_000_000, 400_000_000, 3_000_000_000], size=n)
    t_ns = np.cumsum(dt).astype(np.int64)
    seg = np.repeat([0, 1, 2], [120, 100, 80])
    return t_ns, seg, rng.normal(size=n) * 100


@pytest.mark.parametrize("tau,period", [(100.0, None), (400.0, None),
                                        (1000.0, 2000.0)])
def test_matches_the_definition(tape, tau, period):
    t_ns, seg, f = tape
    # a tiny chunk forces many carries across chunk boundaries
    got = _causal_kernel_filter(t_ns, seg, f, tau, period, chunk_taus=0.5)
    assert np.allclose(got, _naive(t_ns, seg, f, tau, period),
                       rtol=1e-9, atol=1e-9)


def test_is_causal_under_truncation(tape):
    t_ns, seg, f = tape
    full = _causal_kernel_filter(t_ns, seg, f, 400.0, 2000.0)
    part = _causal_kernel_filter(t_ns[:150], seg[:150], f[:150], 400.0, 2000.0)
    assert np.allclose(full[:150], part, rtol=1e-12, atol=1e-12)


def test_resets_at_segment_boundaries():
    t_ns = np.arange(4, dtype=np.int64) * 1_000_000
    seg = np.array([0, 0, 1, 1])
    got = _causal_kernel_filter(t_ns, seg, np.array([5.0, 0, 0, 0]), 1000.0)
    assert got[1].real == pytest.approx(5 * np.exp(-0.001))
    assert got[2] == 0 and got[3] == 0


def test_oscillator_kernel_changes_sign():
    """A unit impulse through the sine kernel swings positive then negative."""
    t_ns = np.array([0, 500_000_000, 1_500_000_000], dtype=np.int64)
    got = _causal_kernel_filter(t_ns, np.zeros(3, int),
                                np.array([1.0, 0, 0]), 1000.0, 2000.0)
    assert got[1].imag > 0 > got[2].imag      # sin(π/2) > 0 > sin(3π/2)


def test_m6_is_m5_plus_the_wave_group_only():
    cfg = Config()
    s = model_feature_sets(cfg)
    assert s["M6_wave"] == s["M5_full"] + wave_feature_names(cfg)
    assert LADDER_ORDER[-2:] == ["M5_full", "M6_wave"]
