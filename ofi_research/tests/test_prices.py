"""Price representation and conversion (P0.6).

The failure this guards against is silent: a fixed-point price divided twice
becomes 1e-7 dollars, one divided zero times becomes 1e11, and either will
produce a spread, a mid and a P&L number without raising anything. Every path
here therefore either converts exactly once or fails loudly.
"""

import numpy as np
import pandas as pd
import pytest

from ofi_research.dbn import (
    FIXED_PRICE_SCALE, UNDEF_PRICE, PriceConversionError,
    assert_plausible_prices, assert_prices_converted_once, infer_tick_size,
    to_dollars, to_fixed_int,
)
from ofi_research.databento_loader import prepare_raw
from ofi_research.tests.dbn_fixtures import (
    SEC, SESSION_OPEN_NS, databento_config, fixed, raw_frame, record,
)


def test_fixed_input_is_converted_exactly_once():
    raw = pd.Series([fixed(100.01), fixed(99.99), fixed(250.50)])
    out = to_dollars(raw, "fixed")
    assert out.tolist() == pytest.approx([100.01, 99.99, 250.50])


def test_float_input_is_not_divided():
    raw = pd.Series([100.01, 99.99, 250.50])
    out = to_dollars(raw, "float")
    assert out.tolist() == pytest.approx([100.01, 99.99, 250.50])


def test_price_mode_must_be_stated_explicitly():
    """Inference would be a silent guess, so there is no default."""
    with pytest.raises(PriceConversionError):
        to_dollars(pd.Series([1, 2, 3]), "auto")


def test_undef_price_becomes_missing_before_arithmetic():
    """INT64_MAX / 1e9 is a plausible-looking 9.2e9 — mask it first."""
    raw = pd.Series([fixed(100.01), UNDEF_PRICE])
    out = to_dollars(raw, "fixed")
    assert out.iloc[0] == pytest.approx(100.01)
    assert np.isnan(out.iloc[1])
    # and the integer copy is null, not a giant number
    assert pd.isna(to_fixed_int(raw).iloc[1])


def test_undef_price_cannot_poison_a_spread():
    bid = to_dollars(pd.Series([fixed(99.99)]), "fixed")
    ask = to_dollars(pd.Series([UNDEF_PRICE]), "fixed")
    assert np.isnan((ask - bid).iloc[0])


def test_converted_once_assertion_passes_on_a_correct_conversion():
    raw = pd.Series([fixed(100.01), fixed(99.99)])
    assert_prices_converted_once(to_dollars(raw, "fixed"), to_fixed_int(raw))


def test_double_conversion_is_caught_by_the_exactness_check():
    raw = pd.Series([fixed(100.01), fixed(99.99)])
    twice = to_dollars(raw, "fixed") / FIXED_PRICE_SCALE
    with pytest.raises(PriceConversionError):
        assert_prices_converted_once(twice, to_fixed_int(raw))


def test_unconverted_fixed_prices_are_caught_by_the_exactness_check():
    raw = pd.Series([fixed(100.01), fixed(99.99)])
    never = to_dollars(raw, "float")   # wrong mode: no division at all
    with pytest.raises(PriceConversionError):
        assert_prices_converted_once(never, to_fixed_int(raw))


def test_plausibility_check_catches_double_conversion():
    df = pd.DataFrame({"bid_price_1": [100.01 / FIXED_PRICE_SCALE] * 3})
    with pytest.raises(PriceConversionError, match="Implausible"):
        assert_plausible_prices(df, ["bid_price_1"], 0.01, 100_000.0)


def test_plausibility_check_catches_unconverted_fixed_point():
    df = pd.DataFrame({"bid_price_1": [float(fixed(100.01))] * 3})
    with pytest.raises(PriceConversionError, match="Implausible"):
        assert_plausible_prices(df, ["bid_price_1"], 0.01, 100_000.0)


def test_plausibility_check_accepts_real_quotes():
    df = pd.DataFrame({"bid_price_1": [99.99, 100.01, 250.50]})
    assert_plausible_prices(df, ["bid_price_1"], 0.01, 100_000.0)  # no raise


def test_integer_quote_comparison_is_exact():
    """Equality on the venue's own integers needs no float reasoning."""
    # 0.07 and 0.29 are not exactly representable in binary floating point
    a = pd.Series([fixed(100.07), fixed(100.29)])
    b = pd.Series([fixed(100.07), fixed(100.29)])
    ints_a, ints_b = to_fixed_int(a), to_fixed_int(b)

    assert (ints_a == ints_b).all()
    # an unchanged price compares equal, which is what the OFI indicator needs
    assert (ints_a.diff().dropna() != 0).all()   # the two levels differ
    assert int(ints_a.iloc[1] - ints_a.iloc[0]) == fixed(0.22)


def test_tick_size_is_inferred_not_assumed():
    penny = pd.Series([99.98, 99.99, 100.00, 100.01])
    assert infer_tick_size(penny) == pytest.approx(0.01)
    # a five-cent increment must not silently read as a penny
    nickel = pd.Series([99.90, 99.95, 100.00, 100.05])
    assert infer_tick_size(nickel) == pytest.approx(0.05)
    # too little variation -> fall back, never invent
    assert infer_tick_size(pd.Series([100.0]), fallback=0.01) == 0.01


# --------------------------------------------------------------------------- #
# End-to-end through the loader
# --------------------------------------------------------------------------- #
def test_prepare_raw_converts_once_and_retains_integers():
    cfg = databento_config()          # price_mode == 'fixed'
    recs = [record(ts_recv=SESSION_OPEN_NS + i * SEC, sequence=i,
                   bid=[(99.99, 100.0)], ask=[(100.01, 100.0)])
            for i in range(3)]
    out = prepare_raw(raw_frame(recs), cfg)

    assert out["bid_px_00"].iloc[0] == pytest.approx(99.99)
    assert int(out["bid_px_00_int"].iloc[0]) == fixed(99.99)
    # the assertion the loader itself runs must hold on its own output
    assert_prices_converted_once(out["bid_px_00"], out["bid_px_00_int"])


def test_prepare_raw_rejects_a_price_mode_mismatch():
    """Declaring fixed-point data as float leaves 1e11 'dollars' — it must fail."""
    cfg = databento_config()
    cfg.data.price_mode = "float"
    recs = [record(ts_recv=SESSION_OPEN_NS, sequence=1)]
    with pytest.raises(PriceConversionError):
        prepare_raw(raw_frame(recs), cfg)


def test_prepare_raw_keeps_undef_levels_missing():
    cfg = databento_config()
    rec = record(ts_recv=SESSION_OPEN_NS, sequence=1,
                 bid=[(99.99, 100.0)], ask=[(100.01, 100.0)])
    rec["ask_px_01"] = UNDEF_PRICE      # absent second level
    rec["ask_sz_01"] = 0
    out = prepare_raw(raw_frame([rec]), cfg)

    assert np.isnan(out["ask_px_01"].iloc[0])
    assert out["ask_px_00"].iloc[0] == pytest.approx(100.01)
