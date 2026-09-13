"""Databento DBN constants, flag parsing, and price conversion (P0.6 / P0.7).

This module is deliberately tiny and vendor-specific. Nothing here knows about
OFI, targets, or backtests; the OFI math must never import vendor semantics.

**Verify against a one-day sample before trusting.** The bit values and enum
letters below follow Databento's published DBN conventions, but they are
constants in someone else's format: ``audit_flags`` exists so a real file can
confirm them. If a sample contradicts this module, fix the module and record
the evidence — do not work around it downstream.

References:
    https://databento.com/docs/standards-and-conventions/common-fields-enums-types
    https://databento.com/docs/schemas-and-data-formats/mbp-10
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# --- Record flags (bitfield) ---
F_LAST = 128            # last record of an event at a given ts_event
F_TOB = 64              # top-of-book record
F_SNAPSHOT = 32         # part of a book snapshot, not an incremental update
F_MBP = 16              # aggregated MBP update
F_BAD_TS_RECV = 8       # ts_recv is suspect
F_MAYBE_BAD_BOOK = 4    # book may be inconsistent / unrecoverable

FLAG_NAMES: Dict[str, int] = {
    "F_LAST": F_LAST,
    "F_TOB": F_TOB,
    "F_SNAPSHOT": F_SNAPSHOT,
    "F_MBP": F_MBP,
    "F_BAD_TS_RECV": F_BAD_TS_RECV,
    "F_MAYBE_BAD_BOOK": F_MAYBE_BAD_BOOK,
}

# --- Actions and sides ---
ACTION_ADD = "A"
ACTION_CANCEL = "C"
ACTION_MODIFY = "M"
ACTION_CLEAR = "R"       # clear book — a hard integrity break
ACTION_TRADE = "T"
ACTION_FILL = "F"
ACTION_NONE = "N"

# Databento encodes the AGGRESSOR side on trade records:
#   'B' -> buy aggressor, 'A' -> sell aggressor (resting ask lifted), 'N' -> none
SIDE_BID = "B"
SIDE_ASK = "A"
SIDE_NONE = "N"

#: int64 sentinel meaning "no price". MUST become NaN before any arithmetic.
UNDEF_PRICE = 9_223_372_036_854_775_807

#: Databento fixed-point price scale (1 unit = 1e-9 dollars).
FIXED_PRICE_SCALE = 1_000_000_000


# --- Flags ---
def flag_set(flags: pd.Series, bit: int) -> pd.Series:
    """Boolean mask: is ``bit`` set in the integer ``flags`` column?"""
    f = pd.to_numeric(flags, errors="coerce").fillna(0).astype("int64")
    return (f & int(bit)) != 0


def decode_flags(df: pd.DataFrame, flags_col: str = "flags") -> pd.DataFrame:
    """Add one boolean column per known flag. Missing column -> all False."""
    out = df.copy()
    if flags_col not in out.columns:
        for name in FLAG_NAMES:
            out[name] = False
        logger.warning("No %r column present; all DBN flags assumed unset. "
                       "Event boundaries will fall back to sequence changes.",
                       flags_col)
        return out
    for name, bit in FLAG_NAMES.items():
        out[name] = flag_set(out[flags_col], bit)
    return out


def audit_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Per-flag counts — item 5 of the one-day pilot audit."""
    rows = []
    n = max(len(df), 1)
    for name in FLAG_NAMES:
        c = int(df[name].sum()) if name in df.columns else 0
        rows.append({"flag": name, "n_set": c, "pct": 100.0 * c / n})
    return pd.DataFrame(rows)


# --- Prices (P0.6) ---
class PriceConversionError(ValueError):
    """Raised when a price column cannot be converted safely."""


def to_dollars(values: pd.Series, price_mode: str,
               scale: int = FIXED_PRICE_SCALE,
               sentinel: int = UNDEF_PRICE) -> pd.Series:
    """Convert a vendor price column to float dollars, exactly once.

    ``price_mode`` must be stated explicitly — it is never inferred, because a
    wrong guess is silent. ``'fixed'`` divides by ``scale``; ``'float'`` does
    not divide at all. The UNDEF sentinel becomes NaN *before* any arithmetic
    so it cannot poison a mid or a spread.
    """
    if price_mode not in ("fixed", "float"):
        raise PriceConversionError(
            f"price_mode must be 'fixed' or 'float', got {price_mode!r}")

    s = pd.to_numeric(values, errors="coerce")
    # sentinel first — dividing INT64_MAX by 1e9 yields a plausible-looking
    # 9.2e9 that would survive a naive range check.
    s = s.mask(s.abs() >= sentinel)

    if price_mode == "fixed":
        return s.astype("float64") / float(scale)
    return s.astype("float64")


#: Suffix for the untouched fixed-point integer copy of a price column.
INT_PRICE_SUFFIX = "_int"


def to_fixed_int(values: pd.Series, sentinel: int = UNDEF_PRICE
                 ) -> pd.Series:
    """Keep a vendor price column as nullable fixed-point integers.

    Dollar columns are for reporting, costing and P&L. Quote *comparisons* —
    "did the bid price change?", "is this one tick wider?" — are better done on
    the integers the venue actually published: they are exact by construction,
    so no reasoning about float representation is needed to trust an equality
    test. The UNDEF sentinel becomes null here too, so it can never be read as
    a real price.

    Returns a pandas nullable ``Int64`` series (``pd.NA`` for absent prices).
    """
    s = pd.to_numeric(values, errors="coerce")
    s = s.mask(s.abs() >= sentinel)
    return s.round().astype("Int64")


def assert_prices_converted_once(dollars: pd.Series, integers: pd.Series,
                                 scale: int = FIXED_PRICE_SCALE,
                                 context: str = "") -> None:
    """Assert ``dollars == integers / scale`` exactly once, elementwise.

    This is the direct check that P0.6 asks for: it fails loudly both when a
    fixed-point column was never divided and when it was divided twice, rather
    than relying on a range heuristic to notice.
    """
    d = pd.to_numeric(dollars, errors="coerce")
    i = pd.to_numeric(integers, errors="coerce")
    both = d.notna() & i.notna()
    if not both.any():
        return
    expected = i[both].astype("float64") / float(scale)
    actual = d[both].astype("float64")
    if not np.allclose(actual.to_numpy(), expected.to_numpy(),
                       rtol=1e-12, atol=1e-12):
        bad = int((~np.isclose(actual.to_numpy(), expected.to_numpy(),
                               rtol=1e-12, atol=1e-12)).sum())
        raise PriceConversionError(
            f"{bad} price(s){' in ' + context if context else ''} do not equal "
            f"their fixed-point integer / {scale:g} — the column was converted "
            "zero times or more than once.")


def assert_plausible_prices(df: pd.DataFrame, cols: List[str],
                            lo: float, hi: float,
                            context: str = "") -> None:
    """Range-check converted prices to catch double conversion or no conversion.

    Double-converted Databento prices land around 1e-7 dollars; unconverted
    fixed-point prices land around 1e11. Both are far outside any real quote,
    so a plain range assertion catches the error the moment it happens.
    """
    problems = []
    for c in cols:
        if c not in df.columns:
            continue
        v = pd.to_numeric(df[c], errors="coerce").dropna()
        v = v[v != 0]
        if v.empty:
            continue
        vmin, vmax = float(v.min()), float(v.max())
        if vmin < lo or vmax > hi:
            problems.append(
                f"{c}: observed [{vmin:.6g}, {vmax:.6g}] outside plausible "
                f"[{lo:g}, {hi:g}]")
    if problems:
        raise PriceConversionError(
            f"Implausible prices{' in ' + context if context else ''} — likely "
            "a price_mode mismatch (double conversion, or fixed-point left "
            "unconverted):\n  " + "\n  ".join(problems))


def infer_tick_size(prices: pd.Series, fallback: float = 0.01) -> float:
    """Infer the quote increment from observed distinct price differences.

    Uses the smallest positive absolute difference between consecutive distinct
    quotes. Returns ``fallback`` when there is too little variation to tell.
    Intended as a *check* on a configured tick size, not a silent override.
    """
    v = pd.to_numeric(prices, errors="coerce").dropna().to_numpy()
    if len(v) < 3:
        return fallback
    d = np.abs(np.diff(np.unique(np.round(v, 10))))
    d = d[d > 0]
    if len(d) == 0:
        return fallback
    return float(np.min(d))
