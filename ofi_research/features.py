"""Microstructure features (sections 5-10).

All features are strictly causal: the value at row ``t`` uses only rows with
timestamp <= t. Rolling aggregations are trailing (right-closed) and are reset
at segment boundaries (new instrument, new trading day, or a data gap larger
than ``gap_reset_ms``) so information never leaks across sessions.

Core object: :func:`build_features` takes a clean canonical dataframe and a
:class:`Config` and returns the dataframe with feature columns added, plus a
list of feature-family availability notes.

The OFI increment implements exactly the section-6 event definition. See
``tests/test_ofi.py`` for hand-computed expected values on synthetic books.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import Config

logger = logging.getLogger(__name__)


# --- Segmentation (reset boundaries) ---
def add_segments(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Add ``segment_id``: resets the OFI chain at DATA-INTEGRITY breaks only.

    A break means the book-state chain is discontinuous, so differencing across
    it is meaningless. That is about data integrity, not market activity (P0.3):
    a quiet second is a genuine state, not a feed failure, and never breaks the
    chain by default — elapsed time is kept as gap_ms / long_quiet_gap instead.

    Hard breaks (each configurable in :class:`IntegrityConfig`): instrument or
    session change, book clear, F_MAYBE_BAD_BOOK, F_SNAPSHOT, halt/auction
    reopen, invalid L1, the row after a non-benign removal (P0.4), and elapsed
    gaps only when ``use_gap_as_integrity_break`` is set.
    """
    icfg = config.integrity
    df = df.copy()

    def col_true(name: str) -> pd.Series:
        """Boolean column if present, else all-False."""
        if name in df.columns:
            return df[name].fillna(False).astype(bool)
        return pd.Series(False, index=df.index)

    breaks = pd.Series(False, index=df.index)

    if icfg.break_on_instrument_change:
        breaks |= (df["instrument"] != df["instrument"].shift())

    if icfg.break_on_session_change and config.data.drop_cross_day_features \
            and config.data.session_boundary == "calendar_day":
        breaks |= (df["session_date"] != df["session_date"].shift())

    if icfg.break_on_book_clear:
        # a clear invalidates differencing INTO the clearing row and out of it
        clear = col_true("book_clear")
        breaks |= clear | clear.shift(1, fill_value=False)

    if icfg.break_on_maybe_bad_book:
        bad = col_true("F_MAYBE_BAD_BOOK")
        breaks |= bad | bad.shift(1, fill_value=False)

    if icfg.break_on_snapshot:
        breaks |= col_true("F_SNAPSHOT")

    if icfg.break_on_halt:
        # A halt resumes via an auction/cross; the reopening book has no
        # differencing relationship to the last pre-halt book.
        halt = col_true("halt_boundary")
        breaks |= halt | halt.shift(1, fill_value=False)

    if icfg.break_on_invalid_book:
        for c in ("bid_price_1", "ask_price_1"):
            if c in df.columns:
                invalid = df[c].isna()
                breaks |= invalid | invalid.shift(1, fill_value=False)

    if icfg.break_after_nonbenign_removal:
        # set by validation.clean_data on the row FOLLOWING a removed row that
        # may have carried a real state transition
        breaks |= col_true("integrity_break")

    # elapsed time: recorded as a feature, break only on explicit opt-in
    gap_ms = (df.groupby("instrument")["timestamp"].diff()
              / pd.Timedelta(milliseconds=1))
    df["gap_ms"] = gap_ms
    df["long_quiet_gap"] = (gap_ms > icfg.long_quiet_gap_ms).fillna(False)
    if config.data.use_gap_as_integrity_break:
        breaks |= (gap_ms > config.data.gap_reset_ms).fillna(True)
        logger.warning("use_gap_as_integrity_break=True: elapsed gaps > %.0f ms "
                       "are being treated as feed failures.",
                       config.data.gap_reset_ms)

    breaks = breaks.fillna(False)
    breaks.iloc[0] = True
    df["segment_id"] = breaks.cumsum().astype("int64")
    df["is_segment_start"] = breaks.to_numpy()

    n_breaks = int(breaks.sum())
    n_quiet = int(df["long_quiet_gap"].sum())
    logger.info("Segments: %d integrity break(s); %d long quiet gap(s) retained "
                "as data (not breaks).", n_breaks, n_quiet)
    return df


# --- Basic variables (section 5) ---
def add_basic_variables(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    eps = config.data.epsilon
    df = df.copy()
    df["midprice"] = (df["ask_price_1"] + df["bid_price_1"]) / 2.0
    df["spread"] = df["ask_price_1"] - df["bid_price_1"]
    df["relative_spread"] = df["spread"] / (df["midprice"] + eps)
    df["log_midprice"] = np.log(df["midprice"].clip(lower=eps))
    df["total_L1_depth"] = df["bid_size_1"] + df["ask_size_1"]
    df["depth_imbalance_L1"] = (
        (df["bid_size_1"] - df["ask_size_1"])
        / (df["bid_size_1"] + df["ask_size_1"] + eps)
    )
    has_l2 = df["bid_price_2"].notna().any()
    if has_l2:
        df["total_L2_depth"] = (df["bid_size_1"] + df["ask_size_1"]
                                + df["bid_size_2"] + df["ask_size_2"])
    else:
        df["total_L2_depth"] = np.nan

    # delta_spread uses current and PAST only, per segment
    df["delta_spread"] = df.groupby("segment_id")["spread"].diff().fillna(0.0)
    return df


# --- OFI increments (section 6 / 7) ---
def _ofi_increment_arrays(bid_p: np.ndarray, bid_s: np.ndarray,
                          ask_p: np.ndarray, ask_s: np.ndarray) -> np.ndarray:
    """Vectorized section-6 OFI increment for one contiguous segment.

    increment[0] = 0 (no previous observation in the segment).
    """
    n = len(bid_p)
    out = np.zeros(n, dtype="float64")
    if n < 2:
        return out
    bp, bp0 = bid_p[1:], bid_p[:-1]
    bs, bs0 = bid_s[1:], bid_s[:-1]
    ap, ap0 = ask_p[1:], ask_p[:-1]
    as_, as0 = ask_s[1:], ask_s[:-1]

    bid_contrib = (bp >= bp0).astype(float) * bs - (bp <= bp0).astype(float) * bs0
    ask_contrib = -(ap <= ap0).astype(float) * as_ + (ap >= ap0).astype(float) * as0
    out[1:] = bid_contrib + ask_contrib
    return out


def compute_ofi_level(df: pd.DataFrame, level: int) -> pd.Series:
    """OFI increment series for a given book level (1 or 2), per segment."""
    bp = df[f"bid_price_{level}"].to_numpy(dtype="float64")
    bs = df[f"bid_size_{level}"].to_numpy(dtype="float64")
    ap = df[f"ask_price_{level}"].to_numpy(dtype="float64")
    as_ = df[f"ask_size_{level}"].to_numpy(dtype="float64")
    out = np.zeros(len(df), dtype="float64")
    for _, idx in df.groupby("segment_id").indices.items():
        idx = np.sort(idx)
        out[idx] = _ofi_increment_arrays(bp[idx], bs[idx], ap[idx], as_[idx])
    return pd.Series(out, index=df.index)


# --- Signed volume + trade classification (section 9) ---
def _inferred_trade_sign(df: pd.DataFrame, config: Config) -> pd.Series:
    """Causal quote / tick / Lee-Ready sign. Never references a future quote."""
    method = config.features.trade_classifier
    mid = df["midprice"]
    tp = df["trade_price"]
    quote_sign = np.sign(tp - mid)  # +1 above mid -> buy, -1 below -> sell
    # tick rule using PAST trade prices only, within segment
    prev_tp = df.groupby("segment_id")["trade_price"].ffill().shift()
    tick_sign = np.sign(tp - prev_tp)

    if method == "quote":
        sign = quote_sign
    elif method == "tick":
        sign = tick_sign
    else:  # lee_ready: quote test, tie-break with tick rule
        sign = quote_sign.where(quote_sign != 0, tick_sign)
    return pd.Series(sign, index=df.index).fillna(0.0)


def classify_trade_side(df: pd.DataFrame, config: Config
                        ) -> Tuple[pd.Series, Dict[str, float]]:
    """Return ``(signed_direction in {+1,-1,0}, report)``.

    Databento already carries the AGGRESSOR side on ``action == 'T'`` records
    ('B' buy aggressor, 'A'/'S' sell aggressor, 'N' unspecified), so inference
    is a fallback for the ``N`` remainder ONLY — never a replacement for the
    vendor's own field (P1.D). Classifying trades the vendor already resolved
    would substitute a noisy heuristic for ground truth and quietly change the
    signed-volume feature.

    ``report`` records how each trade was resolved so the reader can see how
    much of the signed-volume feature rests on inference:
    ``pct_from_vendor``, ``pct_from_inference``, ``pct_unresolved``, and the
    ``inference_method`` used.
    """
    has_trade = df["trade_size"].notna() & (df["trade_size"] > 0)
    n_trades = float(has_trade.sum())
    direction = pd.Series(0.0, index=df.index)
    method = config.features.trade_classifier

    vendor_resolved = pd.Series(False, index=df.index)
    if config.columns.has_trade_side() and df["trade_side"].notna().any():
        side = df["trade_side"]
        buy = side.isin(["B", "b", "buy", "BUY", 1, "1"])
        # 'A' is Databento's sell-aggressor code (the resting ASK was lifted);
        # 'S' is the generic spelling other feeds use.
        sell = side.isin(["A", "a", "S", "s", "sell", "SELL", -1, "-1"])
        direction[has_trade & buy] = 1.0
        direction[has_trade & sell] = -1.0
        vendor_resolved = has_trade & (buy | sell)

    # inference for the unspecified-side remainder only
    need_inference = has_trade & ~vendor_resolved
    n_inferred_ok = 0.0
    if need_inference.any():
        sign = _inferred_trade_sign(df, config)
        direction[need_inference] = sign[need_inference]
        n_inferred_ok = float((need_inference & (direction != 0)).sum())

    unresolved = has_trade & (direction == 0)
    report = {
        "n_trades": n_trades,
        "pct_from_vendor": 100.0 * float(vendor_resolved.sum()) / max(n_trades, 1),
        "pct_from_inference": 100.0 * n_inferred_ok / max(n_trades, 1),
        "pct_unresolved": 100.0 * float(unresolved.sum()) / max(n_trades, 1),
        "inference_method": method,
    }
    logger.info(
        "Trade side: %.2f%% from vendor aggressor field, %.2f%% inferred via "
        "'%s', %.2f%% unresolved (signed volume 0).",
        report["pct_from_vendor"], report["pct_from_inference"], method,
        report["pct_unresolved"])
    return direction, report


# --- Trailing aggregations ---
def _rolling_event_sum(df: pd.DataFrame, values: pd.Series, window: int
                       ) -> pd.Series:
    return (values.groupby(df["segment_id"])
            .rolling(window, min_periods=1).sum()
            .reset_index(level=0, drop=True))


def _rolling_time_sum(df: pd.DataFrame, values: pd.Series, window_ms: float
                      ) -> pd.Series:
    """Trailing time-window sum, computed per segment on a DatetimeIndex."""
    out = pd.Series(np.nan, index=df.index)
    tmp = pd.DataFrame({"t": df["timestamp"], "v": values,
                        "seg": df["segment_id"], "orig": df.index})
    win = f"{int(window_ms)}ms"
    for _, g in tmp.groupby("seg"):
        s = g.set_index("t")["v"]
        r = s.rolling(win, min_periods=1).sum()
        out.loc[g["orig"].values] = r.values
    return out


def _rolling_time_count(df: pd.DataFrame, indicator: pd.Series,
                        window_ms: float) -> pd.Series:
    return _rolling_time_sum(df, indicator.astype(float), window_ms)


def _rolling_event_mean(df: pd.DataFrame, values: pd.Series, window: int
                        ) -> pd.Series:
    return (values.groupby(df["segment_id"])
            .rolling(window, min_periods=1).mean()
            .reset_index(level=0, drop=True))


def _event_window_elapsed_ms(df: pd.DataFrame, window: int) -> pd.Series:
    """Clock time spanned by a trailing ``window``-event aggregate (P0.3).

    An event-count window is not a fixed amount of time: 50 events can span a
    millisecond in a burst or a minute in a lull. Reporting the span makes that
    explicit instead of leaving it implicit in the feature.
    """
    t_ns = df["timestamp"].astype("int64")
    first = (t_ns.groupby(df["segment_id"])
             .rolling(window, min_periods=1).min()
             .reset_index(level=0, drop=True))
    return (t_ns - first) / 1e6


# --- Event intensities (section 10) ---
_EVENT_MAP = {
    "market_buy": "market_buy", "market_sell": "market_sell",
    "limit_bid": "limit_bid", "limit_ask": "limit_ask",
    "cancel_bid": "cancel_bid", "cancel_ask": "cancel_ask",
}


def add_event_intensities(df: pd.DataFrame, config: Config,
                          window_ms: float) -> pd.DataFrame:
    df = df.copy()
    eps = config.data.epsilon
    if not (config.columns.has_event_type() and df["event_type"].notna().any()):
        return df
    et = df["event_type"].astype(str)
    w_sec = window_ms / 1000.0
    lam = {}
    for canon in _EVENT_MAP.values():
        ind = (et == canon)
        lam[canon] = _rolling_time_count(df, ind, window_ms) / w_sec
        df[f"lambda_{canon}"] = lam[canon]
    if "market_buy" in lam and "market_sell" in lam:
        mb, ms = lam["market_buy"], lam["market_sell"]
        # NAMING (P1.D): for XNAS.ITCH these are counts of aggressive EXECUTION
        # records, not of unique parent market orders. One parent order that
        # sweeps several resting orders produces several trade records, and the
        # public normalized feed does not identify the parent. Calling this
        # "market order intensity" would overstate what is observable.
        df["aggressive_trade_intensity_imbalance"] = mb - ms
        df["normalized_aggressive_trade_intensity_imbalance"] = (
            (mb - ms) / (mb + ms + eps))
        # Backwards-compatible aliases (deprecated; same values).
        df["market_order_intensity_imbalance"] = \
            df["aggressive_trade_intensity_imbalance"]
        df["normalized_intensity_imbalance"] = \
            df["normalized_aggressive_trade_intensity_imbalance"]
    return df


# --- Normalized OFI variants (section 8) ---
def _trailing_zscore(df: pd.DataFrame, values: pd.Series, window: int,
                     eps: float, exclude_current: bool = False) -> pd.Series:
    """Trailing z-score.

    ``exclude_current=False`` includes the current observation in its own
    mean/std. That is still causal — it uses no future information — but it
    shrinks the score of a genuine outlier, because the outlier inflates its
    own denominator. ``exclude_current=True`` normalizes by the strictly
    historical window, which is easier to interpret. Both are provided so the
    choice can be made on validation data rather than by assumption (P1.C).
    """
    g = values.groupby(df["segment_id"])
    mean = (g.rolling(window, min_periods=5).mean()
            .reset_index(level=0, drop=True))
    std = (g.rolling(window, min_periods=5).std()
           .reset_index(level=0, drop=True))
    if exclude_current:
        seg = df["segment_id"]
        mean = mean.groupby(seg).shift(1)
        std = std.groupby(seg).shift(1)
    return (values - mean) / (std + eps)


def build_features(df: pd.DataFrame, config: Config
                   ) -> Tuple[pd.DataFrame, List[str]]:
    """Construct all features; return (df_with_features, availability_notes)."""
    eps = config.data.epsilon
    notes: List[str] = []
    df = add_segments(df, config)
    df = add_basic_variables(df, config)

    has_l2 = df["bid_price_2"].notna().any()
    if not has_l2:
        notes.append("Level-2 columns absent -> L2 OFI disabled.")

    # OFI increments per level
    df["OFI_level_1_increment"] = compute_ofi_level(df, 1)
    if has_l2:
        df["OFI_level_2_increment"] = compute_ofi_level(df, 2)
        w1 = config.features.l2_weights["w1"]
        w2 = config.features.l2_weights["w2"]
        df["OFI_1_to_2_equal_increment"] = (
            df["OFI_level_1_increment"] + df["OFI_level_2_increment"])
        df["OFI_1_to_2_weighted_increment"] = (
            w1 * df["OFI_level_1_increment"] + w2 * df["OFI_level_2_increment"])
    else:
        df["OFI_level_2_increment"] = 0.0
        df["OFI_1_to_2_equal_increment"] = df["OFI_level_1_increment"]
        df["OFI_1_to_2_weighted_increment"] = df["OFI_level_1_increment"]

    # signed volume increment (per canonical event)
    direction, side_report = classify_trade_side(df, config)
    df["trade_direction"] = direction
    resolution = (
        f"Aggressor direction came from the vendor side field for "
        f"{side_report['pct_from_vendor']:.2f}% of trades, from a causal "
        f"'{side_report['inference_method']}' classifier for "
        f"{side_report['pct_from_inference']:.2f}%, and was UNRESOLVED for "
        f"{side_report['pct_unresolved']:.2f}% (direction 0, contributing zero "
        f"signed volume — unresolved trades are neutral, never dropped and "
        f"never guessed)")
    if "signed_trade_volume" in df.columns:
        # Vendor-aggregated net signed volume for the event. Preferred: one
        # canonical event may contain BOTH buy and sell executions, in which
        # case direction * total_size would not reproduce the net (P1.D).
        df["signed_volume_increment"] = (
            df["signed_trade_volume"].astype("float64").fillna(0.0))
        notes.append(
            "Signed trade volume is the vendor's `signed_trade_volume`, a NET "
            "per-event quantity (one canonical event may contain both buy and "
            "sell executions, so size x direction would not reproduce it). It "
            "is therefore not derived from the direction percentages that "
            "follow, which describe the SEPARATE per-trade `trade_direction` "
            f"used by the intensity features: {resolution}.")
    else:
        df["signed_volume_increment"] = (
            direction * df["trade_size"].fillna(0.0))
        if config.columns.has_trades() and df["trade_size"].notna().any():
            notes.append(
                "Signed trade volume was CONSTRUCTED as trade size x resolved "
                f"aggressor direction (no vendor net field available). "
                f"{resolution}.")
        else:
            notes.append("Trade columns absent -> signed volume is all zero.")

    # aggregate OFI + signed volume over event and time windows (trailing sums)
    increment_cols = {
        "OFI1": "OFI_level_1_increment",
        "OFI2": "OFI_level_2_increment",
        "OFI12eq": "OFI_1_to_2_equal_increment",
        "OFI12w": "OFI_1_to_2_weighted_increment",
        "signedvol": "signed_volume_increment",
    }
    for tag, col in increment_cols.items():
        for w in config.features.event_windows:
            df[f"{tag}_ev{w}"] = _rolling_event_sum(df, df[col], w)
        for wm in config.features.time_windows_ms:
            df[f"{tag}_ms{int(wm)}"] = _rolling_time_sum(df, df[col], wm)

    # how much CLOCK time each event-count window actually spans (P0.3)
    max_age = config.integrity.max_event_window_age_ms
    for w in config.features.event_windows:
        span = _event_window_elapsed_ms(df, w)
        df[f"event_window_elapsed_ms_ev{w}"] = span
        if max_age is not None:
            df[f"event_window_stale_ev{w}"] = span > max_age
    if max_age is not None:
        ref_w = config.features.event_windows[len(config.features.event_windows) // 2]
        n_stale = int(df[f"event_window_stale_ev{ref_w}"].sum())
        if n_stale:
            notes.append(
                f"{n_stale} rows have a {ref_w}-event window spanning more than "
                f"{max_age:.0f} ms (marked stale, not dropped).")

    # normalized OFI variants on a representative window (event window = middle)
    ref_ev = config.features.event_windows[len(config.features.event_windows) // 2]
    df["OFI_L1_ref"] = df[f"OFI1_ev{ref_ev}"]
    df["OFI_1_to_2_ref"] = df[f"OFI12w_ev{ref_ev}"]
    floor = config.features.depth_denominator_floor

    def _safe(denom: pd.Series) -> pd.Series:
        """Guard against zero / near-zero depth denominators (P1.C)."""
        return denom.clip(lower=floor) + eps

    # (a) current-depth normalizer. Causal, but mechanically spikes right after
    #     the queue is depleted: a small denominator inflates the ratio even
    #     when the numerator is ordinary.
    df["nOFI_L1"] = df["OFI_L1_ref"] / _safe(
        df["bid_size_1"] + df["ask_size_1"])
    if has_l2:
        df["nOFI_L2"] = df["OFI_1_to_2_ref"] / _safe(
            df["bid_size_1"] + df["ask_size_1"]
            + df["bid_size_2"] + df["ask_size_2"])
    else:
        df["nOFI_L2"] = df["nOFI_L1"]

    # (b) trailing-average-depth normalizer over the SAME event window as the
    #     OFI sum, so numerator and denominator are aligned. Less sensitive to
    #     instantaneous depletion.
    trailing_depth = _rolling_event_mean(df, df["total_L1_depth"], ref_ev)
    df["trailing_depth_L1_ref"] = trailing_depth
    df["nOFI_L1_trailing_depth"] = df["OFI_L1_ref"] / _safe(trailing_depth)

    df["zOFI"] = _trailing_zscore(
        df, df["OFI_L1_ref"], config.features.trailing_stat_window, eps
    ).fillna(0.0)
    # strictly-historical normalizer variant (current value excluded)
    df["zOFI_shifted"] = _trailing_zscore(
        df, df["OFI_L1_ref"], config.features.trailing_stat_window, eps,
        exclude_current=True
    ).fillna(0.0)

    # event intensities on a representative time window (1s)
    ref_ms = 1000.0 if 1000.0 in config.features.time_windows_ms \
        else config.features.time_windows_ms[-1]
    df = add_event_intensities(df, config, ref_ms)
    if "aggressive_trade_intensity_imbalance" not in df.columns:
        for c in ("aggressive_trade_intensity_imbalance",
                  "normalized_aggressive_trade_intensity_imbalance",
                  "market_order_intensity_imbalance",
                  "normalized_intensity_imbalance"):
            df[c] = 0.0
        notes.append("event_type absent -> aggressive-trade intensity = 0.")

    # trailing forecast-volatility proxy (per segment, STRICTLY causal): std of
    # recent midprice increments — used later for Z-score normalization.
    # Fallbacks are causal: rolling std -> expanding std -> forward fill ->
    # constant tick size. No backward fill (that would peek into the future).
    dmid = df.groupby("segment_id")["midprice"].diff()
    roll_vol = (dmid.groupby(df["segment_id"])
                .rolling(config.features.trailing_stat_window, min_periods=10)
                .std().reset_index(level=0, drop=True))
    exp_vol = (dmid.groupby(df["segment_id"])
               .expanding(min_periods=2).std().reset_index(level=0, drop=True))
    vol = roll_vol.fillna(exp_vol)
    vol = vol.groupby(df["segment_id"]).ffill()
    df["trailing_mid_vol"] = vol.fillna(config.features.tick_size)

    for n in notes:
        logger.info("FEATURE NOTE: %s", n)
    return df, notes


def winsor_bounds(train_values: pd.Series, config: Config
                  ) -> Tuple[float, float]:
    """Compute winsorization bounds from TRAINING data only (section 8)."""
    lo = float(train_values.quantile(config.features.winsor_lower_q))
    hi = float(train_values.quantile(config.features.winsor_upper_q))
    return lo, hi


def apply_winsor(values: pd.Series, bounds: Tuple[float, float]) -> pd.Series:
    return values.clip(lower=bounds[0], upper=bounds[1])
