"""Exploratory statistics with overlap-aware inference (sections 13, 14, 22, 25).

Because targets at horizon H overlap across rows, ordinary iid standard errors
are invalid. Every reported slope / correlation significance here uses
Newey-West (HAC) standard errors with a lag chosen to cover the overlap.

Functions:
    feature_summary       -> moments + autocorrelation of a feature
    hac_correlation       -> Pearson/Spearman + HAC slope t-stat vs a target
    decile_response       -> train-fitted deciles applied out-of-sample
    monotonicity_score    -> Spearman(decile rank, mean future move)
    fdr_adjust            -> Benjamini-Hochberg FDR adjustment (section 25)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .config import Config

try:
    import statsmodels.api as sm
    _HAS_SM = True
except Exception:  # pragma: no cover
    _HAS_SM = False

logger = logging.getLogger(__name__)


def hac_lag_rule(n: int, min_lag: int = 1) -> int:
    """Newey-West automatic lag ~ floor(4*(n/100)^(2/9))."""
    if n <= 1:
        return min_lag
    return max(min_lag, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))


def feature_summary(x: pd.Series) -> Dict[str, float]:
    x = pd.Series(x).dropna()
    n = len(x)
    if n < 3:
        return {"count": n}
    ac1 = x.autocorr(lag=1) if n > 2 else np.nan
    return {
        "count": n,
        "mean": float(x.mean()),
        "std": float(x.std()),
        "skew": float(stats.skew(x)),
        "kurtosis": float(stats.kurtosis(x)),  # excess kurtosis
        "autocorr_lag1": float(ac1),
    }


@dataclass
class HACResult:
    feature: str
    target: str
    n: int
    pearson: float
    spearman: float
    slope: float
    slope_se_hac: float
    t_stat_hac: float
    p_value_hac: float
    hac_lag: int
    #: Share of directional agreement among observations that actually MOVED.
    #: Unchanged outcomes are excluded rather than counted as misses; see
    #: ``pct_target_unchanged`` for how much of the sample that removes.
    directional_accuracy: float
    pct_target_unchanged: float
    cond_mean_pos: float   # E[y | feature > 0]
    cond_mean_neg: float   # E[y | feature < 0]
    cond_mean_se_pos: float
    cond_mean_se_neg: float
    ci95_slope_low: float
    ci95_slope_high: float


def hac_correlation(feature: pd.Series, target: pd.Series, feature_name: str,
                    target_name: str, hac_lag: Optional[int] = None) -> HACResult:
    """Correlation + HAC-robust slope inference of ``target ~ feature``."""
    df = pd.DataFrame({"x": feature, "y": target}).replace(
        [np.inf, -np.inf], np.nan).dropna()
    n = len(df)
    if n < 10:
        return HACResult(feature_name, target_name, n, *([np.nan] * 15))
    x, y = df["x"].to_numpy(), df["y"].to_numpy()
    pearson = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 else np.nan
    spearman = float(stats.spearmanr(x, y).correlation)
    lag = hac_lag if hac_lag is not None else hac_lag_rule(n)

    slope = se = t = p = ci_lo = ci_hi = np.nan
    if _HAS_SM and np.std(x) > 0:
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
        slope = float(model.params[1])
        se = float(model.bse[1])
        t = float(model.tvalues[1])
        p = float(model.pvalues[1])
        ci = model.conf_int(alpha=0.05)[1]
        ci_lo, ci_hi = float(ci[0]), float(ci[1])

    # Directional accuracy over MOVES only. On high-frequency data most short
    # -horizon mid changes are exactly zero; scoring those as misses (sign 0
    # never equals +-1) drove this below 50% on a day whose Pearson was
    # positive, which reads as an inverted signal when it is an artifact of the
    # unchanged outcomes. The unchanged share is reported alongside so the
    # denominator is never implicit.
    moved = (np.sign(y) != 0) & (np.sign(x) != 0)
    dir_acc = float(np.mean(np.sign(x[moved]) == np.sign(y[moved]))) \
        if moved.any() else np.nan
    pct_unchanged = 100.0 * float((np.sign(y) == 0).mean())
    pos_mask = x > 0
    neg_mask = x < 0
    yp, yn = y[pos_mask], y[neg_mask]
    cm_pos = float(yp.mean()) if yp.size else np.nan
    cm_neg = float(yn.mean()) if yn.size else np.nan
    se_pos = float(yp.std(ddof=1) / np.sqrt(yp.size)) if yp.size > 1 else np.nan
    se_neg = float(yn.std(ddof=1) / np.sqrt(yn.size)) if yn.size > 1 else np.nan

    return HACResult(
        feature=feature_name, target=target_name, n=n,
        pearson=pearson, spearman=spearman, slope=slope,
        slope_se_hac=se, t_stat_hac=t, p_value_hac=p, hac_lag=lag,
        directional_accuracy=dir_acc, pct_target_unchanged=pct_unchanged,
        cond_mean_pos=cm_pos, cond_mean_neg=cm_neg,
        cond_mean_se_pos=se_pos, cond_mean_se_neg=se_neg,
        ci95_slope_low=ci_lo, ci95_slope_high=ci_hi,
    )


@dataclass
class DecileResult:
    decile: int
    n: int
    mean_future: float
    median_future: float
    se_future: float
    prob_up: float
    avg_spread: float
    avg_vol: float
    avg_depth: float


def decile_response(train_feature: pd.Series, test_feature: pd.Series,
                    test_target: pd.Series, test_aux: pd.DataFrame,
                    n_bins: int = 10) -> Tuple[List[DecileResult], float, float]:
    """Fit decile edges on TRAIN, apply to TEST; per-decile out-of-sample stats.

    Returns (rows, monotonicity_spearman, top_minus_bottom_mean).
    ``test_aux`` should carry columns 'spread', 'vol', 'depth'.
    """
    tf = pd.Series(train_feature).replace([np.inf, -np.inf], np.nan).dropna()
    edges = np.unique(np.quantile(tf, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    df = pd.DataFrame({
        "x": test_feature.values, "y": test_target.values,
        "spread": test_aux["spread"].values,
        "vol": test_aux["vol"].values,
        "depth": test_aux["depth"].values,
    }).replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty or len(edges) < 3:
        return [], np.nan, np.nan
    df["bin"] = pd.cut(df["x"], bins=edges, labels=False, include_lowest=True)

    rows: List[DecileResult] = []
    for b, g in df.groupby("bin"):
        yv = g["y"].to_numpy()
        rows.append(DecileResult(
            decile=int(b), n=len(g),
            mean_future=float(yv.mean()),
            median_future=float(np.median(yv)),
            se_future=float(yv.std(ddof=1) / np.sqrt(len(yv))) if len(yv) > 1
            else np.nan,
            prob_up=float(np.mean(yv > 0)),
            avg_spread=float(g["spread"].mean()),
            avg_vol=float(g["vol"].mean()),
            avg_depth=float(g["depth"].mean()),
        ))
    rows.sort(key=lambda r: r.decile)
    means = [r.mean_future for r in rows]
    ranks = [r.decile for r in rows]
    mono = float(stats.spearmanr(ranks, means).correlation) if len(rows) > 2 \
        else np.nan
    top_bottom = means[-1] - means[0] if len(means) >= 2 else np.nan
    return rows, mono, top_bottom


def fdr_adjust(pvalues: List[float], alpha: float = 0.05) -> List[float]:
    """Benjamini-Hochberg FDR-adjusted p-values (section 25)."""
    p = np.asarray(pvalues, dtype=float)
    mask = ~np.isnan(p)
    out = np.full_like(p, np.nan)
    if mask.sum() == 0:
        return out.tolist()
    pv = p[mask]
    m = len(pv)
    order = np.argsort(pv)
    ranked = pv[order]
    adj = ranked * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    res = np.empty(m)
    res[order] = adj
    out[mask] = res
    return out.tolist()


# --- Overlap-aware uncertainty: day blocks (section 25) ---
def day_block_bootstrap(values: pd.Series, days: pd.Series,
                        n_boot: int = 1000, seed: int = 11,
                        statistic: str = "mean") -> Dict[str, float]:
    """Bootstrap a statistic by resampling whole TRADING DAYS with replacement.

    Rows whose target windows overlap are not independent draws, so an iid
    standard error on a per-row statistic is not merely imprecise — it is
    wrong in a known direction, understating the true sampling error, often by
    an order of magnitude at high frequency. Resampling whole sessions keeps
    every within-day dependence (overlap, autocorrelation, intraday
    seasonality) intact inside the block, and only assumes that separate
    trading days are approximately independent.

    Returns the point estimate, bootstrap SE, a percentile 95% interval, and
    the fraction of resamples on the same side of zero as the estimate.
    """
    d = pd.DataFrame({"v": pd.Series(values).to_numpy(),
                      "d": pd.Series(days).astype(str).to_numpy()})
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    out = {"n_obs": int(len(d)), "n_days": int(d["d"].nunique()),
           "estimate": np.nan, "se_day_bootstrap": np.nan,
           "ci95_low": np.nan, "ci95_high": np.nan,
           "frac_same_sign": np.nan, "statistic": statistic}
    if d.empty:
        return out

    def stat(frame: pd.DataFrame) -> float:
        v = frame["v"].to_numpy()
        if statistic == "sum":
            return float(np.sum(v))
        return float(np.mean(v))

    out["estimate"] = stat(d)
    day_names = d["d"].unique()
    n_days = len(day_names)
    if n_days < 2 or n_boot <= 0:
        # A single day carries no between-day variation; refusing to invent an
        # interval is more useful than printing a meaninglessly tight one.
        return out

    by_day = {k: g for k, g in d.groupby("d")}
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype="float64")
    for b in range(n_boot):
        pick = rng.integers(0, n_days, size=n_days)
        frame = pd.concat([by_day[day_names[j]] for j in pick],
                          ignore_index=True)
        draws[b] = stat(frame)

    est = out["estimate"]
    out["se_day_bootstrap"] = float(np.std(draws, ddof=1))
    out["ci95_low"] = float(np.percentile(draws, 2.5))
    out["ci95_high"] = float(np.percentile(draws, 97.5))
    out["frac_same_sign"] = float(np.mean(np.sign(draws) == np.sign(est))) \
        if est != 0 else np.nan
    return out


def day_clustered_mean(values: pd.Series, days: pd.Series) -> Dict[str, float]:
    """Mean with a day-clustered standard error.

    Each trading day contributes ONE effective observation: the SE is computed
    from the spread of per-day means, not from per-row scatter. It is the
    cheap companion to :func:`day_block_bootstrap` and it answers the question
    that matters for a walk-forward result — "would another month of days have
    produced this?" — rather than "would another millisecond have?".
    """
    d = pd.DataFrame({"v": pd.Series(values).to_numpy(),
                      "d": pd.Series(days).astype(str).to_numpy()})
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    if d.empty:
        return {"n_obs": 0, "n_days": 0, "mean": np.nan,
                "se_day_clustered": np.nan, "t_day_clustered": np.nan}
    per_day = d.groupby("d")["v"].mean()
    n_days = int(len(per_day))
    mean = float(per_day.mean())
    se = float(per_day.std(ddof=1) / np.sqrt(n_days)) if n_days > 1 else np.nan
    return {
        "n_obs": int(len(d)), "n_days": n_days, "mean": mean,
        "se_day_clustered": se,
        "t_day_clustered": mean / se if se and se > 0 else np.nan,
    }


# --- Tick regime (P1.E) ---
def tick_regime(df: pd.DataFrame, config: Config,
                mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Classify the symbol's tick regime from quoted spread, on TRAIN rows only.

    The regime must be fixed before performance is looked at, and large- and
    small-tick symbols must not be pooled into one statistic: in a large-tick
    name the spread is a binding constraint and most of the OFI signal is
    absorbed by queue dynamics at a single price, while in a small-tick name
    the midprice moves nearly continuously. They are different experiments.
    """
    tcfg = config.tick_regime
    tick = config.features.tick_size
    sub = df if mask is None else df[mask]
    spread = pd.to_numeric(sub.get("spread"), errors="coerce").dropna() \
        if "spread" in sub.columns else pd.Series(dtype="float64")
    out = {"n_obs": int(len(spread)), "tick_size": tick,
           "median_spread_ticks": np.nan, "pct_at_one_tick": np.nan,
           "regime": "unknown",
           "basis": "training rows only; preregister before evaluating P&L"}
    if spread.empty or tick <= 0:
        return out
    in_ticks = spread / tick
    med = float(in_ticks.median())
    out["median_spread_ticks"] = med
    if tcfg.report_pct_at_one_tick:
        out["pct_at_one_tick"] = 100.0 * float(
            np.isclose(in_ticks.to_numpy(), 1.0, atol=1e-6).mean())
    if med <= tcfg.large_tick_max_median_spread_ticks:
        out["regime"] = "large_tick"
    elif med >= tcfg.small_tick_min_median_spread_ticks:
        out["regime"] = "small_tick"
    else:
        out["regime"] = "intermediate"
    return out


def add_regime_labels(df: pd.DataFrame, train_mask: np.ndarray,
                      ofi_col: str = "OFI_L1_ref") -> pd.DataFrame:
    """Attach regime labels using boundaries estimated on TRAIN rows only.

    Regimes: spread (narrow/wide), volatility (low/high), depth (low/high),
    intensity (low/high), time-of-day (open/mid/close), OFI sign (pos/neg),
    OFI extremity (normal/extreme). Time-of-day thirds are computed per session.
    """
    df = df.copy()
    train = df[train_mask]

    def med(col):
        return float(train[col].median()) if col in train and \
            train[col].notna().any() else np.nan

    df["regime_spread"] = np.where(df["spread"] <= med("spread"),
                                   "narrow", "wide")
    df["regime_vol"] = np.where(
        df["trailing_mid_vol"] <= med("trailing_mid_vol"), "low_vol", "high_vol")
    df["regime_depth"] = np.where(
        df["total_L1_depth"] <= med("total_L1_depth"), "low_depth", "high_depth")
    if "market_order_intensity_imbalance" in df:
        thr = float(train["market_order_intensity_imbalance"].abs().median()) \
            if train["market_order_intensity_imbalance"].notna().any() else 0.0
        df["regime_intensity"] = np.where(
            df["market_order_intensity_imbalance"].abs() <= thr,
            "low_intensity", "high_intensity")
    # time of day: thirds of each session
    grp = df.groupby(["instrument", "session_date"])["timestamp"]
    start = grp.transform("min")
    end = grp.transform("max")
    frac = ((df["timestamp"] - start) /
            (end - start).replace(pd.Timedelta(0), pd.Timedelta(seconds=1)))
    df["regime_tod"] = np.where(frac < 1 / 3, "open",
                                np.where(frac < 2 / 3, "midday", "close"))
    if ofi_col in df:
        df["regime_ofi_sign"] = np.where(df[ofi_col] >= 0, "ofi_pos", "ofi_neg")
        hi = float(train[ofi_col].abs().quantile(0.9)) \
            if train[ofi_col].notna().any() else np.nan
        df["regime_ofi_extreme"] = np.where(df[ofi_col].abs() >= hi,
                                            "extreme", "normal")
    return df


def contemporaneous_cks(df: pd.DataFrame, config: Config,
                        window_ms: float,
                        ofi_increment_col: str = "OFI_level_1_increment"
                        ) -> Dict[str, float]:
    """Cont-Kukanov-Stoikov CONTEMPORANEOUS regression (P1.A).

    Regresses the midprice change over a bin on the OFI accumulated over the
    *same* bin::

        m_{t+D} - m_t  ~  a + b * OFI_(t, t+D]

    This is the original CKS specification, and it is a **data and formula
    sanity check only**. It is not a trading signal: the regressor is not known
    until the end of the interval it explains. A strong contemporaneous R^2 is
    the expected result on any correct order-book dataset and says nothing
    about out-of-sample predictability — do not quote it as expected alpha.

    Bins are non-overlapping and never span a segment boundary.
    """
    work = df[["timestamp", "segment_id", "midprice", ofi_increment_col]].copy()
    work = work.dropna(subset=["midprice"])
    if work.empty:
        return {"window_ms": window_ms, "n_bins": 0, "beta": np.nan,
                "r_squared": np.nan, "t_stat_hac": np.nan}

    t_ns = work["timestamp"].astype("int64").to_numpy()
    bin_ns = int(window_ms * 1_000_000)
    # bin index restarts within each segment
    seg = work["segment_id"].to_numpy()
    seg_start = pd.Series(t_ns).groupby(seg).transform("min").to_numpy()
    work["_bin"] = (seg.astype("int64") * (10 ** 12)
                    + (t_ns - seg_start) // bin_ns)

    g = work.groupby("_bin")
    agg = pd.DataFrame({
        "ofi": g[ofi_increment_col].sum(),
        "mid_first": g["midprice"].first(),
        "mid_last": g["midprice"].last(),
        "n": g.size(),
    })
    agg = agg[agg["n"] >= 2]
    if len(agg) < 10:
        return {"window_ms": window_ms, "n_bins": int(len(agg)),
                "beta": np.nan, "r_squared": np.nan, "t_stat_hac": np.nan}

    y = (agg["mid_last"] - agg["mid_first"]).to_numpy(dtype="float64")
    x = agg["ofi"].to_numpy(dtype="float64")
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 10 or np.std(x) == 0:
        return {"window_ms": window_ms, "n_bins": int(len(x)), "beta": np.nan,
                "r_squared": np.nan, "t_stat_hac": np.nan}

    X = np.column_stack([np.ones(len(x)), x])
    if _HAS_SM:
        res = sm.OLS(y, X).fit(cov_type="HAC",
                               cov_kwds={"maxlags": hac_lag_rule(len(x))})
        beta = float(res.params[1])
        r2 = float(res.rsquared)
        tstat = float(res.tvalues[1])
    else:  # pragma: no cover
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        beta = float(b[1])
        resid = y - X @ b
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else np.nan
        tstat = np.nan

    return {"window_ms": window_ms, "n_bins": int(len(x)), "beta": beta,
            "r_squared": r2, "t_stat_hac": tstat}


def contemporaneous_cks_table(df: pd.DataFrame, config: Config,
                              windows_ms: Optional[List[float]] = None
                              ) -> pd.DataFrame:
    """Run :func:`contemporaneous_cks` across several bin widths.

    Bins narrower than the decision clock are DROPPED, not reported. Sampling
    keeps roughly one row per ``decision_interval_ms``, so a finer bin is mostly
    empty and the few that survive need two observations to form a mid change:
    on INTC at a 200 ms clock the 100 ms row returned 62k bins where the sample
    implies ~4.7M, i.e. 1.3% of the data, silently. An aliased regression is
    worse than a missing one.
    """
    windows = windows_ms or config.features.time_windows_ms
    floor_ms = float(config.sampling.decision_interval_ms or 0.0)
    usable = [w for w in windows if float(w) >= floor_ms]
    if len(usable) < len(windows):
        logger.warning(
            "CKS: dropped %d bin width(s) below the %.0f ms decision clock "
            "(%s); they alias the sampled grid.",
            len(windows) - len(usable), floor_ms,
            ",".join(f"{w:.0f}" for w in windows if float(w) < floor_ms))
    if not usable:
        return pd.DataFrame()
    rows = [contemporaneous_cks(df, config, w) for w in usable]
    out = pd.DataFrame(rows)
    out["interpretation"] = (
        "CONTEMPORANEOUS sanity check only — not predictive, not tradeable")
    return out


def summarize_hac_table(results: List[HACResult]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(r) for r in results])
    if not df.empty and "p_value_hac" in df:
        df["p_value_fdr"] = fdr_adjust(df["p_value_hac"].tolist())
    return df
