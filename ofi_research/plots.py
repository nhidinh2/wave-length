"""Plotting utilities (section 27).

Uses the non-interactive Agg backend so figures render without a display. Every
figure gets a title, labeled axes with units, a sample-size annotation where
relevant, and an explicit train/validation/test status note. When the data is
synthetic, callers pass ``synthetic=True`` and a SYNTHETIC banner is stamped on.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

logger = logging.getLogger(__name__)


def _finalize(fig, ax, title: str, xlabel: str, ylabel: str,
              note: Optional[str], synthetic: bool, path: Path) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    banner = note or ""
    if synthetic:
        banner = ("SYNTHETIC DATA — NOT A REAL RESULT  |  " + banner).strip(" |")
    if banner:
        fig.text(0.01, 0.01, banner, fontsize=7, color="firebrick")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    logger.info("Saved plot %s", path)


def plot_distribution(x: pd.Series, title: str, xlabel: str, path: str,
                      split: str = "all", synthetic: bool = False) -> None:
    x = pd.Series(x).replace([np.inf, -np.inf], np.nan).dropna()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(x, bins=60, color="steelblue", alpha=0.8)
    _finalize(fig, ax, title, xlabel, "count",
              f"n={len(x)} | split={split}", synthetic, Path(path))


def plot_decile_response(deciles: pd.DataFrame, path: str, horizon: str,
                         split: str = "test", synthetic: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(deciles["decile"], deciles["mean_future"],
                yerr=deciles["se_future"], marker="o", capsize=3,
                color="darkgreen")
    ax.axhline(0, color="black", lw=0.8)
    n = int(deciles["n"].sum())
    _finalize(fig, ax, f"OFI decile vs future mid change ({horizon})",
              "OFI decile (train-fitted edges)",
              "mean future mid change (price units)",
              f"n={n} | split={split}", synthetic, Path(path))


def plot_correlation_vs_horizon(corr_df: pd.DataFrame, path: str,
                                value_col: str = "pearson",
                                split: str = "test",
                                synthetic: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(corr_df)), corr_df[value_col], marker="o")
    ax.set_xticks(range(len(corr_df)))
    ax.set_xticklabels(corr_df["target"], rotation=45, ha="right", fontsize=7)
    ax.axhline(0, color="black", lw=0.8)
    _finalize(fig, ax, "OFI-future correlation vs horizon",
              "horizon", f"{value_col} correlation",
              f"split={split}", synthetic, Path(path))


def plot_beta_over_time(fold_df: pd.DataFrame, path: str, coef_col: str,
                        synthetic: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(fold_df["fold"], fold_df[coef_col], marker="o", color="purple")
    ax.axhline(0, color="black", lw=0.8)
    _finalize(fig, ax, f"OFI coefficient across walk-forward folds ({coef_col})",
              "fold index (time ordered)", "standardized coefficient",
              "each point = one out-of-sample fold", synthetic, Path(path))


def plot_cumulative_pnl(ledger: pd.DataFrame, path: str,
                        synthetic: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    if len(ledger):
        t = ledger["exit_time"]
        ax.plot(t, ledger["gross_mid_pnl"].cumsum(), label="gross (mid-to-mid)",
                color="gray", ls="--")
        ax.plot(t, ledger["net_pnl"].cumsum(), label="net (bid/ask+costs)",
                color="navy")
        ax.legend()
    _finalize(fig, ax, "Cumulative P&L (test)", "time",
              "cumulative P&L (price units)",
              f"n_trades={len(ledger)} | split=test", synthetic, Path(path))


def plot_drawdown(ledger: pd.DataFrame, path: str,
                  synthetic: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    if len(ledger):
        cum = ledger["net_pnl"].cumsum().to_numpy()
        dd = cum - np.maximum.accumulate(cum)
        ax.fill_between(range(len(dd)), dd, color="firebrick", alpha=0.5)
    _finalize(fig, ax, "Net P&L drawdown (test)", "trade index",
              "drawdown (price units)", "split=test", synthetic, Path(path))


def plot_regime_bar(regime_metrics: pd.DataFrame, path: str, metric: str,
                    title: str, synthetic: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(regime_metrics["regime"].astype(str), regime_metrics[metric],
           color="teal", alpha=0.8)
    ax.axhline(0, color="black", lw=0.8)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    _finalize(fig, ax, title, "regime", metric, "split=test", synthetic,
              Path(path))


def plot_sensitivity(x: List[float], y: List[float], path: str, title: str,
                     xlabel: str, ylabel: str = "net P&L (price units)",
                     synthetic: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, marker="o", color="darkorange")
    ax.axhline(0, color="black", lw=0.8)
    _finalize(fig, ax, title, xlabel, ylabel, "split=test", synthetic,
              Path(path))
