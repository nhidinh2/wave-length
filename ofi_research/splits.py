"""Walk-forward splits by trading day (section 16).

Market data is never shuffled. Splits are contiguous blocks of whole trading
days: train / validation / test, advancing by ``step_days``. Both rolling
(fixed train window) and expanding (growing train window) schemes are supported.

Each fold yields boolean row masks so downstream code fits transforms and models
on train only, tunes on validation, and evaluates once on test.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from .config import Config

logger = logging.getLogger(__name__)


@dataclass
class Fold:
    index: int
    train_days: List[str]
    val_days: List[str]
    test_days: List[str]
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray


def make_folds(df: pd.DataFrame, config: Config) -> List[Fold]:
    scfg = config.splits
    days = sorted(pd.Series(df["session_date"].unique()).tolist())
    day_str = [str(d) for d in days]
    n = len(days)
    folds: List[Fold] = []

    def mask_for(day_subset: List[str]) -> np.ndarray:
        wanted = set(day_subset)
        return df["session_date"].astype(str).isin(wanted).to_numpy()

    i = 0
    fold_idx = 0
    while True:
        if scfg.scheme == "rolling":
            tr_start = i
            tr_end = tr_start + scfg.train_days
        else:  # expanding
            tr_start = 0
            tr_end = max(scfg.min_train_days, scfg.train_days) + i
        val_end = tr_end + scfg.validation_days
        test_end = val_end + scfg.test_days
        if test_end > n:
            break
        tr = day_str[tr_start:tr_end]
        va = day_str[tr_end:val_end]
        te = day_str[val_end:test_end]
        folds.append(Fold(
            index=fold_idx,
            train_days=tr, val_days=va, test_days=te,
            train_mask=mask_for(tr), val_mask=mask_for(va), test_mask=mask_for(te),
        ))
        fold_idx += 1
        i += scfg.step_days

    logger.info("Built %d %s walk-forward folds from %d days",
                len(folds), scfg.scheme, n)
    if not folds:
        logger.warning("No folds: need >= train+val+test = %d days, have %d",
                       scfg.train_days + scfg.validation_days + scfg.test_days, n)
    return folds
