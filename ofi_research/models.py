"""Prediction models (section 15) with train-only fitting.

The linear baseline is the core. Feature standardization uses statistics from
TRAIN only; coefficients are reported with Newey-West (HAC) t-statistics because
the overlapping targets induce autocorrelated residuals. Optional regularized
(Ridge/Lasso/ElasticNet), robust (Huber) and directional (Logistic) models are
provided for comparison but the linear model is the reference.

Each fold: fit scaler+model on train -> pick hyperparameters on validation ->
freeze -> predict once on test.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import (
    Ridge, Lasso, ElasticNet, HuberRegressor, LogisticRegression,
)
from sklearn.preprocessing import StandardScaler

try:
    import statsmodels.api as sm
    _HAS_SM = True
except Exception:  # pragma: no cover
    _HAS_SM = False

from .config import Config
from .diagnostics import hac_lag_rule
from .features import wave_feature_names

logger = logging.getLogger(__name__)


def feature_groups(config: Config) -> Dict[str, List[str]]:
    """Named feature groups. Every model is a union of these (P1.B).

    Keeping the groups explicit is what makes the ablation attributable: each
    rung of the ladder adds exactly ONE group to a common base, and each
    leave-one-out removes exactly one group from the full model.
    """
    ev = config.features.event_windows
    ref = ev[len(ev) // 2]
    return {
        "ofi": [f"OFI1_ev{ref}"],
        "ofi_normalized": ["nOFI_L1"],
        "spread": ["delta_spread", "relative_spread"],
        "depth": ["depth_imbalance_L1"],
        "signed_volume": [f"signedvol_ev{ref}"],
        "intensity": ["aggressive_trade_intensity_imbalance"],
        "volatility": ["trailing_mid_vol"],
        # u_t: OFI through the Duhamel kernel basis; weights fitted per fold.
        "wave": wave_feature_names(config),
    }


def model_feature_sets(config: Config) -> Dict[str, List[str]]:
    """Feature lists for the model ladder, ablations, and L2 variants.

    The ladder holds raw OFI FIXED and adds one group at a time, so a change in
    performance is attributable to the group that was added. ``M1N`` swaps the
    OFI representation while adding nothing, which isolates the effect of
    normalization itself.
    """
    g = feature_groups(config)
    ev = config.features.event_windows
    ref = ev[len(ev) // 2]
    ofi = g["ofi"]
    signedvol = g["signed_volume"]

    sets: Dict[str, List[str]] = {
        # ---- ladder: common base, parallel single-group additions ----
        "M0_constant": [],
        "M1_ofi": list(ofi),
        "M1N_ofi_normalized": list(g["ofi_normalized"]),
        "M2_ofi_spread_depth": ofi + g["spread"] + g["depth"],
        "M3_ofi_signedvol": ofi + signedvol,
        "M4_ofi_intensity": ofi + g["intensity"],
        "M5_full": (ofi + g["spread"] + g["depth"] + signedvol
                    + g["intensity"] + g["volatility"]),
        # Stage 2: the reference model plus u_t, and nothing else, so any
        # change is attributable to the wave group alone.
        "M6_wave": (ofi + g["spread"] + g["depth"] + signedvol
                    + g["intensity"] + g["volatility"] + g["wave"]),

        # ---- L2 comparison (fixed weights are DESIGN CHOICES, not CKS) ----
        "L2_ofi1_only": list(ofi),
        "L2_ofi1_ofi2_vector": [f"OFI1_ev{ref}", f"OFI2_ev{ref}"],
        "L2_equal_scalar": [f"OFI12eq_ev{ref}"],
        "L2_fixed_half_scalar": [f"OFI12w_ev{ref}"],

        # ---- normalization variants (compared on VALIDATION only) ----
        "N_current_depth": ["nOFI_L1"],
        "N_trailing_depth": ["nOFI_L1_trailing_depth"],
        "N_z_inclusive": ["zOFI"],
        "N_z_shifted": ["zOFI_shifted"],
    }

    # ---- leave-one-group-out from the full model ----
    full_groups = ["ofi", "spread", "depth", "signed_volume", "intensity",
                   "volatility"]
    for drop in full_groups:
        cols: List[str] = []
        for name in full_groups:
            if name != drop:
                cols.extend(g[name])
        sets[f"LOGO_minus_{drop}"] = cols

    # ---- legacy names retained so older configs/reports keep resolving ----
    sets["M2_nofi"] = list(g["ofi_normalized"])
    sets["M3_nofi_signedvol"] = g["ofi_normalized"] + signedvol
    sets["M4_full"] = (g["ofi_normalized"] + signedvol + g["intensity"]
                       + g["spread"] + g["depth"])
    sets["A_signedvol"] = list(signedvol)
    sets["A_intensity"] = list(g["intensity"])
    sets["A_ofi_spread"] = ofi + g["spread"] + g["depth"]
    return sets


#: The attributable ladder, in the order it should be reported.
LADDER_ORDER = ["M0_constant", "M1_ofi", "M1N_ofi_normalized",
                "M2_ofi_spread_depth", "M3_ofi_signedvol",
                "M4_ofi_intensity", "M5_full", "M6_wave"]

#: Leave-one-group-out models, reported against ``M5_full``.
LOGO_ORDER = ["LOGO_minus_ofi", "LOGO_minus_spread", "LOGO_minus_depth",
              "LOGO_minus_signed_volume", "LOGO_minus_intensity",
              "LOGO_minus_volatility"]

#: L2 aggregation comparison.
L2_ORDER = ["L2_ofi1_only", "L2_ofi1_ofi2_vector", "L2_equal_scalar",
            "L2_fixed_half_scalar"]

#: Normalization comparison — decide on VALIDATION data, never on test.
NORMALIZATION_ORDER = ["N_current_depth", "N_trailing_depth", "N_z_inclusive",
                       "N_z_shifted"]


def available_models(config: Config, columns) -> Dict[str, List[str]]:
    """Subset of the ladder whose features all exist in ``columns``."""
    have = set(columns)
    out = {}
    for name, cols in model_feature_sets(config).items():
        if all(c in have for c in cols):
            out[name] = cols
        else:
            missing = [c for c in cols if c not in have]
            logger.info("Model %s unavailable (missing %s)", name, missing)
    return out


@dataclass
class FitResult:
    name: str
    features: List[str]
    coef: Dict[str, float]           # standardized-space coefficients
    intercept: float
    t_stats: Dict[str, float]        # HAC t-stats (linear model only)
    r2_in_sample: float
    sigma_resid: float               # training residual std (forecast noise)
    n_train: int


class LinearModel:
    """OLS with train-only standardization and HAC inference."""

    def __init__(self, name: str, features: List[str],
                 hac_lag: Optional[int] = None):
        self.name = name
        self.features = features
        self.hac_lag = hac_lag
        self.scaler: Optional[StandardScaler] = None
        self.params: Optional[np.ndarray] = None
        self.result: Optional[FitResult] = None

    def _design(self, df: pd.DataFrame, fit_scaler: bool) -> np.ndarray:
        if not self.features:
            return np.ones((len(df), 1))
        X = df[self.features].to_numpy(dtype="float64")
        if fit_scaler:
            self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        return np.column_stack([np.ones(len(df)), Xs])

    def fit(self, train: pd.DataFrame, target_col: str) -> FitResult:
        d = train[self.features + [target_col]].replace(
            [np.inf, -np.inf], np.nan).dropna() if self.features \
            else train[[target_col]].replace([np.inf, -np.inf], np.nan).dropna()
        y = d[target_col].to_numpy(dtype="float64")
        X = self._design(d, fit_scaler=True)
        n = len(d)
        lag = self.hac_lag if self.hac_lag is not None else hac_lag_rule(n)

        coef = {}
        tstats = {}
        if _HAS_SM:
            model = sm.OLS(y, X).fit(cov_type="HAC",
                                     cov_kwds={"maxlags": lag})
            self.params = model.params
            r2 = float(model.rsquared)
            names = ["const"] + self.features
            for i, nm in enumerate(names):
                if nm == "const":
                    continue
                coef[nm] = float(model.params[i])
                tstats[nm] = float(model.tvalues[i])
            intercept = float(model.params[0])
            resid = y - X @ self.params
        else:  # pragma: no cover
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            self.params = beta
            resid = y - X @ beta
            ss_res = float(np.sum(resid ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            intercept = float(beta[0])
            for i, nm in enumerate(self.features):
                coef[nm] = float(beta[i + 1])
                tstats[nm] = np.nan

        sigma = float(np.std(resid, ddof=1)) if n > 2 else float(np.std(resid))
        self.result = FitResult(
            name=self.name, features=self.features, coef=coef,
            intercept=intercept, t_stats=tstats, r2_in_sample=r2,
            sigma_resid=sigma, n_train=n)
        return self.result

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.params is None:
            raise RuntimeError("model not fit")
        if not self.features:
            return np.full(len(df), self.params[0])
        X = df[self.features].to_numpy(dtype="float64")
        # rows with NaN features -> NaN prediction (no trade)
        nan_rows = np.isnan(X).any(axis=1)
        Xs = np.where(nan_rows[:, None], 0.0, X)
        Xs = self.scaler.transform(Xs)
        design = np.column_stack([np.ones(len(df)), Xs])
        pred = design @ self.params
        pred[nan_rows] = np.nan
        return pred


_SKLEARN_BUILDERS = {
    "ridge": lambda a: Ridge(alpha=a),
    "lasso": lambda a: Lasso(alpha=a, max_iter=10000),
    "elasticnet": lambda a: ElasticNet(alpha=a, l1_ratio=0.5, max_iter=10000),
    "huber": lambda a: HuberRegressor(alpha=a, max_iter=2000),
}


class SklearnModel:
    """Regularized/robust regressor with train-only scaling; alpha via val."""

    def __init__(self, name: str, features: List[str], kind: str):
        self.name = name
        self.features = features
        self.kind = kind
        self.scaler: Optional[StandardScaler] = None
        self.model = None
        self.alpha: Optional[float] = None
        self.result: Optional[FitResult] = None

    def fit(self, train: pd.DataFrame, target_col: str, alpha: float) -> None:
        d = train[self.features + [target_col]].replace(
            [np.inf, -np.inf], np.nan).dropna()
        X = d[self.features].to_numpy(dtype="float64")
        y = d[target_col].to_numpy(dtype="float64")
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        self.model = _SKLEARN_BUILDERS[self.kind](alpha).fit(Xs, y)
        self.alpha = alpha
        resid = y - self.model.predict(Xs)
        sigma = float(np.std(resid, ddof=1)) if len(y) > 2 else float(np.std(resid))
        coef = {f: float(c) for f, c in zip(self.features, self.model.coef_)}
        self.result = FitResult(
            name=self.name, features=self.features, coef=coef,
            intercept=float(self.model.intercept_), t_stats={},
            r2_in_sample=float(self.model.score(Xs, y)),
            sigma_resid=sigma, n_train=len(y))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.features].to_numpy(dtype="float64")
        nan_rows = np.isnan(X).any(axis=1)
        Xf = np.where(nan_rows[:, None], 0.0, X)
        pred = self.model.predict(self.scaler.transform(Xf))
        pred[nan_rows] = np.nan
        return pred


class DirectionModel:
    """Logistic regression for up/down direction (section 15 optional)."""

    def __init__(self, features: List[str]):
        self.features = features
        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[LogisticRegression] = None

    def fit(self, train: pd.DataFrame, target_col: str) -> None:
        d = train[self.features + [target_col]].replace(
            [np.inf, -np.inf], np.nan).dropna()
        y = (d[target_col].to_numpy() > 0).astype(int)
        X = d[self.features].to_numpy(dtype="float64")
        self.scaler = StandardScaler().fit(X)
        if len(np.unique(y)) < 2:
            self.model = None
            return
        self.model = LogisticRegression(max_iter=1000).fit(
            self.scaler.transform(X), y)

    def predict_proba_up(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.full(len(df), 0.5)
        X = df[self.features].to_numpy(dtype="float64")
        nan_rows = np.isnan(X).any(axis=1)
        Xf = np.where(nan_rows[:, None], 0.0, X)
        p = self.model.predict_proba(self.scaler.transform(Xf))[:, 1]
        p[nan_rows] = np.nan
        return p
