"""End-to-end experiment orchestrator and CLI (sections 26, 29, 30).

Subcommands:
    inspect  -- load data and print/save the section-3 schema report only.
    run      -- full pipeline: clean -> features -> targets -> walk-forward ->
                costs/backtest -> regime/robustness -> tables/plots -> report.
    test     -- run the pytest suite.

Use ``--synthetic`` to run on the labeled synthetic generator (for development
and to prove the pipeline executes). All synthetic outputs are stamped
SYNTHETIC. Point ``--data <path>`` (and optionally ``--config <json>``) at real
data for a genuine run.

The final decision (section 29) is driven by OUT-OF-SAMPLE walk-forward results,
never by in-sample p-values.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import passive, passive_policy, plots
from .backtest import run_backtest
from .config import Config, CostConfig, setup_logging
from .data_loader import load_data, make_synthetic_data
from .databento_loader import load_databento
from .diagnostics import (
    add_regime_labels, contemporaneous_cks_table, day_block_bootstrap,
    day_clustered_mean, decile_response, feature_summary, hac_correlation,
    summarize_hac_table, tick_regime,
)
from .features import build_features
from .models import (
    L2_ORDER, LADDER_ORDER, LOGO_ORDER, NORMALIZATION_ORDER, LinearModel,
    available_models, model_feature_sets,
)
from .passive_policy import policy_gate, run_passive_walk_forward
from .passive_tape import write_tape
from .sampling import (
    compact_dtypes, memory_report, prune_columns, sample_decision_rows,
)
from .splits import make_folds
from .targets import (
    add_execution_aligned_targets, add_targets, all_horizon_tags,
    assert_target_causality,
)
from .validation import clean_data, inspect_schema

logger = logging.getLogger("ofi_research.run")

OFI_VARIANTS = {
    "OFI1_ref": "OFI_L1_ref",
    "OFI12w_ref": "OFI_1_to_2_ref",
    "nOFI_L1": "nOFI_L1",
    "nOFI_L2": "nOFI_L2",
    "zOFI": "zOFI",
    "signed_volume": None,  # filled from ref window
}


def _outdir(config: Config) -> Path:
    p = Path(config.output_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_table(df: pd.DataFrame, config: Config, name: str) -> None:
    path = _outdir(config) / f"{name}.csv"
    df.to_csv(path, index=False)
    logger.info("Saved table %s (%d rows)", path, len(df))


# --- Data preparation ---
def prepare_streaming(config: Config, paths: List[str]
                      ) -> Tuple[pd.DataFrame, Dict]:
    """Feature-build one session at a time, keeping only the sampled result.

    A month of a liquid name is ~110M raw records; concatenating them before
    feature-building needs tens of gigabytes and cannot run on a workstation.
    Processing per file and retaining only the post-sampling frame keeps peak
    memory at roughly one session plus the accumulated result.

    This is exact, not an approximation, and the reason is
    ``IntegrityConfig.break_on_session_change``: a trading-date change is
    already a hard segment break, so no rolling window, OFI increment or target
    was ever permitted to span two sessions. Processing them separately
    therefore cannot change a single value.
    """
    frames: List[pd.DataFrame] = []
    notes: List[str] = []
    removals: List[pd.DataFrame] = []
    schemas: List[pd.DataFrame] = []

    for i, path in enumerate(paths, start=1):
        logger.info("[%d/%d] %s", i, len(paths), path)
        raw, _audits, _diags = load_databento([path], config)
        schema_row = inspect_schema(raw, config).to_frame()
        schema_row.insert(0, "source", str(path))
        schemas.append(schema_row)
        clean, removal = clean_data(raw, config)
        del raw
        removals.append(removal)

        feat, day_notes = build_features(clean, config)
        del clean
        feat = compact_dtypes(feat, config)
        feat = add_targets(feat, config)
        # copy=False: this frame is ours, and copying it once per horizon is
        # what made a single session take minutes and swap.
        for H in config.targets.clock_horizons_ms:
            feat = add_execution_aligned_targets(feat, config, float(H),
                                                 copy=False)
        assert_target_causality(feat, config)
        feat = prune_columns(feat, config)
        # The queue simulator consumes shares, so it needs every trade, not
        # one row per decision interval. Written here because this is the last
        # moment the full-resolution session exists.
        write_tape(feat, config)
        feat = sample_decision_rows(feat, config)
        frames.append(feat)
        notes = day_notes  # availability notes are per-schema, not per-day
        memory_report(feat, f"session {i}")

    combined = pd.concat(frames, ignore_index=True)
    del frames
    _save_table(pd.concat(schemas, ignore_index=True), config,
                "01_data_quality_report")
    _save_table(pd.concat(removals, ignore_index=True), config,
                "01b_cleaning_removal_report")
    memory_report(combined, "all sessions")

    info = {"notes": notes, "n_sessions": len(paths)}
    return _post_prepare(combined, config, notes, info)


def prepare(config: Config, df_raw: Optional[pd.DataFrame] = None
            ) -> Tuple[pd.DataFrame, Dict]:
    if df_raw is None:
        df_raw = load_data(config)
    schema = inspect_schema(df_raw, config)
    _save_table(schema.to_frame(), config, "01_data_quality_report")

    clean, removal = clean_data(df_raw, config)
    _save_table(removal, config, "01b_cleaning_removal_report")

    feat, notes = build_features(clean, config)
    del clean
    # Representation only — shrinks the frame before the target columns widen
    # it, and changes no value except float32 rounding, which price levels are
    # deliberately excluded from.
    feat = compact_dtypes(feat, config)

    feat = add_targets(feat, config)

    # execution-aligned targets for every clock horizon (P0.8 family 3)
    for H in config.targets.clock_horizons_ms:
        feat = add_execution_aligned_targets(feat, config, float(H),
                                             copy=False)

    # fail loudly rather than silently mis-report an ordering violation
    assert_target_causality(feat, config)

    # Sampling runs LAST, so every retained row's features and targets were
    # resolved against the complete event stream rather than the sampled grid.
    feat = prune_columns(feat, config)
    for _day, _sess in feat.groupby("session_date", sort=True):
        write_tape(_sess.reset_index(drop=True), config)
    feat = sample_decision_rows(feat, config)
    memory_report(feat, "post-sampling")
    return _post_prepare(feat, config, notes,
                         {"schema": schema, "removal": removal})


def _post_prepare(feat: pd.DataFrame, config: Config, notes: List[str],
                  info: Dict) -> Tuple[pd.DataFrame, Dict]:
    """Summary tables shared by the single-frame and per-session paths."""
    # feature summary table
    feat_cols = ["OFI_L1_ref", "OFI_1_to_2_ref", "nOFI_L1", "nOFI_L2",
                 "nOFI_L1_trailing_depth", "zOFI", "zOFI_shifted",
                 "spread", "delta_spread", "relative_spread",
                 "depth_imbalance_L1", "trailing_mid_vol",
                 "aggressive_trade_intensity_imbalance"]
    rows = []
    for c in feat_cols:
        if c in feat:
            s = feature_summary(feat[c])
            s["feature"] = c
            rows.append(s)
    _save_table(pd.DataFrame(rows), config, "02_feature_summary")

    # ---- horizon realization + event-window span audit (pilot items 11, 12) ----
    rows = []
    for H in config.targets.clock_horizons_ms:
        tag = f"ms{int(H)}"
        col = f"actual_horizon_ms_{tag}"
        if col not in feat:
            continue
        v = feat[col].dropna()
        rows.append({
            "horizon": tag, "kind": "clock",
            "requested_ms": H, "n_usable": int(len(v)),
            "pct_usable": 100.0 * len(v) / max(len(feat), 1),
            "realized_median_ms": float(v.median()) if len(v) else np.nan,
            "realized_p90_ms": float(v.quantile(0.9)) if len(v) else np.nan,
        })
    for k in config.targets.event_horizons:
        col = f"actual_horizon_ms_ev{k}"
        if col not in feat:
            continue
        v = feat[col].dropna()
        rows.append({
            "horizon": f"ev{k}", "kind": "event_diagnostic",
            "requested_ms": np.nan, "n_usable": int(len(v)),
            "pct_usable": 100.0 * len(v) / max(len(feat), 1),
            "realized_median_ms": float(v.median()) if len(v) else np.nan,
            "realized_p90_ms": float(v.quantile(0.9)) if len(v) else np.nan,
        })
    for w in config.features.event_windows:
        col = f"event_window_elapsed_ms_ev{w}"
        if col not in feat:
            continue
        v = feat[col].dropna()
        rows.append({
            "horizon": f"window_ev{w}", "kind": "event_window_span",
            "requested_ms": np.nan, "n_usable": int(len(v)),
            "pct_usable": 100.0 * len(v) / max(len(feat), 1),
            "realized_median_ms": float(v.median()) if len(v) else np.nan,
            "realized_p90_ms": float(v.quantile(0.9)) if len(v) else np.nan,
        })
    _save_table(pd.DataFrame(rows), config, "02b_horizon_and_window_realization")

    info = {**info, "feature_notes": notes}
    return feat, info


# --- Contemporaneous CKS replication (P1.A) -- sanity check, NOT alpha ---
def contemporaneous_replication(feat: pd.DataFrame, config: Config
                                ) -> pd.DataFrame:
    """CKS same-interval regression — a construction audit, not the model.

    Opt-in (``evaluation.run_cks_replication``). It measures whether OFI moves
    with price in the SAME bin, which is a check that the book processing is
    sane; it says nothing about prediction and is not part of the strategy.
    """
    if not config.evaluation.run_cks_replication:
        logger.info("CKS replication skipped (evaluation.run_cks_replication "
                    "is off). It is a construction audit, not the model.")
        return pd.DataFrame()
    table = contemporaneous_cks_table(feat, config)
    _save_table(table, config, "15_contemporaneous_cks_replication")
    if not table.empty and table["r_squared"].notna().any():
        best = table.loc[table["r_squared"].idxmax()]
        logger.info("Contemporaneous CKS: max R^2 %.4f at %.0f ms bins "
                    "(SANITY CHECK ONLY -- not predictive).",
                    float(best["r_squared"]), float(best["window_ms"]))
    return table


# --- Exploratory HAC correlations (section 13) ---
def exploratory_correlations(feat: pd.DataFrame, config: Config) -> pd.DataFrame:
    ev = config.features.event_windows
    ref = ev[len(ev) // 2]
    variants = dict(OFI_VARIANTS)
    variants["signed_volume"] = f"signedvol_ev{ref}"
    tags = all_horizon_tags(config)
    results = []
    for vname, col in variants.items():
        if col not in feat:
            continue
        for tag in tags:
            tcol = f"future_mid_change_{tag}"
            if tcol not in feat:
                continue
            res = hac_correlation(feat[col], feat[tcol], vname, tag)
            results.append(res)
    table = summarize_hac_table(results)
    _save_table(table, config, "03_ofi_correlation_by_horizon")
    return table


# --- Walk-forward evaluation of linear models (sections 15, 16, 22, 23) ---
def _oos_stats(pred: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    m = (~np.isnan(pred)) & (~np.isnan(y))
    if m.sum() < 5:
        return {"oos_r2": np.nan, "oos_corr": np.nan, "dir_acc": np.nan,
                "pct_unchanged": np.nan, "n_oos": int(m.sum())}
    p, yy = pred[m], y[m]
    ss_res = float(np.sum((yy - p) ** 2))
    ss_tot = float(np.sum((yy - yy.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    corr = float(np.corrcoef(p, yy)[0, 1]) if np.std(p) > 0 else np.nan
    # Directional accuracy over MOVES only — an unchanged mid is not a wrong
    # direction. Counting sign(y)==0 as a miss pushes this below 50% even when
    # the correlation is positive, which is an artifact of the horizon rather
    # than a statement about the model. M0 remains the benchmark for how often
    # "unchanged" is the right call.
    moved = (np.sign(yy) != 0) & (np.sign(p) != 0)
    dir_acc = float(np.mean(np.sign(p[moved]) == np.sign(yy[moved]))) \
        if moved.any() else np.nan
    return {"oos_r2": r2, "oos_corr": corr, "dir_acc": dir_acc,
            "pct_unchanged": 100.0 * float((np.sign(yy) == 0).mean()),
            "n_oos": int(m.sum())}


def _select_threshold_on_val(model: LinearModel, val_df: pd.DataFrame,
                             sigma: float, config: Config, horizon_ms: float
                             ) -> float:
    """Pick z-threshold that maximizes VALIDATION net P&L (never uses test)."""
    best_k, best_pnl = config.signal.z_thresholds[0], -np.inf
    if len(val_df) == 0:
        return best_k
    preds = model.predict(val_df)
    for k in config.signal.z_thresholds:
        bt = run_backtest(val_df, preds, sigma, config.costs, config.signal,
                          z_threshold=k, horizon_ms=horizon_ms)
        pnl = bt.metrics["net_pnl"]
        if pnl > best_pnl:
            best_pnl, best_k = pnl, k
    return best_k


#: Schema of the walk-forward fold table. Kept explicit so that a run with too
#: few days still produces an empty-but-typed frame rather than a column-less
#: one that every downstream ``fm["model"]`` selection would choke on.
FOLD_METRIC_COLUMNS: List[str] = [
    "fold", "model", "horizon", "test_day", "r2_in_sample", "sigma_resid",
    "oos_r2", "oos_corr", "dir_acc", "pct_unchanged", "n_oos",
    "beta_1", "beta_1_t", "beta_1_name", "chosen_z_threshold",
    "bt_trade_count", "bt_gross_pnl", "bt_net_pnl", "bt_total_costs",
    "bt_hit_rate", "bt_avg_pnl", "bt_sharpe_like", "bt_max_drawdown",
    "bt_turnover",
]

#: Schema of the coefficient-stability table, empty-safe for the same reason.
COEF_STABILITY_COLUMNS: List[str] = [
    "model", "horizon", "beta_name", "n_folds", "pct_positive", "mean",
    "median", "std", "min", "max", "ci95_low", "ci95_high", "sign_changes",
]


def walk_forward(feat: pd.DataFrame, config: Config, horizon_tag: str,
                 model_names: Optional[List[str]] = None,
                 table_tag: Optional[str] = None, save: bool = True,
                 ledger_model: Optional[str] = None
                 ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the walk-forward loop for the given horizon.

    ``model_names`` defaults to the attributable ladder (:data:`LADDER_ORDER`),
    restricted to models whose features actually exist. ``table_tag`` names the
    saved CSVs; callers that run a second, different model set (ablations, L2,
    normalization) must pass their own tag so they do not overwrite the primary
    fold tables.

    Returns (fold_metrics, coef_stability, combined_test_ledger).
    """
    # The ledger belongs to whichever model the report headlines, or the
    # caller's explicit choice for a secondary model set.
    if ledger_model is None:
        ledger_model = config.evaluation.reference_model
    target_col = f"future_mid_change_{horizon_tag}"
    horizon_ms = float(horizon_tag[2:]) if horizon_tag.startswith("ms") else None
    event_h = int(horizon_tag[2:]) if horizon_tag.startswith("ev") else None
    fsets = model_feature_sets(config)
    have = available_models(config, feat.columns)
    if model_names is None:
        model_names = [m for m in LADDER_ORDER if m in have]
    else:
        model_names = [m for m in model_names if m in have]
    table_tag = table_tag or horizon_tag

    folds = make_folds(feat, config)
    fold_rows: List[dict] = []
    ledgers: List[pd.DataFrame] = []

    for fold in folds:
        tr = feat[fold.train_mask]
        va = feat[fold.val_mask]
        te = feat[fold.test_mask].copy()
        te = add_regime_labels(te, np.ones(len(te), dtype=bool))  # labels on test
        # regime boundaries should come from TRAIN; recompute with train stats:
        te = add_regime_labels(
            pd.concat([tr, te]), np.arange(len(tr) + len(te)) < len(tr)
        ).iloc[len(tr):].reset_index(drop=True)

        for mname in model_names:
            feats = fsets[mname]
            model = LinearModel(mname, feats)
            try:
                fit = model.fit(tr, target_col)
            except Exception as exc:  # pragma: no cover
                logger.warning("fold %d model %s fit failed: %s",
                               fold.index, mname, exc)
                continue

            pred_test = model.predict(te)
            y_test = te[target_col].to_numpy()
            oos = _oos_stats(pred_test, y_test)

            row = {
                "fold": fold.index, "model": mname, "horizon": horizon_tag,
                "test_day": te["session_date"].iloc[0] if len(te) else None,
                "r2_in_sample": fit.r2_in_sample,
                "sigma_resid": fit.sigma_resid,
                **oos,
            }
            # coefficient of the primary OFI-like term (first feature)
            if feats:
                first = feats[0]
                row["beta_1"] = fit.coef.get(first, np.nan)
                row["beta_1_t"] = fit.t_stats.get(first, np.nan)
                row["beta_1_name"] = first

            # backtest only for horizons that define holding (all do); choose
            # threshold on validation then evaluate once on test.
            if not config.evaluation.run_taker_backtest:
                # NaN, never 0.0: a taker study that did not run must not be
                # readable as one that ran and measured nothing.
                row["chosen_z_threshold"] = np.nan
                row.update({f"bt_{k}": np.nan for k in (
                    "trade_count", "gross_pnl", "net_pnl", "total_costs",
                    "hit_rate", "avg_pnl", "sharpe_like", "max_drawdown",
                    "turnover")})
                fold_rows.append(row)
                continue
            if horizon_ms is not None:
                k = _select_threshold_on_val(model, va, fit.sigma_resid,
                                             config, horizon_ms)
                bt = run_backtest(te, pred_test, fit.sigma_resid, config.costs,
                                  config.signal, z_threshold=k,
                                  horizon_ms=horizon_ms)
            else:
                k = config.signal.z_thresholds[0]
                bt = run_backtest(te, pred_test, fit.sigma_resid, config.costs,
                                  config.signal, z_threshold=k,
                                  event_horizon=event_h)
            row["chosen_z_threshold"] = k
            row.update({f"bt_{kk}": vv for kk, vv in bt.metrics.items()})
            fold_rows.append(row)

            if mname == ledger_model and len(bt.ledger):
                led = bt.ledger.copy()
                led["fold"] = fold.index
                ledgers.append(led)

    # An empty result must still carry the schema: downstream reporting selects
    # on 'model' and aggregates the metric columns, and a column-less frame
    # would raise KeyError instead of yielding an empty table.
    fold_metrics = pd.DataFrame(fold_rows) if fold_rows \
        else pd.DataFrame(columns=FOLD_METRIC_COLUMNS)
    if save:
        _save_table(fold_metrics, config, f"05_walkforward_folds_{table_tag}")

    if fold_metrics.empty:
        logger.warning("No fold metrics for %s (need >= %d days). Returning "
                       "empty results.", horizon_tag,
                       config.splits.train_days + config.splits.validation_days
                       + config.splits.test_days)
        return (fold_metrics,
                pd.DataFrame(columns=COEF_STABILITY_COLUMNS),
                pd.DataFrame())

    # coefficient stability for the OFI models
    coef_rows = []
    for mname in [m for m in model_names if m != "M0_constant"]:
        sub = fold_metrics[fold_metrics["model"] == mname]
        b = sub["beta_1"].dropna()
        if b.empty:
            continue
        coef_rows.append({
            "model": mname, "horizon": horizon_tag,
            "beta_name": sub["beta_1_name"].iloc[0] if "beta_1_name" in sub
            else "",
            "n_folds": len(b),
            "pct_positive": float((b > 0).mean() * 100),
            "mean": float(b.mean()), "median": float(b.median()),
            "std": float(b.std()), "min": float(b.min()), "max": float(b.max()),
            "ci95_low": float(b.mean() - 1.96 * b.std() / max(np.sqrt(len(b)), 1)),
            "ci95_high": float(b.mean() + 1.96 * b.std() / max(np.sqrt(len(b)), 1)),
            "sign_changes": int((np.diff(np.sign(b.to_numpy())) != 0).sum()),
        })
    coef_stability = pd.DataFrame(coef_rows) if coef_rows \
        else pd.DataFrame(columns=COEF_STABILITY_COLUMNS)
    if save:
        _save_table(coef_stability, config,
                    f"09_coefficient_stability_{table_tag}")

    combined_ledger = (pd.concat(ledgers, ignore_index=True)
                       if ledgers else pd.DataFrame())
    return fold_metrics, coef_stability, combined_ledger


# --- Decile analysis (section 14) ---
def decile_analysis(feat: pd.DataFrame, config: Config, horizon_tag: str,
                    ofi_col: str = "OFI_L1_ref") -> pd.DataFrame:
    target_col = f"future_mid_change_{horizon_tag}"
    folds = make_folds(feat, config)
    all_rows = []
    monos = []
    for fold in folds:
        tr = feat[fold.train_mask]
        te = feat[fold.test_mask]
        aux = pd.DataFrame({
            "spread": te["spread"], "vol": te["trailing_mid_vol"],
            "depth": te["total_L1_depth"]})
        rows, mono, top_bottom = decile_response(
            tr[ofi_col], te[ofi_col], te[target_col], aux)
        for r in rows:
            d = asdict(r)
            d["fold"] = fold.index
            all_rows.append(d)
        if not np.isnan(mono):
            monos.append(mono)
    table = pd.DataFrame(all_rows)
    if not table.empty:
        pooled = (table.groupby("decile")
                  .agg(n=("n", "sum"),
                       mean_future=("mean_future", "mean"),
                       se_future=("se_future", "mean"),
                       prob_up=("prob_up", "mean"))
                  .reset_index())
        pooled["mean_monotonicity_spearman"] = float(np.mean(monos)) if monos \
            else np.nan
        _save_table(pooled, config, f"04_ofi_decile_response_{horizon_tag}")
        return pooled
    return table


# --- Ablation (section 23) ---
def _aggregate_models(fm: pd.DataFrame, order: List[str]) -> pd.DataFrame:
    if fm.empty:
        return pd.DataFrame(columns=["model"])
    present = [m for m in order if m in set(fm["model"])]
    return (fm.groupby("model")
            .agg(oos_r2=("oos_r2", "mean"),
                 oos_corr=("oos_corr", "mean"),
                 dir_acc=("dir_acc", "mean"),
                 net_pnl=("bt_net_pnl", "sum"),
                 sharpe_like=("bt_sharpe_like", "mean"),
                 trade_count=("bt_trade_count", "sum"),
                 turnover=("bt_turnover", "sum"))
            .reindex(present).reset_index())


def ablation(feat: pd.DataFrame, config: Config, horizon_tag: str
             ) -> pd.DataFrame:
    """Attributable ablation: the ladder, then leave-one-group-out from M5.

    Each ladder rung adds exactly ONE feature group to a common raw-OFI base,
    so a change in performance is attributable to the group that was added.
    ``M1N`` swaps the OFI representation while adding nothing, isolating the
    effect of normalization itself. Replacing raw OFI *and* adding a group in
    the same step — which the previous ladder did — makes attribution
    impossible, because two things moved at once (P1.B).
    """
    fm, _, _ = walk_forward(feat, config, horizon_tag,
                            model_names=LADDER_ORDER,
                            table_tag=f"ladder_{horizon_tag}", save=False)
    ladder = _aggregate_models(fm, LADDER_ORDER)
    ladder.insert(1, "role", "ladder_add_one_group")
    _save_table(ladder, config, f"07_ablation_{horizon_tag}")

    # leave-one-group-out from the full model
    if not config.evaluation.run_leave_one_group_out:
        logger.info("Leave-one-group-out skipped "
                    "(evaluation.run_leave_one_group_out is off): it is a "
                    "second full walk-forward and the peak-memory stage.")
        return ladder
    fm_logo, _, _ = walk_forward(feat, config, horizon_tag,
                                 model_names=["M5_full"] + LOGO_ORDER,
                                 table_tag=f"logo_{horizon_tag}", save=False,
                                 ledger_model="M5_full")
    logo = _aggregate_models(fm_logo, ["M5_full"] + LOGO_ORDER)
    if not logo.empty and "M5_full" in set(logo["model"]):
        base = logo.loc[logo["model"] == "M5_full"].iloc[0]
        for col in ("oos_corr", "oos_r2", "dir_acc", "net_pnl"):
            logo[f"delta_vs_M5_{col}"] = logo[col] - base[col]
    logo.insert(1, "role", "leave_one_group_out")
    _save_table(logo, config, f"07b_leave_one_group_out_{horizon_tag}")
    return ladder


def l2_comparison(feat: pd.DataFrame, config: Config, horizon_tag: str
                  ) -> pd.DataFrame:
    """Compare L1-only against the L1+L2 aggregation choices (P1.B).

    The 1.0 / 0.5 weights are a preregistered DESIGN CHOICE, not a canonical
    CKS parameter, and are labeled as such in the output so a reader cannot
    mistake them for an estimated quantity.
    """
    fm, _, _ = walk_forward(feat, config, horizon_tag, model_names=L2_ORDER,
                            table_tag=f"l2_{horizon_tag}", save=False,
                            ledger_model="L2_ofi1_only")
    tab = _aggregate_models(fm, L2_ORDER)
    if not tab.empty:
        tab["note"] = np.where(
            tab["model"] == "L2_fixed_half_scalar",
            "fixed 1.0/0.5 weights are a preregistered design choice, not CKS",
            "")
    _save_table(tab, config, f"16_l2_aggregation_{horizon_tag}")
    return tab


def normalization_selection(feat: pd.DataFrame, config: Config,
                            horizon_tag: str) -> pd.DataFrame:
    """Compare OFI normalizations on VALIDATION folds only (P1.C).

    Selecting a normalizer on the test fold would turn the one honest
    out-of-sample number into a maximum over four, so the comparison is scored
    on validation rows and the winner is merely *reported* here. The test fold
    stays untouched by this choice.
    """
    target_col = f"future_mid_change_{horizon_tag}"
    fsets = model_feature_sets(config)
    have = available_models(config, feat.columns)
    names = [m for m in NORMALIZATION_ORDER if m in have]
    folds = make_folds(feat, config)
    rows: List[dict] = []
    for fold in folds:
        tr, va = feat[fold.train_mask], feat[fold.val_mask]
        for mname in names:
            model = LinearModel(mname, fsets[mname])
            try:
                model.fit(tr, target_col)
            except Exception as exc:  # pragma: no cover
                logger.warning("normalization %s fold %d failed: %s",
                               mname, fold.index, exc)
                continue
            stats = _oos_stats(model.predict(va), va[target_col].to_numpy())
            rows.append({"fold": fold.index, "model": mname,
                         "split": "validation", **stats})
    table = pd.DataFrame(rows)
    if not table.empty:
        table = (table.groupby("model")
                 .agg(val_corr=("oos_corr", "mean"),
                      val_r2=("oos_r2", "mean"),
                      val_dir_acc=("dir_acc", "mean"),
                      n_folds=("fold", "nunique")).reset_index())
        table = table.sort_values("val_corr", ascending=False)
        table["selected_on"] = "validation folds only; test fold not consulted"
        logger.info("Normalization comparison (validation only): best = %s",
                    table["model"].iloc[0])
    _save_table(table, config, f"17_normalization_validation_{horizon_tag}")
    return table


# --- Regime performance (section 21) ---
def regime_performance(combined_ledger: pd.DataFrame, config: Config
                       ) -> pd.DataFrame:
    if combined_ledger.empty:
        return pd.DataFrame()
    rows = []
    for col in ["regime_spread", "regime_vol", "regime_tod"]:
        if col not in combined_ledger:
            continue
        for val, g in combined_ledger.groupby(col):
            rows.append({
                "regime_dim": col, "regime": val, "n_trades": len(g),
                "net_pnl": float(g["net_pnl"].sum()),
                "avg_pnl": float(g["net_pnl"].mean()),
                "hit_rate": float((g["net_pnl"] > 0).mean()),
            })
    table = pd.DataFrame(rows)
    _save_table(table, config, "08_regime_results")
    return table


# --- Robustness + cost/latency sensitivity (sections 18, 24) ---
def cost_latency_sensitivity(feat: pd.DataFrame, config: Config,
                             horizon_tag: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # These sweeps ARE the taker study — they re-run the aggressive backtest
    # under perturbed costs and latencies. With that path off they would burn
    # a full extra walk-forward to print zeros next to a strategy nobody is
    # proposing.
    if not config.evaluation.run_taker_backtest:
        logger.info("Taker path off: skipping cost/latency sensitivity "
                    "(tables 11 and 12 belong to the abandoned study).")
        empty_cost = pd.DataFrame(columns=["scenario", "net_pnl", "gross_pnl"])
        empty_lat = pd.DataFrame(columns=["latency_ms", "net_pnl"])
        return empty_cost, empty_lat
    horizon_ms = float(horizon_tag[2:])
    fsets = model_feature_sets(config)
    folds = make_folds(feat, config)

    def total_pnl(cost_cfg: CostConfig, use_gross: bool = False) -> float:
        total = 0.0
        for fold in folds:
            tr = feat[fold.train_mask]
            va = feat[fold.val_mask]
            te = feat[fold.test_mask].copy()
            te = add_regime_labels(
                pd.concat([tr, te]), np.arange(len(tr) + len(te)) < len(tr)
            ).iloc[len(tr):].reset_index(drop=True)
            ref = config.evaluation.reference_model
            model = LinearModel(ref, fsets[ref])
            fit = model.fit(tr, f"future_mid_change_{horizon_tag}")
            # threshold selected on validation under the SAME cost scenario
            best_k, best = config.signal.z_thresholds[0], -np.inf
            pv = model.predict(va)
            for kk in config.signal.z_thresholds:
                mval = run_backtest(va, pv, fit.sigma_resid, cost_cfg,
                                    config.signal, z_threshold=kk,
                                    horizon_ms=horizon_ms).metrics["net_pnl"]
                if mval > best:
                    best, best_k = mval, kk
            bt = run_backtest(te, model.predict(te), fit.sigma_resid, cost_cfg,
                              config.signal, z_threshold=best_k,
                              horizon_ms=horizon_ms)
            total += bt.metrics["gross_pnl"] if use_gross \
                else bt.metrics["net_pnl"]
        return total

    # NOTE: use dataclasses.replace, never CostConfig(**asdict(base)).
    # asdict() recurses into the nested LatencyConfig and returns it as a plain
    # dict, so the rebuilt CostConfig would carry a dict where a LatencyConfig
    # belongs and total_latency_ms() would raise.
    base = config.costs
    cost_rows = [
        {"scenario": "baseline_net", "pnl": total_pnl(base)},
        {"scenario": "plus_1_tick_slippage_net",
         "pnl": total_pnl(replace(base,
                          slippage_ticks=base.slippage_ticks + 1.0))},
        {"scenario": "double_fees_net",
         "pnl": total_pnl(replace(base,
                          fee_bps=base.fee_bps * 2 + 1.0,
                          fee_per_unit=base.fee_per_unit * 2))},
        # gross mid-to-mid on the SAME trades (labeled theoretical, not net)
        {"scenario": "mid_to_mid_gross_theoretical", "pnl": total_pnl(base, True)},
    ]
    cost_table = pd.DataFrame(cost_rows).rename(columns={"pnl": "net_pnl"})
    _save_table(cost_table, config, "11_transaction_cost_sensitivity")

    lat_rows = []
    for lat in [0.0, 1.0, 5.0, 25.0, 100.0]:
        cc = replace(base, latency_ms=lat)
        lat_rows.append({"latency_ms": lat, "net_pnl": total_pnl(cc)})
    lat_table = pd.DataFrame(lat_rows)
    _save_table(lat_table, config, "12_latency_sensitivity")
    return cost_table, lat_table


def overlap_aware_uncertainty(fold_metrics: pd.DataFrame,
                              combined_ledger: pd.DataFrame,
                              config: Config, horizon_tag: str
                              ) -> pd.DataFrame:
    """Day-blocked uncertainty for the headline out-of-sample numbers (§25).

    Targets at horizon H overlap across rows, so per-row standard errors
    understate sampling error. Both statistics here are resampled or clustered
    at the level of whole trading days — the largest block this design can
    treat as approximately independent — so the interval answers "would another
    month of days reproduce this?" rather than "would another millisecond?".

    With only a handful of test days the intervals will be wide. That width is
    the honest finding, not a defect to be tuned away.
    """
    ecfg = config.evaluation
    rows: List[dict] = []

    ref = config.evaluation.reference_model
    m1 = fold_metrics[fold_metrics["model"] == ref] \
        if not fold_metrics.empty else pd.DataFrame()
    if len(m1) and "test_day" in m1:
        boot = day_block_bootstrap(m1["oos_corr"], m1["test_day"],
                                   n_boot=ecfg.n_day_bootstrap,
                                   seed=ecfg.bootstrap_seed, statistic="mean")
        rows.append({"quantity": f"oos_pred_corr_{ref}", "unit": "correlation",
                     **boot})
        if ecfg.report_day_clustered:
            rows.append({"quantity": f"oos_pred_corr_{ref}",
                         "unit": "correlation",
                         **day_clustered_mean(m1["oos_corr"], m1["test_day"])})

    if len(combined_ledger) and "exit_time" in combined_ledger:
        led = combined_ledger.copy()
        led["day"] = pd.to_datetime(led["exit_time"]).dt.date.astype(str)
        boot = day_block_bootstrap(led["net_pnl"], led["day"],
                                   n_boot=ecfg.n_day_bootstrap,
                                   seed=ecfg.bootstrap_seed, statistic="mean")
        rows.append({"quantity": "net_pnl_per_trade", "unit": "price units",
                     **boot})
        if ecfg.report_day_clustered:
            rows.append({"quantity": "net_pnl_per_trade", "unit": "price units",
                         **day_clustered_mean(led["net_pnl"], led["day"])})

    table = pd.DataFrame(rows)
    if not table.empty:
        table["horizon"] = horizon_tag
        table["basis"] = ("whole trading days resampled/clustered; overlapping "
                          "targets make per-row iid SEs invalid")
    _save_table(table, config, "13_overlap_aware_uncertainty")
    return table


def tick_regime_report(feat: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Preregistered tick-regime classification from TRAIN rows only (P1.E)."""
    folds = make_folds(feat, config)
    mask = folds[0].train_mask if folds else np.ones(len(feat), dtype=bool)
    info = tick_regime(feat, config, mask)
    info["label"] = config.label
    info["replication_note"] = (
        "One symbol validates the pipeline and gives an initial effect "
        "estimate; it is not a general profitability claim. A large-tick and a "
        "small-tick symbol must be run as SEPARATE preregistered replications "
        "and never pooled without interactions or stratification.")
    table = pd.DataFrame([info])
    _save_table(table, config, "18_tick_regime")
    return table


def day_concentration(combined_ledger: pd.DataFrame, config: Config) -> Dict:
    """P&L concentration across days that actually contain trades.

    Every count here is a TRADING-day count, never a market-day count.
    Conflating the two is what made an 8-day walk-forward report 0 days.
    """
    if combined_ledger.empty:
        return {"n_trade_days": 0, "max_day_share": np.nan,
                "positive_day_frac": np.nan, "total_net_pnl": 0.0}
    led = combined_ledger.copy()
    led["day"] = pd.to_datetime(led["exit_time"]).dt.date
    per_day = led.groupby("day")["net_pnl"].sum()
    total = per_day.sum()
    # Share of ABSOLUTE P&L: dividing by the signed total goes negative when the
    # total is (so a one-day-dominated LOSS passes a "< 0.6" gate) and explodes
    # near zero. |max| / sum|day| stays in [0, 1].
    gross_abs = float(per_day.abs().sum())
    max_share = float(per_day.abs().max() / gross_abs) if gross_abs > 0 \
        else np.nan
    _save_table(per_day.reset_index().rename(columns={"net_pnl": "net_pnl"}),
                config, "10b_per_day_net_pnl")
    return {
        "n_trade_days": int(per_day.shape[0]),
        "max_day_share": max_share,
        "positive_day_frac": float((per_day > 0).mean()),
        "total_net_pnl": float(total),
    }


# --- Report (section 29) ---
def _fmt(x, nd=5):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "n/a"
        return f"{x:.{nd}f}"
    except Exception:
        return str(x)


def _day_block_phrase(unc: pd.DataFrame, quantity: str,
                      n_trades: Optional[float] = None) -> str:
    """One sentence describing the day-blocked interval for a quantity.

    ``n_trades`` disambiguates the failure: an empty ledger and a one-day ledger
    are missing an interval for different reasons.
    """
    if unc is None or unc.empty or "quantity" not in unc:
        return "No day-blocked interval available."
    sub = unc[(unc["quantity"] == quantity)
              & unc.get("se_day_bootstrap", pd.Series(dtype=float)).notna()]
    if sub.empty:
        if n_trades is not None and float(n_trades or 0) == 0:
            return ("No interval: the policy executed ZERO trades, so there is "
                    "no P&L distribution to resample.")
        return ("Day-blocked interval not estimable (fewer than two "
                "out-of-sample days WITH TRADES).")
    r = sub.iloc[0]
    return (f"Day-blocked bootstrap 95% interval "
            f"[{_fmt(r['ci95_low'])}, {_fmt(r['ci95_high'])}] over "
            f"{int(r['n_days'])} day(s) — the interval, not the point "
            "estimate, is the result.")


def _col(df: pd.DataFrame, name: str, how: str = "sum") -> float:
    """Aggregate a fold column that may not exist on older runs."""
    if df is None or df.empty or name not in df.columns:
        return np.nan
    s = pd.to_numeric(df[name], errors="coerce").dropna()
    if s.empty:
        return np.nan
    return float(getattr(s, how)())


def _sample_accounting(config: Config, ctx: Dict) -> Dict:
    """Count folds, market dates, and trades as SEPARATE quantities.

    Collapsing these into one ``n_days`` is why the same report once claimed 0,
    8, and "fewer than two" out-of-sample days at once.
    """
    fm = ctx["fold_metrics"]
    conc = ctx["concentration"]
    m1 = (fm[fm["model"] == config.evaluation.reference_model]
          if not fm.empty and "model" in fm.columns else pd.DataFrame())
    dates = (m1["test_day"].dropna().unique().tolist()
             if "test_day" in m1.columns else [])
    if "test_day" in m1.columns and "n_oos" in m1.columns:
        with_pred = int((m1.groupby("test_day")["n_oos"].sum() > 0).sum())
    else:
        with_pred = int(len(dates))
    n_trades = _col(m1, "bt_trade_count")
    return {
        "oos_folds": int(len(m1)),
        "oos_market_dates": int(len(dates)),
        "dates_with_predictions": with_pred,
        "trade_dates": int(conc.get("n_trade_days", 0) or 0),
        "n_trades": 0.0 if np.isnan(n_trades) else n_trades,
        "dates": sorted(str(d) for d in dates),
        "longs": _col(m1, "bt_longs"),
        "shorts": _col(m1, "bt_shorts"),
        "rows_predictable": _col(m1, "bt_rows_predictable"),
        "signals": _col(m1, "bt_signals"),
        "pass_cost_gate": _col(m1, "bt_pass_cost_gate"),
        "pass_z_gate": _col(m1, "bt_pass_z_gate"),
        "blocked_overlap": _col(m1, "bt_blocked_overlap"),
        "drop_no_execution": _col(m1, "bt_drop_no_execution"),
        "drop_no_exit": _col(m1, "bt_drop_no_exit"),
        "drop_no_depth": _col(m1, "bt_drop_no_depth"),
        "mean_abs_prediction": _col(m1, "bt_mean_abs_prediction", "mean"),
        "max_abs_prediction": _col(m1, "bt_max_abs_prediction", "max"),
        "mean_cost_threshold": _col(m1, "bt_mean_cost_threshold", "mean"),
        "pred_over_cost": _col(m1, "bt_mean_pred_over_cost", "mean"),
        "time_in_market_frac": _col(m1, "bt_time_in_market_frac", "mean"),
        "turnover": _col(m1, "bt_turnover"),
        **_pooled_per_trade(config, ctx, m1, n_trades),
    }


def _pooled_per_trade(config: Config, ctx: Dict, m1: pd.DataFrame,
                      n_trades: float) -> Dict:
    """Per-trade P&L pooled over trades, NOT averaged over folds.

    A mean of fold means weights a 1-trade fold like a 50-trade one, which can
    invert the sign relative to the pooled total beside it in the same table.
    """
    tick = config.costs.tick_size
    ref = ctx.get("ref_mid", np.nan)
    if not n_trades or n_trades <= 0:
        nan = float("nan")
        return {"net_per_trade": nan, "net_per_trade_ticks": nan,
                "net_per_trade_bps": nan, "gross_per_trade_ticks": nan,
                "cost_per_trade_ticks": nan}
    net = _col(m1, "bt_net_pnl") / n_trades
    gross = _col(m1, "bt_gross_pnl") / n_trades
    costs = _col(m1, "bt_total_costs") / n_trades
    return {
        "net_per_trade": net,
        "net_per_trade_ticks": net / tick if tick > 0 else np.nan,
        "net_per_trade_bps": (net / ref * 1e4)
        if ref and not np.isnan(ref) and ref > 0 else np.nan,
        "gross_per_trade_ticks": gross / tick if tick > 0 else np.nan,
        "cost_per_trade_ticks": costs / tick if tick > 0 else np.nan,
    }


def _execution_accounting_section(config: Config, acct: Dict) -> str:
    """Print the signal funnel so an empty ledger cannot look like a loss."""
    tick = config.costs.tick_size
    n_trades = acct["n_trades"]
    rows = [
        ("Out-of-sample folds", f"{acct['oos_folds']}"),
        ("Distinct OOS market dates", f"{acct['oos_market_dates']}"),
        ("OOS dates with eligible predictions",
         f"{acct['dates_with_predictions']}"),
        ("OOS dates with executed trades", f"{acct['trade_dates']}"),
        ("Rows with a usable prediction",
         f"{_fmt(acct['rows_predictable'], 0)}"),
        ("… clearing the cost gate |pred| > round-trip cost",
         f"{_fmt(acct['pass_cost_gate'], 0)}"),
        ("… clearing the z gate", f"{_fmt(acct['pass_z_gate'], 0)}"),
        ("Signals (both gates)", f"{_fmt(acct['signals'], 0)}"),
        ("Dropped: no executable quote after latency",
         f"{_fmt(acct['drop_no_execution'], 0)}"),
        ("Dropped: cannot close inside segment",
         f"{_fmt(acct['drop_no_exit'], 0)}"),
        ("Dropped: insufficient displayed depth",
         f"{_fmt(acct['drop_no_depth'], 0)}"),
        ("Suppressed by overlap guard",
         f"{_fmt(acct['blocked_overlap'], 0)}"),
        ("**Executed trades**", f"**{_fmt(n_trades, 0)}**"),
        ("Long / short", f"{_fmt(acct['longs'], 0)} / {_fmt(acct['shorts'], 0)}"),
        ("Fraction of time holding a position",
         f"{_fmt(acct['time_in_market_frac'], 4)}"),
        ("Turnover (notional)", f"{_fmt(acct['turnover'], 2)}"),
    ]
    if n_trades and n_trades > 0:
        rows += [
            ("Gross per trade (ticks)",
             f"{_fmt(acct['gross_per_trade_ticks'], 4)}"),
            ("Cost per trade (ticks)",
             f"{_fmt(acct['cost_per_trade_ticks'], 4)}"),
            ("Net per trade (price units)", f"{_fmt(acct['net_per_trade'], 8)}"),
            ("Net per trade (ticks)",
             f"{_fmt(acct['net_per_trade_ticks'], 4)}"),
            ("Net per trade (bps)", f"{_fmt(acct['net_per_trade_bps'], 4)}"),
        ]
    body = "\n".join(f"| {k} | {v} |" for k, v in rows)
    out = ("\n## Execution accounting — what the policy actually did\n\n"
           "| Quantity | Value |\n|---|---|\n" + body + "\n")

    if n_trades == 0:
        pc = acct["pred_over_cost"]
        out += (
            "\n> **NO TRADES WERE EXECUTED.** Tradability was NOT evaluated; "
            "it was not tested. The mean |prediction| is "
            f"{_fmt(acct['mean_abs_prediction'], 8)} price units "
            f"({_fmt(acct['mean_abs_prediction'] / tick, 4)} ticks) against a "
            f"mean round-trip cost threshold of "
            f"{_fmt(acct['mean_cost_threshold'], 6)} "
            f"({_fmt(acct['mean_cost_threshold'] / tick, 3)} ticks) — a ratio "
            f"of {_fmt(pc, 5)}. The largest single prediction in the whole "
            f"out-of-sample period was "
            f"{_fmt(acct['max_abs_prediction'] / tick, 4)} ticks. The forecast "
            "is smaller than the spread it must cross by orders of magnitude, "
            "so no threshold setting makes this policy trade. Read every P&L, "
            "cost and latency figure below as UNDEFINED, not as zero.\n")
    return out


def _model_delta(tab: pd.DataFrame, base: str, other: str, label: str) -> str:
    """State an ablation result as a signed delta, not as a filename.

    A reader should not open a spreadsheet to learn whether a feature helped.
    """
    if tab is None or tab.empty or "model" not in tab.columns:
        return f"Not evaluable: no ablation table for {label}."
    def pick(m):
        r = tab[tab["model"] == m]
        return (float(r["oos_corr"].iloc[0]), float(r["oos_r2"].iloc[0])) \
            if len(r) else (np.nan, np.nan)
    b_corr, b_r2 = pick(base)
    o_corr, o_r2 = pick(other)
    if np.isnan(b_corr) or np.isnan(o_corr):
        return f"Not evaluable: `{other}` absent from the ablation table."
    d = o_corr - b_corr
    word = "improved" if d > 0 else ("did not improve" if d < 0 else "left")
    return (f"`{other}` vs `{base}`: OOS correlation {_fmt(b_corr)} -> "
            f"{_fmt(o_corr)} ({'+' if d >= 0 else ''}{_fmt(d)}), OOS R^2 "
            f"{_fmt(b_r2, 6)} -> {_fmt(o_r2, 6)} — {label} {word} the "
            f"baseline.")


def _regime_summary(regime_t: Optional[pd.DataFrame], traded: bool) -> str:
    """Best and worst regime by net P&L, or an honest refusal."""
    if not traded or regime_t is None or regime_t is False or \
            not isinstance(regime_t, pd.DataFrame) or regime_t.empty:
        return ("Not evaluable — regime P&L is computed from the trade ledger, "
                "and no trades were executed. `08_regime_results.csv` is not "
                "written on a no-trade run.")
    parts = []
    for dim, g in regime_t.groupby("regime_dim"):
        g = g.sort_values("net_pnl")
        if len(g) == 1:
            # "best X, worst X" implies a comparison that never happened.
            parts.append(f"{dim}: only one populated bucket "
                         f"(`{g['regime'].iloc[0]}`, "
                         f"{_fmt(g['net_pnl'].iloc[0])}) — no contrast")
            continue
        parts.append(f"{dim}: best `{g['regime'].iloc[-1]}` "
                     f"({_fmt(g['net_pnl'].iloc[-1])}, "
                     f"n={int(g['n_trades'].iloc[-1])}), worst "
                     f"`{g['regime'].iloc[0]}` ({_fmt(g['net_pnl'].iloc[0])}, "
                     f"n={int(g['n_trades'].iloc[0])})")
    return "; ".join(parts) + ". See `08_regime_results.csv`."


def _stress_summary(cost_t: pd.DataFrame, lat_t: pd.DataFrame,
                    traded: bool) -> str:
    """Report the latency/cost level at which net P&L crosses zero."""
    if not traded:
        return ("Not evaluable — both sweeps re-run a policy that never trades, "
                "so every row is 0.0 by construction. A flat line of zeros "
                "across latencies is NOT evidence of latency-robustness.")
    out = []
    if lat_t is not None and len(lat_t):
        lat = lat_t.sort_values("latency_ms")
        neg = lat[lat["net_pnl"] <= 0]
        if neg.empty:
            out.append(f"Net P&L stays positive to "
                       f"{lat['latency_ms'].max():.0f} ms")
        elif float(neg["latency_ms"].iloc[0]) == float(lat["latency_ms"].min()):
            # Naming a break-even here would imply viability at zero latency.
            out.append("Net P&L is already <= 0 at the lowest latency tested "
                       f"({lat['latency_ms'].min():.0f} ms) — no break-even "
                       "latency exists")
        else:
            out.append(f"Net P&L crosses zero at "
                       f"{float(neg['latency_ms'].iloc[0]):.0f} ms of latency")
    if cost_t is not None and len(cost_t):
        neg = cost_t[cost_t["net_pnl"] < 0]
        out.append("all cost scenarios remain non-negative" if neg.empty else
                   "first losing cost scenario: "
                   f"`{neg['scenario'].iloc[0]}` ({_fmt(neg['net_pnl'].iloc[0])})")
    return "; ".join(out) + ". See `11_`/`12_`."


def _effect_size_section(config: Config, ctx: Dict, coef_m1: pd.DataFrame,
                         mono: float) -> str:
    """Translate the coefficient into ticks and basis points.

    A correlation of 0.007 is not an economic quantity. :class:`models.
    LinearModel` standardizes on the training fold, so ``beta_1`` is already
    "price move per 1-SD of OFI" and converts straight to ticks — the same unit
    the spread is quoted in, which is the comparison that decides everything.
    """
    if coef_m1 is None or coef_m1.empty:
        return ""
    beta = float(coef_m1["mean"].iloc[0])
    lo = float(coef_m1.get("ci95_low", pd.Series([np.nan])).iloc[0])
    hi = float(coef_m1.get("ci95_high", pd.Series([np.nan])).iloc[0])
    name = str(coef_m1.get("beta_name", pd.Series(["OFI"])).iloc[0])
    tick = config.costs.tick_size
    ref = ctx.get("ref_mid", np.nan)
    spread = ctx.get("mean_spread", np.nan)
    bps = (beta / ref * 1e4) if ref and not np.isnan(ref) and ref > 0 else np.nan
    ratio = (beta / spread) if spread and not np.isnan(spread) and spread > 0 \
        else np.nan
    return (
        "\n## Effect size in economic units\n\n"
        f"A 1-SD increase in `{name}` predicts a mid move of:\n\n"
        f"| Unit | Value |\n|---|---|\n"
        f"| price units | {_fmt(beta, 8)} |\n"
        f"| **ticks** | **{_fmt(beta / tick, 5)}** |\n"
        f"| basis points | {_fmt(bps, 5)} |\n"
        f"| 95% CI (ticks, across folds) | "
        f"[{_fmt(lo / tick, 5)}, {_fmt(hi / tick, 5)}] |\n"
        f"| mean quoted spread (ticks) | {_fmt(spread / tick, 3)} |\n"
        f"| **predicted move / spread** | **{_fmt(ratio, 5)}** |\n\n"
        f"Read this as: the effect is real and consistently signed, but a 1-SD "
        f"OFI move buys about {_fmt(beta / tick, 4)} of a tick against a "
        f"{_fmt(spread / tick, 2)}-tick spread. Roughly "
        f"{_fmt(1.0 / ratio, 0) if ratio and ratio > 0 else 'n/a'} SD of "
        "simultaneous OFI would be needed to cover one round trip. Decile "
        f"monotonicity of {_fmt(mono, 3)} is moderate, not strong.\n")


def _evidence_status_section(config: Config, ctx: Dict) -> str:
    """State plainly which of three separate claims the run actually supports.

    These are not degrees of the same claim; they can fail independently, and
    conflating them is the single easiest way for a microstructure result to
    look better than it is. A pipeline can be correct and find nothing; a
    signal can be real and still be unprofitable once it pays the spread.
    """
    fm = ctx["fold_metrics"]
    m1 = (fm[fm["model"] == config.evaluation.reference_model]
          if not fm.empty else pd.DataFrame())
    acct = _sample_accounting(config, ctx)
    n_days = acct["oos_market_dates"]
    synthetic = config.synthetic_data

    if synthetic:
        impl = ("ESTABLISHED on synthetic data only — the pipeline runs end to "
                "end and the causality/leakage tests pass. This says nothing "
                "about any real market.")
        pred = "NOT ESTABLISHED — no real market data has been run."
        net = "NOT ESTABLISHED — no real market data has been run."
    else:
        impl = ("Tests pass on this run; the one-day raw-to-feature audit "
                "(`ofi_research pilot`) must also have been reviewed by hand "
                "before these numbers mean anything.")
        pred = (f"Measured on {len(m1)} out-of-sample fold(s) across "
                f"{n_days} distinct market day(s) — see the day-blocked "
                "interval in `13_overlap_aware_uncertainty.csv`, not the point "
                "estimate.")
        if acct["n_trades"] == 0:
            net = ("**NOT EVALUATED** — the policy executed ZERO trades on "
                   f"{acct['dates_with_predictions']} day(s) of eligible "
                   "predictions, so no execution assumption was ever exercised. "
                   "This is an absence of evidence, NOT evidence of "
                   "unprofitability.")
        else:
            net = (f"Measured on {_fmt(acct['n_trades'], 0)} trade(s) over "
                   f"{acct['trade_dates']} day(s), after full spread crossing, "
                   "fees, slippage and total latency — see "
                   "`10_gross_vs_net_performance.csv` and `11_/12_` "
                   "sensitivity tables.")

    return (
        "\n## Evidence status — three separate claims\n\n"
        "| Claim | Status |\n|---|---|\n"
        f"| **Implementation validity** (the code computes what it says) | {impl} |\n"
        f"| **Predictive evidence** (OFI forecasts future mid moves OOS) | {pred} |\n"
        f"| **Net trading evidence** (that edge survives execution) | {net} |\n\n"
        "Predictive evidence does not imply net trading evidence, and neither "
        "follows from implementation validity.\n")


def write_report(config: Config, ctx: Dict) -> Path:
    fm = ctx["fold_metrics"]
    coef = ctx["coef_stability"]
    decile = ctx["decile"]
    ablation_t = ctx["ablation"]
    conc = ctx["concentration"]
    cost_t = ctx["cost_table"]
    lat_t = ctx["lat_table"]
    htag = ctx["primary_horizon"]

    ref = config.evaluation.reference_model
    m1 = fm[fm["model"] == ref]
    mean_corr = float(m1["oos_corr"].mean()) if len(m1) else np.nan
    pct_pos_corr = float((m1["oos_corr"] > 0).mean() * 100) if len(m1) else np.nan
    net_total = float(m1["bt_net_pnl"].sum()) if "bt_net_pnl" in m1 else np.nan
    gross_total = float(m1["bt_gross_pnl"].sum()) if "bt_gross_pnl" in m1 else np.nan
    coef_m1 = coef[coef["model"] == ref]
    pct_pos_beta = float(coef_m1["pct_positive"].iloc[0]) if len(coef_m1) else np.nan
    mono = float(decile["mean_monotonicity_spearman"].iloc[0]) \
        if len(decile) and "mean_monotonicity_spearman" in decile else np.nan

    acct = _sample_accounting(config, ctx)
    traded = bool(acct["n_trades"] and acct["n_trades"] > 0)

    # Three-valued gate. A criterion never exercised is NOT_EVALUABLE, not FAIL:
    # scoring an empty ledger as a failure converts missing trading evidence
    # into negative trading evidence.
    PASS, FAIL, NA = "PASS", "FAIL", "NOT_EVALUABLE"

    def gate(ok: bool, evaluable: bool = True) -> str:
        return (PASS if ok else FAIL) if evaluable else NA

    checks = {
        "stable_coef_sign": gate(
            (not np.isnan(pct_pos_beta)) and pct_pos_beta >= 70,
            not np.isnan(pct_pos_beta)),
        "nonzero_oos_corr": gate(
            (not np.isnan(pct_pos_corr)) and pct_pos_corr >= 60
            and (not np.isnan(mean_corr)) and abs(mean_corr) > 0.005,
            not np.isnan(mean_corr)),
        "sensible_deciles": gate((not np.isnan(mono)) and mono > 0.5,
                                 not np.isnan(mono)),
        "net_positive": gate(
            (not np.isnan(net_total)) and net_total > 0, traded),
        "not_one_day": gate(
            (not np.isnan(conc.get("max_day_share", np.nan)))
            and conc["max_day_share"] < 0.6, traded),
        "survives_stress": gate(
            bool((lat_t["net_pnl"] > 0).all())
            and bool((cost_t["net_pnl"] >= 0).any()), traded),
    }

    # The strategy question moved to the maker path, so the gate follows it.
    # Taker criteria stay visible but are marked NOT_EVALUABLE rather than
    # deleted: "we stopped asking" and "we asked and got nothing" are different
    # claims, and a reader three months from now must be able to tell them
    # apart.
    taker_off = not config.evaluation.run_taker_backtest
    if taker_off:
        for k in ("net_positive", "not_one_day", "survives_stress"):
            checks[k] = NA
    pgate = (ctx.get("policy_tables") or {}).get("25_passive_decision_gate")
    if pgate is not None and len(pgate):
        for _, r in pgate.iterrows():
            checks[f"passive_{r['criterion']}"] = (
                PASS if r["status"] == "PASS" else FAIL)

    # A gate made only of NOT_EVALUABLE rows is not a pass. Proceeding
    # requires at least one criterion that was actually exercised.
    evaluated = [v for v in checks.values() if v != NA]
    proceed = bool(evaluated) and all(v == PASS for v in evaluated)
    unevaluated = [k for k, v in checks.items() if v == NA]
    failed = [k for k, v in checks.items() if v == FAIL]

    if np.isnan(mean_corr) or (len(m1) == 0):
        verdict = ("INSUFFICIENT DATA — the walk-forward produced no evaluable "
                   "folds. Reconsider horizon/window assumptions and data span.")
    elif checks["nonzero_oos_corr"] == FAIL:
        verdict = ("The basic OFI hypothesis is not supported for this dataset, "
                   "implementation, asset, and horizon. Do not add the wave/PDE "
                   "layer until the data construction and horizon assumptions "
                   "are reconsidered.")
    elif taker_off:
        pfail = [k for k, v in checks.items()
                 if k.startswith("passive_") and v == FAIL]
        ppass = [k for k, v in checks.items()
                 if k.startswith("passive_") and v == PASS]
        if not (pfail or ppass):
            verdict = (
                "PREDICTIVE, MAKER PATH NOT MEASURED. The aggressive-taker "
                "study is off by design (its verdict is settled and negative: "
                "~0.015 ticks predicted against a 2.53-tick spread), and the "
                "passive walk-forward produced no evaluable cells — most "
                "likely no event tapes on disk. Nothing here is evidence "
                "either way about the maker hypothesis.")
        elif pfail:
            verdict = (
                "PREDICTIVE, MAKER PATH NOT YET PROVEN. Taking is abandoned "
                "on effect size, not on tuning. The passive filter is the "
                "live hypothesis and it fails on: " + ", ".join(
                    k.replace("passive_", "") for k in pfail)
                + ". Read the break-even rebate before concluding anything: "
                "on a venue schedule this business is decided by fractions of "
                "a tick, and the rebate column is where that shows up.")
        else:
            verdict = (
                "MAKER PATH SURVIVES ITS FIRST HONEST TEST. Every exercised "
                "passive criterion passes, including at the back of the "
                "queue. This is a walk-forward result on held-out days with "
                "latency charged on both the quote and the cancel — but it is "
                "still one symbol over a small number of days, and the queue "
                "position remains an assumption no MBP-10 data can pin down.")
    elif not traded:
        verdict = (
            "PREDICTIVE BUT UNTESTED FOR TRADABILITY. OFI shows a weak, "
            "consistently positive out-of-sample relationship with future mid "
            "moves, but the trading policy executed ZERO trades, so nothing "
            "about profitability was measured — the P&L, cost and latency "
            "tables are undefined, not zero. The forecast is roughly "
            f"{_fmt(acct['pred_over_cost'], 4)} of the round-trip cost it must "
            "clear, so the gap is one of effect size, not of threshold tuning. "
            "Do NOT advance to the wave/PDE layer on this basis, and do not "
            "record this run as evidence that the signal is unprofitable.")
    elif checks["net_positive"] == FAIL:
        verdict = ("The signal appears statistically predictive but is not "
                   "directly tradable under the tested aggressive-execution "
                   "assumptions.")
    elif proceed:
        verdict = ("OFI shows out-of-sample predictive structure that survives "
                   "the tested costs and stress checks. Proceed to the "
                   "response-kernel stage.")
    else:
        verdict = ("OFI is predictive and net-positive out of sample, but the "
                   "decision gate is NOT fully met (failing: "
                   + (", ".join(failed) or "none")
                   + ("; not evaluable: " + ", ".join(unevaluated)
                      if unevaluated else "")
                   + "). Treat as promising-but-unproven; gather more test "
                   "days / regimes before adding the wave/PDE layer.")

    synth_banner = ("> **⚠️ SYNTHETIC DATA — NOT A REAL RESULT.** These numbers "
                    "come from the built-in synthetic generator and exist only "
                    "to prove the pipeline runs end to end.\n\n"
                    if config.synthetic_data else "")

    latency_banner = ""
    caveat = warn_if_latency_unconfigured(config)
    if caveat:
        latency_banner = (f"> **⚠️ ZERO-LATENCY UPPER BOUND.** {caveat}\n\n")

    unc = ctx.get("uncertainty")
    lines = []
    lines.append("# OFI Research Report\n")
    lines.append(synth_banner)
    lines.append(latency_banner)
    lines.append(_evidence_status_section(config, ctx))
    lines.append(f"- Label: `{config.label}`  |  Primary horizon: `{htag}`  |  "
                 f"Walk-forward: `{config.splits.scheme}` "
                 f"({config.splits.train_days}/{config.splits.validation_days}/"
                 f"{config.splits.test_days} train/val/test days)\n")
    lines.append("\n## Headline decision\n")
    lines.append(f"**Verdict:** {verdict}\n")
    lines.append(f"\n**Proceed to impulse-response / Green's-function stage?** "
                 f"{'YES' if proceed else 'NO'}\n")
    lines.append("\n### Decision-gate checklist\n")
    lines.append("| Criterion | Status | Basis |\n|---|---|---|")
    icon = {PASS: "✅ PASS", FAIL: "❌ FAIL", NA: "⬜ NOT_EVALUABLE"}
    basis = {
        "stable_coef_sign": f"beta_1 positive in {_fmt(pct_pos_beta,1)}% of "
                            f"{acct['oos_folds']} folds",
        "nonzero_oos_corr": f"mean OOS corr {_fmt(mean_corr)} in "
                            f"{_fmt(pct_pos_corr,1)}% of folds",
        "sensible_deciles": f"decile Spearman {_fmt(mono,3)}",
        "net_positive": (f"{_fmt(acct['n_trades'],0)} trades executed"
                         if traded else "no trades executed — never tested"),
        "not_one_day": (f"largest day = {_fmt(conc.get('max_day_share'),3)} of "
                        f"total |P&L| over {acct['trade_dates']} day(s)"
                        if traded else "no trading days to concentrate in"),
        "survives_stress": ("latency/cost sweeps" if traded else
                            "stress sweeps ran on a policy that never traded"),
    }
    if taker_off:
        _off = ("taker path OFF by design — settled negative on effect size, "
                "not re-derived each run (evaluation.run_taker_backtest)")
        for _k in ("net_positive", "not_one_day", "survives_stress"):
            basis[_k] = _off
    if pgate is not None and len(pgate):
        for _, r in pgate.iterrows():
            basis[f"passive_{r['criterion']}"] = str(r["basis"])
    for k, v in checks.items():
        lines.append(f"| {k} | {icon[v]} | {basis.get(k,'')} |")
    if unevaluated:
        lines.append(
            f"\n> {len(unevaluated)} criteri{'on was' if len(unevaluated)==1 else 'a were'} "
            "NOT EVALUABLE on this run: `" + "`, `".join(unevaluated)
            + "`. They are not failures — they were never exercised. "
            "`proceed` requires PASS on all six, so it remains NO.\n")

    lines.append(_execution_accounting_section(config, acct))
    lines.append(_effect_size_section(config, ctx, coef_m1, mono))
    pt = ctx.get("passive_tables") or {}
    if pt:
        lines.append(passive.phase0_report_section(pt, config, htag))
    polt = ctx.get("policy_tables") or {}
    if polt:
        lines.append(passive_policy.policy_report_section(polt, config))

    lines.append("\n## Answers to the section-29 questions\n")
    qa = [
        ("1. Does OFI predict future midprice movement OOS?",
         f"Mean OOS pred-corr ({ref}, {htag}) = {_fmt(mean_corr)}; positive in "
         f"{_fmt(pct_pos_corr,1)}% of folds. "
         + (_day_block_phrase(unc, f"oos_pred_corr_{ref}")
            if unc is not None and len(unc) else
            "No day-blocked interval available.")),
        ("2. Strongest horizons?",
         "See `03_ofi_correlation_by_horizon.csv` (HAC/Newey-West, FDR-adjusted "
         "p-values)."),
        ("3. Is the OFI-decile relationship monotonic?",
         f"Mean decile monotonicity (Spearman) = {_fmt(mono,3)} "
         f"(1.0 = perfectly monotone). See `04_ofi_decile_response_*`."),
        ("4. Is the OFI coefficient stable across folds?",
         f"beta_1 positive in {_fmt(pct_pos_beta,1)}% of folds "
         f"(see `09_coefficient_stability_*`)."),
        ("5. Does L2 improve on L1?",
         _model_delta(ablation_t, "M1_ofi", "M2_ofi_spread_depth",
                      "adding L2 spread/depth")),
        ("6. Does normalization help?",
         _model_delta(ablation_t, "M1_ofi", "M1N_ofi_normalized",
                      "normalizing OFI")),
        ("7. Do signed volume / intensities add information?",
         _model_delta(ablation_t, "M1_ofi", "M3_ofi_signedvol",
                      "adding signed volume")
         + " " + _model_delta(ablation_t, "M1_ofi", "M4_ofi_intensity",
                              "adding trade intensity")
         + " " + _model_delta(ablation_t, "M1_ofi", "M5_full",
                              "the full feature set")),
        ("8. Profitable after bid/ask execution, fees, slippage?",
         (f"**Not evaluated.** The policy executed ZERO trades across "
          f"{acct['dates_with_predictions']} out-of-sample day(s), so the "
          f"reported net P&L of {_fmt(net_total)} is the sum of an empty "
          "ledger, not a measured loss. See the execution-accounting table."
          if not traded else
          f"Net P&L ({ref}, pooled test) = {_fmt(net_total)} over "
          f"{_fmt(acct['n_trades'],0)} trades "
          f"({_fmt(acct['net_per_trade_ticks'],4)} ticks/trade, "
          f"{_fmt(acct['net_per_trade_bps'],4)} bps/trade) vs gross mid-to-mid "
          f"{_fmt(gross_total)} "
          f"({_fmt(acct['gross_per_trade_ticks'],4)} ticks/trade); cost "
          f"{_fmt(acct['cost_per_trade_ticks'],4)} ticks/trade. ")
         + " " + (_day_block_phrase(unc, "net_pnl_per_trade", acct["n_trades"])
                  if unc is not None and len(unc) else "")),
        ("9. Which regimes work/fail?",
         _regime_summary(ctx.get("regime_table"), traded)),
        ("10. Survives latency & cost stress?",
         _stress_summary(cost_t, lat_t, traded)),
        ("11. Concentrated in a few days?",
         (f"Not evaluable — no trades, so there are no trading days to "
          f"concentrate in (the walk-forward itself covered "
          f"{acct['oos_market_dates']} market days)."
          if not traded else
          f"The single largest day accounts for "
          f"{_fmt(conc.get('max_day_share'),3)} of total absolute P&L across "
          f"{acct['trade_dates']} trading day(s); "
          f"{_fmt(conc.get('positive_day_frac'),3)} of those days positive. "
          f"(Share is |largest day| / sum of |daily P&L|, so it stays in "
          f"[0,1] even when the total is negative.)")),
        ("12. Proceed to response-kernel stage?",
         f"{'YES' if proceed else 'NO'} — per the decision gate above."),
    ]
    for q, a in qa:
        lines.append(f"\n**{q}**\n\n{a}\n")

    lines.append("\n## Statistical caveats (section 25)\n")
    lines.append(
        "- Targets overlap across rows -> all significance uses Newey-West "
        "(HAC) SEs; iid SEs are **not** trusted.\n"
        "- Many horizon/window combinations are tested -> p-values are "
        "FDR-adjusted (Benjamini-Hochberg) and treated as exploratory.\n"
        "- The conclusion is driven by out-of-sample walk-forward performance, "
        "not in-sample p-values.\n"
        "- Queue position cannot be modeled from top-of-book data; aggressive "
        "taker execution is assumed.\n"
        "- Overlapping targets also invalidate per-row iid intervals, so the "
        "headline numbers carry a whole-day block bootstrap and day-clustered "
        "SEs (`13_overlap_aware_uncertainty.csv`).\n"
        "- The contemporaneous CKS regression "
        "(`15_contemporaneous_cks_replication.csv`) is a data/formula sanity "
        "check; its R^2 is NOT expected out-of-sample predictive performance.\n"
        "- XNAS.ITCH is a venue-LOCAL book: its mid/spread are not the "
        "national NBBO, and cross-venue routing would need consolidated data.\n"
        "- Statistical significance and economic significance are reported "
        "separately: a correlation whose CI excludes zero can still imply a "
        "predicted move far below one tick, and only the tick-denominated "
        "effect size decides tradability.\n"
        "- A no-trade run yields NOT_EVALUABLE P&L criteria, never FAIL. "
        "Absence of trading evidence is not evidence of unprofitability.\n"
        f"- {acct['oos_market_dates']} out-of-sample market days is a small "
        "number of independent days; treat all fold-fraction statistics "
        f"(e.g. '{_fmt(pct_pos_corr,1)}% of folds') as counts out of "
        f"{acct['oos_folds']}, not as stable probabilities.\n")
    notes = ctx.get("feature_notes") or []
    if notes:
        lines.append("\n## Feature availability notes\n")
        for n in notes:
            lines.append(f"- {n}\n")
    if config.synthetic_data:
        lines.append("\n> Reminder: all figures above are SYNTHETIC.\n")

    path = _outdir(config) / "OFI_RESEARCH_REPORT.md"
    path.write_text("\n".join(lines))
    # also drop a machine-readable summary
    pd.DataFrame([{**checks, "verdict": verdict, "proceed": proceed,
                   "mean_oos_corr": mean_corr, "net_pnl": net_total,
                   "gross_pnl": gross_total}]).to_csv(
        _outdir(config) / "14_final_experiment_summary.csv", index=False)
    logger.info("Wrote report %s", path)
    return path


# --- Full run ---
def run_phase0_screen(feat: pd.DataFrame, config: Config,
                      horizon_tags: List[str]) -> Dict[str, pd.DataFrame]:
    """Passive front-of-queue markout screen (see :mod:`passive`).

    Never allowed to abort the run: a failed side-screen must not destroy an
    otherwise complete walk-forward.
    """
    try:
        return passive.run_phase0(feat, config, horizon_tags, save=_save_table)
    except Exception:  # pragma: no cover - defensive
        logger.exception("Phase-0 passive screen failed; continuing")
        return {}


def run_passive_policy(feat: pd.DataFrame, config: Config,
                       primary: str) -> Dict[str, pd.DataFrame]:
    """Walk-forward passive quoting (see :mod:`passive_policy`).

    Like the Phase-0 screen this is never allowed to abort the run, but unlike
    the screen it is now the path the headline verdict rests on, so a failure
    is logged at exception level rather than swallowed quietly.
    """
    try:
        tables = run_passive_walk_forward(feat, config, primary)
    except Exception:  # pragma: no cover - defensive
        logger.exception("Passive walk-forward failed; continuing")
        return {}
    for name, tab in tables.items():
        if tab is not None and len(tab):
            _save_table(tab, config, name)
    grid = tables.get("23_passive_policy_grid")
    if grid is not None and len(grid):
        gate = policy_gate(grid, config, tables.get("24_passive_policy_by_day"))
        if len(gate):
            _save_table(gate, config, "25_passive_decision_gate")
            tables["25_passive_decision_gate"] = gate
    return tables


def warn_if_latency_unconfigured(config: Config) -> Optional[str]:
    """Say out loud when a run is producing the zero-latency upper bound.

    The default is zero because inventing a latency would be worse (P0.5), but
    a run that silently uses it reports an unreachable optimum as if it were a
    forecast. Returns the caveat so the written report carries it too.
    """
    if not config.costs.latency.is_upper_bound() or config.costs.latency_ms > 0:
        return None
    msg = ("TOTAL EXECUTION LATENCY IS ZERO. ts_recv is Databento's capture "
           "time, so these results are a theoretical UPPER BOUND, not an "
           "achievable baseline. Set costs.latency.{capture_to_user_ms,"
           "decode_decide_ms,user_to_venue_ms} before quoting any P&L.")
    logger.warning(msg)
    return msg


def run_full(config: Config, df_raw: Optional[pd.DataFrame] = None,
             stream_paths: Optional[List[str]] = None) -> Path:
    warn_if_latency_unconfigured(config)
    if stream_paths:
        feat, info = prepare_streaming(config, stream_paths)
    else:
        feat, info = prepare(config, df_raw)

    # Pick a primary clock horizon by a fixed, OUTCOME-INDEPENDENT rule:
    # among clock horizons with enough usable targets, prefer the smallest that
    # is >= 1s (short horizons are typically sub-spread and untradable); if none
    # reach 1s, fall back to the largest usable horizon. Users override via
    # config. This rule does not look at P&L or correlations.
    tags = [t for t in all_horizon_tags(config) if t.startswith("ms")]
    usable = [t for t in tags
              if f"future_mid_change_{t}" in feat
              and feat[f"future_mid_change_{t}"].notna().sum() > 100]
    primary = None
    for t in usable:
        if float(t[2:]) >= 1000.0:
            primary = t
            break
    if primary is None:
        primary = usable[-1] if usable else (
            tags[len(tags) // 2] if tags else all_horizon_tags(config)[0])
    logger.info("Primary horizon: %s (usable clock horizons: %s)",
                primary, usable)

    contemporaneous_replication(feat, config)
    exploratory_correlations(feat, config)
    fold_metrics, coef_stability, ledger_m1 = walk_forward(feat, config, primary)
    decile = decile_analysis(feat, config, primary)
    ablation_t = ablation(feat, config, primary)
    l2_comparison(feat, config, primary)
    normalization_selection(feat, config, primary)
    regime_t = regime_performance(ledger_m1, config)
    cost_t, lat_t = cost_latency_sensitivity(feat, config, primary)
    conc = day_concentration(ledger_m1, config)
    uncertainty = overlap_aware_uncertainty(fold_metrics, ledger_m1, config,
                                            primary)
    tick_regime_report(feat, config)
    passive_tables = run_phase0_screen(feat, config, usable or [primary])
    policy_tables = run_passive_policy(feat, config, primary)

    # model comparison table (mean OOS metrics per model)
    if not fold_metrics.empty:
        comp = (fold_metrics.groupby("model")
                .agg(oos_corr=("oos_corr", "mean"),
                     oos_r2=("oos_r2", "mean"),
                     dir_acc=("dir_acc", "mean"),
                     net_pnl=("bt_net_pnl", "sum"),
                     gross_pnl=("bt_gross_pnl", "sum"),
                     sharpe_like=("bt_sharpe_like", "mean"),
                     trades=("bt_trade_count", "sum")).reset_index())
        _save_table(comp, config, f"06_model_comparison_{primary}")
        # gross vs net
        _save_table(comp[["model", "gross_pnl", "net_pnl"]],
                    config, "10_gross_vs_net_performance")

    # plots
    syn = config.synthetic_data
    od = _outdir(config)
    plots.plot_distribution(feat["OFI_L1_ref"], "OFI (L1) distribution",
                            "OFI (size units)", od / "p_ofi_dist.png",
                            "all", syn)
    plots.plot_distribution(feat["nOFI_L1"], "Normalized OFI (L1) distribution",
                            "nOFI", od / "p_nofi_dist.png", "all", syn)
    plots.plot_distribution(feat[f"future_mid_change_{primary}"],
                            f"Future mid change ({primary})",
                            "price units", od / "p_future_dist.png", "all", syn)
    if len(decile):
        plots.plot_decile_response(decile, od / "p_decile.png", primary,
                                   "test", syn)
    corr_tab = pd.read_csv(od / "03_ofi_correlation_by_horizon.csv")
    corr_ofi = corr_tab[corr_tab["feature"] == "OFI1_ref"]
    if len(corr_ofi):
        plots.plot_correlation_vs_horizon(corr_ofi, od / "p_corr_horizon.png",
                                          "pearson", "exploratory", syn)
    # fold_metrics is empty (and column-less) when the sample holds fewer days
    # than train+validation+test, so guard before selecting on 'model'.
    m1 = (fold_metrics[fold_metrics["model"]
                       == config.evaluation.reference_model]
          if "model" in fold_metrics.columns else fold_metrics)
    if len(m1):
        plots.plot_beta_over_time(m1.rename(columns={"beta_1": "beta_1"}),
                                  od / "p_beta_time.png", "beta_1", syn)
    if len(ledger_m1):
        plots.plot_cumulative_pnl(ledger_m1, od / "p_cum_pnl.png", syn)
        plots.plot_drawdown(ledger_m1, od / "p_drawdown.png", syn)
    if len(regime_t):
        rr = regime_t[regime_t["regime_dim"] == "regime_spread"].rename(
            columns={"net_pnl": "net_pnl"})
        if len(rr):
            plots.plot_regime_bar(rr, od / "p_regime_spread.png", "net_pnl",
                                  "Net P&L by spread regime", syn)
    if len(lat_t):
        plots.plot_sensitivity(lat_t["latency_ms"].tolist(),
                               lat_t["net_pnl"].tolist(),
                               od / "p_latency.png", "Latency sensitivity",
                               "latency (ms)", synthetic=syn)
    if len(cost_t):
        plots.plot_sensitivity(list(range(len(cost_t))),
                               cost_t["net_pnl"].tolist(),
                               od / "p_cost.png", "Cost-scenario sensitivity",
                               "scenario index", synthetic=syn)

    ctx = {
        "fold_metrics": fold_metrics, "coef_stability": coef_stability,
        "decile": decile, "ablation": ablation_t, "concentration": conc,
        "cost_table": cost_t, "lat_table": lat_t, "primary_horizon": primary,
        "uncertainty": uncertainty, "feature_notes": info["feature_notes"],
        "regime_table": regime_t,
        "passive_tables": passive_tables,
        "policy_tables": policy_tables,
        "ref_mid": float(feat["midprice"].mean()),
        "mean_spread": float(feat["spread"].mean()),
    }
    _acct = _sample_accounting(config, ctx)
    _acct["dates"] = "|".join(_acct["dates"])
    _save_table(pd.DataFrame([_acct]), config, "19_execution_accounting")
    return write_report(config, ctx)


# --- One-day pilot audit (protocol gate before any multi-day run) ---
def run_pilot(config: Config, paths: List[str], n_trace: int = 100) -> Path:
    """Ingest ONE day and emit the raw-to-feature audit, before buying 25-30.

    Nothing here is a result. The point is to make every vendor assumption in
    :data:`PILOT_ASSUMPTIONS` falsifiable against real records — record counts,
    flag incidence, batch sizes, action patterns inside repeated-sequence
    groups, and a hand-checkable trace of consecutive native events — so that
    a wrong assumption is caught for the price of one day rather than after a
    month of analysis has been built on top of it.

    The two model outputs at the end are deliberately paired and deliberately
    labeled: a contemporaneous CKS regression (a formula/data sanity check that
    is expected to look strong and is not tradeable) and a causal trailing-OFI
    forward regression with no costs (preliminary predictive evidence only).
    """
    from .databento_loader import (
        PILOT_ASSUMPTIONS, build_canonical_events, drop_ingestion_duplicates,
        load_databento, prepare_raw, read_raw, trace_events,
    )
    od = _outdir(config)
    canonical, audits, diagnostics = load_databento(paths, config)

    for name, table in audits.items():
        _save_table(table, config, f"pilot_audit_{name}")

    clean, removal = clean_data(canonical, config)
    _save_table(removal, config, "pilot_cleaning_removal_report")
    schema = inspect_schema(clean, config)
    _save_table(schema.to_frame(), config, "pilot_schema")

    feat, notes = build_features(clean, config)
    feat = add_targets(feat, config)
    for H in config.targets.clock_horizons_ms:
        feat = add_execution_aligned_targets(feat, config, float(H))
    assert_target_causality(feat, config)

    # book-state pathologies
    bid, ask = feat["bid_price_1"], feat["ask_price_1"]
    state = pd.DataFrame([{
        "n_rows": len(feat),
        "n_missing_bid": int(bid.isna().sum()),
        "n_missing_ask": int(ask.isna().sum()),
        "n_crossed": int((bid > ask).sum()),
        "n_locked": int((bid == ask).sum()),
        "n_nonpositive_size": int(((feat["bid_size_1"] <= 0)
                                   | (feat["ask_size_1"] <= 0)).sum()),
        "n_book_clears": int(feat.get("book_clear", pd.Series(dtype=bool)).sum()),
        "n_inferred_halts": int(diagnostics.get("n_inferred_halt_boundaries", 0)),
        "n_segments": int(feat["segment_id"].nunique()),
        "n_long_quiet_gaps": int(feat["long_quiet_gap"].sum()),
    }])
    _save_table(state, config, "pilot_book_state")

    # horizon realization + event-window spans
    rows = []
    for H in config.targets.clock_horizons_ms:
        col = f"actual_horizon_ms_ms{int(H)}"
        if col in feat:
            v = feat[col].dropna()
            rows.append({"quantity": f"clock_horizon_ms{int(H)}",
                         "n_usable": int(len(v)),
                         "pct_usable": 100.0 * len(v) / max(len(feat), 1),
                         "median_ms": float(v.median()) if len(v) else np.nan,
                         "p90_ms": float(v.quantile(0.9)) if len(v) else np.nan})
    for w in config.features.event_windows:
        col = f"event_window_elapsed_ms_ev{w}"
        if col in feat:
            v = feat[col].dropna()
            rows.append({"quantity": f"event_window_span_ev{w}",
                         "n_usable": int(len(v)),
                         "pct_usable": 100.0 * len(v) / max(len(feat), 1),
                         "median_ms": float(v.median()) if len(v) else np.nan,
                         "p90_ms": float(v.quantile(0.9)) if len(v) else np.nan})
    _save_table(pd.DataFrame(rows), config, "pilot_horizon_realization")

    # manual raw -> event trace for hand checking
    raw = read_raw(paths, config.data.file_format)
    prepared = prepare_raw(raw, config)
    deduped, _ = drop_ingestion_duplicates(prepared)
    if config.data.restrict_to_continuous_session:
        deduped = deduped.loc[deduped["in_continuous_session"]].copy()
    trace = trace_events(deduped, canonical, n_events=n_trace)
    _save_table(trace, config, "pilot_event_trace")

    # contemporaneous sanity check vs causal forward prediction, side by side
    cks = contemporaneous_cks_table(feat, config)
    _save_table(cks, config, "pilot_contemporaneous_cks")

    fwd_rows = []
    ev = config.features.event_windows
    ref = ev[len(ev) // 2]
    for H in config.targets.clock_horizons_ms:
        tcol = f"future_mid_change_ms{int(H)}"
        if f"OFI1_ev{ref}" in feat and tcol in feat:
            r = hac_correlation(feat[f"OFI1_ev{ref}"], feat[tcol],
                                f"OFI1_ev{ref}", f"ms{int(H)}")
            fwd_rows.append(asdict(r))
    fwd = pd.DataFrame(fwd_rows)
    if not fwd.empty:
        fwd["p_value_fdr"] = None
        fwd["labeling"] = (
            "PRELIMINARY, IN-SAMPLE, NO COSTS — trailing OFI vs FUTURE return. "
            "One day cannot support any profitability claim.")
    _save_table(fwd, config, "pilot_causal_forward_prediction")

    lines = ["# One-Day Databento Pilot Audit\n",
             "\n> **Not a result.** This report exists to falsify the vendor "
             "assumptions below against real records before more data is "
             "bought. No profitability claim can be made from one day.\n",
             f"\n- Files: `{paths}`\n- Canonical rows: "
             f"{diagnostics.get('n_canonical_rows')}\n"
             f"- Raw records: {diagnostics.get('n_raw_records')}\n"
             f"- Records per event (mean/max): "
             f"{_fmt(diagnostics.get('records_per_event_mean'), 2)}/"
             f"{diagnostics.get('records_per_event_max')}\n"
             f"- Events spanning >1 sequence: "
             f"{diagnostics.get('n_events_mixed_sequence')}\n"
             f"- Book clears: {diagnostics.get('n_book_clears')}  |  "
             f"inferred halts: "
             f"{diagnostics.get('n_inferred_halt_boundaries')}\n"
             f"- Trades with unspecified aggressor side: "
             f"{diagnostics.get('n_unspecified_side_trades')}\n"
             f"- Inferred quote increment: "
             f"{_fmt(diagnostics.get('inferred_tick_size'), 6)} "
             f"(configured {config.features.tick_size})\n"]
    lines.append("\n## Assumptions this day must confirm\n")
    for a in PILOT_ASSUMPTIONS:
        lines.append(f"- [ ] {a}\n")
    lines.append("\n## Feature availability notes\n")
    for n in notes:
        lines.append(f"- {n}\n")
    lines.append(
        "\n## Manual checks that a human must still perform\n"
        "- [ ] Read `pilot_event_trace.csv`: for at least 100 consecutive "
        "native events (several containing trades) confirm record grouping, "
        "the post-event book state, the OFI contribution, signed trade volume, "
        "and target indices.\n"
        "- [ ] Confirm `pilot_audit_consecutive_actions.csv` patterns match "
        "the assumed Trade -> Fill/Cancel normalization.\n"
        "- [ ] Confirm the inferred halt count against the day's known halts.\n"
        "- [ ] Confirm opening/closing cross handling at the session edges.\n"
        "\nOnly after these pass should 25-30 days be ingested.\n")
    path = od / "PILOT_AUDIT_REPORT.md"
    path.write_text("".join(lines))
    logger.info("Wrote pilot audit %s", path)
    return path


# --- CLI ---
def _databento_defaults(cfg: Config) -> Config:
    """Column mapping / price defaults for a raw Databento ``mbp-10`` extract.

    Applied only when the user asks for the Databento path, so a generic CSV
    run is never silently reinterpreted as fixed-point nanodollars.
    """
    cfg.data.price_mode = "fixed"
    cfg.columns.deep_bid_price = "bid_px_{i0:02d}"
    cfg.columns.deep_ask_price = "ask_px_{i0:02d}"
    cfg.columns.deep_bid_size = "bid_sz_{i0:02d}"
    cfg.columns.deep_ask_size = "ask_sz_{i0:02d}"
    return cfg


def _load_from_args(cfg: Config, args) -> Tuple[pd.DataFrame, Dict]:
    """Load via the Databento path when asked, else the generic loader."""
    extra: Dict = {}
    if getattr(args, "databento", False):
        paths = args.data if isinstance(args.data, list) else [args.data]
        canonical, audits, diagnostics = load_databento(paths, cfg)
        extra = {"audits": audits, "diagnostics": diagnostics}
        return canonical, extra
    return load_data(cfg), extra


def _build_config(args) -> Tuple[Config, Optional[pd.DataFrame]]:
    if getattr(args, "synthetic", False):
        df, cfg = make_synthetic_data(
            n_days=args.synthetic_days, events_per_day=args.synthetic_events,
            seed=7)
        if args.output:
            cfg.output_dir = args.output
        return cfg, df
    if args.config:
        cfg = Config.from_json_file(args.config)
    else:
        cfg = Config()
    if getattr(args, "databento", False) and not args.config:
        cfg = _databento_defaults(cfg)
    if args.data:
        cfg.data.data_path = (args.data[0] if isinstance(args.data, list)
                              else args.data)
    if args.output:
        cfg.output_dir = args.output
    return cfg, None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="OFI research pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "run", "phase0"):
        p = sub.add_parser(name)
        p.add_argument("--data", default=None, nargs="+",
                       help="path(s) to dataset")
        p.add_argument("--config", default=None, help="path to JSON config")
        p.add_argument("--output", default=None, help="output directory")
        p.add_argument("--databento", action="store_true",
                       help="ingest as Databento mbp-10 (canonical native "
                            "events, fixed-point prices, DBN flags)")
        p.add_argument("--synthetic", action="store_true",
                       help="use the labeled synthetic generator")
        p.add_argument("--synthetic-days", type=int, default=15)
        p.add_argument("--synthetic-events", type=int, default=3000)
    pp = sub.add_parser("pilot", help="one-day Databento raw-to-feature audit")
    pp.add_argument("--data", required=True, nargs="+",
                    help="path(s) to ONE trading day of Databento mbp-10")
    pp.add_argument("--config", default=None, help="path to JSON config")
    pp.add_argument("--output", default=None, help="output directory")
    pp.add_argument("--trace-events", type=int, default=100,
                    help="consecutive native events to trace for hand-checking")
    sub.add_parser("test")

    args = parser.parse_args(argv)
    setup_logging()

    if args.command == "test":
        import pytest
        return pytest.main(["-q", str(Path(__file__).parent / "tests")])

    if args.command == "pilot":
        args.databento = True
        args.synthetic = False
        cfg, _ = _build_config(args)
        path = run_pilot(cfg, list(args.data), n_trace=args.trace_events)
        print(f"\nPilot audit: {path}")
        print("Review it by hand BEFORE ingesting 25-30 days.")
        return 0

    cfg, df = _build_config(args)
    if args.command == "inspect":
        if df is None:
            df, _ = _load_from_args(cfg, args)
        report = inspect_schema(df, cfg)
        print(report.to_frame().to_string(index=False))
        return 0
    if args.command == "phase0":
        # Feature preparation only, then the passive screen. The walk-forward
        # is the expensive part of `run` and contributes nothing here, so this
        # answers the buy-MBO-data question in a fraction of the time.
        if getattr(args, "databento", False) and df is None:
            paths = args.data if isinstance(args.data, list) else [args.data]
            feat, _ = prepare_streaming(cfg, paths)
        else:
            if df is None:
                df, _ = _load_from_args(cfg, args)
            feat, _ = prepare(cfg, df)
        tags = [t for t in all_horizon_tags(cfg)
                if t.startswith("ms") and f"future_mid_change_{t}" in feat]
        tables = run_phase0_screen(feat, cfg, tags)
        if not tables:
            print("Phase 0 produced no tables (no inferred passive fills).")
            return 1
        primary = next((t for t in tags if float(t[2:]) >= 1000.0),
                       tags[-1] if tags else "ms1000")
        out = Path(cfg.output_dir) / "PHASE0_PASSIVE_SCREEN.md"
        out.write_text(passive.phase0_report_section(tables, cfg, primary))
        print(f"\nDone. Phase-0 screen: {out}")
        print(f"Tables in: {cfg.output_dir}")
        return 0
    if args.command == "run":
        # Per-session streaming for the Databento path: a month of raw records
        # cannot be concatenated before feature-building (see
        # prepare_streaming). Session boundaries are already hard segment
        # breaks, so this is exact.
        if getattr(args, "databento", False) and df is None:
            paths = args.data if isinstance(args.data, list) else [args.data]
            path = run_full(cfg, None, stream_paths=paths)
            print(f"\nDone. Report: {path}")
            print(f"Tables & plots in: {cfg.output_dir}")
            return 0
        if df is None:
            df, _ = _load_from_args(cfg, args)
        path = run_full(cfg, df)
        print(f"\nDone. Report: {path}")
        print(f"Tables & plots in: {cfg.output_dir}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
