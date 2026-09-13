"""Phase 0: front-of-queue passive-fill markout, conditioned on OFI.

The taker study is settled and negative: OFI predicts ~0.025 ticks per 1-SD
against a full-spread round trip (~2.5 ticks on INTC). A maker does not PAY
that spread, it EARNS it, which reframes OFI's job from "forecast a move bigger
than the spread" (hopeless) to "forecast whether THIS fill is about to go bad"
(plausible).

Answering that properly needs market-by-order data for queue position, which is
a separate purchase. This module exists so the purchase is never made on faith:
it bounds passive performance from the MBP-10 data already on disk, assuming
our order sits at the FRONT of the queue and so fills on every trade at our
price. Real queue position is always worse, so an unprofitable bound means
every realistic queue position is unprofitable too -> stop, buy nothing.

Not a backtest: no queue model, no fill probability, no competition for the
level, no inventory limit. Every number is optimistic by construction.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import Config

logger = logging.getLogger(__name__)

#: Columns the analysis cannot run without.
REQUIRED = ("segment_id", "session_date", "midprice", "spread",
            "bid_price_1", "ask_price_1")

#: Net signed aggressor volume, in order of preference. Only the Databento
#: loader makes the vendor field, so requiring it would restrict the screen to
#: that path — including the fixtures that prove the screen is correct.
SIGNED_FLOW_CANDIDATES = ("signed_trade_volume", "signed_volume_increment")


def _signed_flow(df: pd.DataFrame) -> pd.Series:
    """Net signed aggressor volume per bar, from whichever source exists."""
    for c in SIGNED_FLOW_CANDIDATES:
        if c in df.columns:
            # Present-but-all-zero is a legitimate answer ("no trades in this
            # sample"), not a reason to fall through to a worse source. Only
            # absence justifies looking further.
            logger.info("Passive study: aggressor flow from '%s'", c)
            return pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    if "trade_direction" in df.columns and "trade_size" in df.columns:
        logger.info("Passive study: aggressor flow from trade_direction x "
                    "trade_size")
        return (pd.to_numeric(df["trade_direction"], errors="coerce").fillna(0)
                * pd.to_numeric(df["trade_size"], errors="coerce").fillna(0.0))
    raise ValueError(
        "passive study needs a signed aggressor-flow column; looked for "
        f"{SIGNED_FLOW_CANDIDATES} and (trade_direction, trade_size)")


def build_passive_fills(feat: pd.DataFrame, config: Config,
                        ofi_col: str = "OFI_L1_ref") -> pd.DataFrame:
    """Identify front-of-queue passive fills and the state known before them.

    A net SELL aggressor hits the resting bid (maker filled LONG); a net BUY
    lifts the resting ask (maker filled SHORT).

    Every input comes from the PREVIOUS row, for two reasons: the fill price
    must be the quote standing before the trade, not one that already moved in
    response to it; and the conditioning OFI must predate the fill, or the
    trade is used to predict itself.
    """
    missing = [c for c in REQUIRED if c not in feat.columns]
    if missing:
        raise ValueError(f"passive study needs columns {missing}")
    if ofi_col not in feat.columns:
        raise ValueError(f"conditioning column '{ofi_col}' absent")

    df = feat.copy()
    g = df.groupby("segment_id", sort=False)
    # Everything the maker could have known at quoting time.
    df["prev_bid"] = g["bid_price_1"].shift(1)
    df["prev_ask"] = g["ask_price_1"].shift(1)
    df["prev_mid"] = g["midprice"].shift(1)
    df["prev_spread"] = g["spread"].shift(1)
    df["prev_ofi"] = g[ofi_col].shift(1)

    stv = _signed_flow(df)
    # side = +1 when WE end up long (a sell aggressor hit our bid), -1 when we
    # end up short (a buy aggressor lifted our ask).
    df["side"] = -np.sign(stv)
    df["fill_price"] = np.where(df["side"] > 0, df["prev_bid"], df["prev_ask"])
    df["aggressor_size"] = stv.abs()

    # OFI signed by the position the fill leaves us in. Adverse selection is
    # symmetric -- negative OFI leaves us long into falling prices, positive
    # leaves us short into rising ones -- so markout against RAW OFI is U-shaped
    # and its monotonicity reads ~0 even when OFI predicts toxicity perfectly.
    # Multiplying by the fill side folds the U into a line.
    df["fill_aligned_ofi"] = df["side"] * df["prev_ofi"]
    df["abs_ofi"] = df["prev_ofi"].abs()

    ok = (
        (df["side"] != 0)
        & df["prev_bid"].notna() & df["prev_ask"].notna()
        & df["prev_mid"].notna() & df["prev_ofi"].notna()
        & (df["prev_ask"] >= df["prev_bid"])
        & np.isfinite(df["fill_price"])
    )
    if "in_continuous_session" in df.columns:
        ok &= df["in_continuous_session"].astype(bool)
    if "integrity_break" in df.columns:
        ok &= ~df["integrity_break"].astype(bool)

    fills = df.loc[ok].copy()
    logger.info("Passive study: %d inferred front-of-queue fills from %d rows "
                "(%.2f%%)", len(fills), len(df),
                100.0 * len(fills) / max(len(df), 1))
    return fills


def markout_columns(fills: pd.DataFrame, config: Config,
                    horizon_tags: List[str]) -> pd.DataFrame:
    """Attach per-horizon markout in price units per share.

    ``markout = side * (mid(t+h) - fill_price) + rebate``, which starts near the
    half-spread captured and is eaten by adverse selection as h grows. Already
    NET of the spread earned: this is the maker's edge, not a gross return.

    ``*_cross_out`` is the same after paying a half-spread to flatten. The
    honest result is between the two; reporting only markout would assume free
    liquidation.
    """
    out = fills
    rebate = float(config.costs.maker_rebate_per_unit)
    side = out["side"].to_numpy(dtype="float64")
    fill_px = out["fill_price"].to_numpy(dtype="float64")
    mid_now = out["midprice"].to_numpy(dtype="float64")
    half_spread_out = out["spread"].to_numpy(dtype="float64") / 2.0

    for tag in horizon_tags:
        col = f"future_mid_change_{tag}"
        if col not in out.columns:
            continue
        fut_mid = mid_now + out[col].to_numpy(dtype="float64")
        mk = side * (fut_mid - fill_px) + rebate
        out[f"markout_{tag}"] = mk
        out[f"markout_{tag}_cross_out"] = mk - half_spread_out
    return out


def _day_clustered(x: pd.Series, days: pd.Series) -> Dict[str, float]:
    """Mean with a day-clustered SE.

    Markouts overlap heavily between adjacent fills, so an iid SE over hundreds
    of thousands of them would be absurdly tight.
    """
    d = pd.DataFrame({"x": x.to_numpy(), "day": days.to_numpy()}).dropna()
    if d.empty:
        return {"mean": np.nan, "se": np.nan, "n": 0, "n_days": 0}
    per_day = d.groupby("day")["x"].mean()
    n_days = int(per_day.shape[0])
    se = (float(per_day.std(ddof=1)) / np.sqrt(n_days)) if n_days > 1 else np.nan
    return {"mean": float(d["x"].mean()), "se": se, "n": int(len(d)),
            "n_days": n_days}


def unconditional_markout(fills: pd.DataFrame, config: Config,
                          horizon_tags: List[str]) -> pd.DataFrame:
    """Is passive quoting profitable AT ALL before any OFI conditioning?

    If the unconditional front-of-queue maker already loses at every horizon,
    OFI must turn a losing business profitable by selective withdrawal — a far
    stronger claim than "OFI predicts adverse selection".
    """
    tick = config.costs.tick_size
    rows = []
    for tag in horizon_tags:
        col = f"markout_{tag}"
        if col not in fills.columns:
            continue
        for variant, c in ((("markout"), col),
                           ("markout_after_crossing_out", f"{col}_cross_out")):
            st = _day_clustered(fills[c], fills["session_date"])
            rows.append({
                "horizon": tag, "variant": variant,
                "mean_price_units": st["mean"],
                "mean_ticks": st["mean"] / tick if tick > 0 else np.nan,
                "se_day_clustered": st["se"],
                "se_ticks": st["se"] / tick if tick > 0 and st["se"] == st["se"]
                else np.nan,
                "t_stat": (st["mean"] / st["se"]) if st["se"] and
                st["se"] == st["se"] and st["se"] > 0 else np.nan,
                "n_fills": st["n"], "n_days": st["n_days"],
                "pct_positive": float((fills[c] > 0).mean() * 100),
            })
    return pd.DataFrame(rows)


def markout_by_ofi_decile(fills: pd.DataFrame, config: Config,
                          horizon_tags: List[str], n_buckets: int = 10,
                          bucket_on: str = "fill_aligned_ofi") -> pd.DataFrame:
    """THE Phase-0 question: does OFI separate toxic fills from benign ones?

    Default axis ``fill_aligned_ofi`` is expected to be monotone increasing —
    the bottom bucket is "filled hard against the flow". ``prev_ofi`` gives the
    raw view, expected U-shaped, whose flat Spearman is NOT evidence of absence.

    Deciles are cut pooled, so the edges peek at the full period; a positive
    result must be re-established walk-forward before it counts as predictive.
    """
    tick = config.costs.tick_size
    out = fills.copy()
    if bucket_on not in out.columns:
        raise ValueError(f"unknown bucketing column '{bucket_on}'")
    try:
        out["ofi_bucket"] = pd.qcut(out[bucket_on], n_buckets,
                                    labels=False, duplicates="drop")
    except ValueError:
        logger.warning("Passive study: %s too degenerate to bucket", bucket_on)
        return pd.DataFrame()

    rows = []
    for tag in horizon_tags:
        col = f"markout_{tag}"
        if col not in out.columns:
            continue
        for b, g in out.groupby("ofi_bucket"):
            st = _day_clustered(g[col], g["session_date"])
            rows.append({
                "horizon": tag, "bucket_on": bucket_on,
                "ofi_decile": int(b), "n_fills": st["n"],
                "mean_bucket_value": float(g[bucket_on].mean()),
                "mean_prev_ofi": float(g["prev_ofi"].mean()),
                "frac_passive_buys": float((g["side"] > 0).mean()),
                "markout_ticks": st["mean"] / tick if tick > 0 else np.nan,
                "se_ticks": (st["se"] / tick)
                if tick > 0 and st["se"] == st["se"] else np.nan,
                "markout_after_crossing_out_ticks":
                float(g[f"{col}_cross_out"].mean()) / tick
                if tick > 0 else np.nan,
                "pct_positive": float((g[col] > 0).mean() * 100),
            })
    tab = pd.DataFrame(rows)
    if tab.empty:
        return tab

    # Monotonicity across buckets, per horizon: the single number that says
    # whether OFI orders fill quality at all.
    mono = (tab.groupby("horizon")
            .apply(lambda g: g[["ofi_decile", "markout_ticks"]]
                   .corr(method="spearman").iloc[0, 1], include_groups=False)
            .rename("decile_monotonicity_spearman").reset_index())
    return tab.merge(mono, on="horizon", how="left")


def side_breakdown(fills: pd.DataFrame, config: Config,
                   horizon_tags: List[str]) -> pd.DataFrame:
    """Split by passive buy vs passive sell.

    An edge that exists only on one side is much weaker evidence than a
    symmetric one — it may be sample drift rather than a flow response.
    """
    tick = config.costs.tick_size
    rows = []
    for tag in horizon_tags:
        col = f"markout_{tag}"
        if col not in fills.columns:
            continue
        for label, sub in (("passive_buy", fills[fills["side"] > 0]),
                           ("passive_sell", fills[fills["side"] < 0])):
            st = _day_clustered(sub[col], sub["session_date"])
            rows.append({
                "horizon": tag, "fill_side": label, "n_fills": st["n"],
                "markout_ticks": st["mean"] / tick if tick > 0 else np.nan,
                "se_ticks": (st["se"] / tick)
                if tick > 0 and st["se"] == st["se"] else np.nan,
            })
    return pd.DataFrame(rows)


def run_phase0(feat: pd.DataFrame, config: Config, horizon_tags: List[str],
               ofi_col: str = "OFI_L1_ref",
               save: Optional[callable] = None) -> Dict[str, pd.DataFrame]:
    """Run the whole Phase-0 screen and return its tables."""
    fills = build_passive_fills(feat, config, ofi_col=ofi_col)
    if fills.empty:
        logger.warning("Passive study: no inferred fills; skipping")
        return {}
    fills = markout_columns(fills, config, horizon_tags)
    tables = {
        "20_passive_markout_unconditional":
            unconditional_markout(fills, config, horizon_tags),
        # Primary: OFI aligned to the fill side, where the expected shape is
        # monotone and the go/no-go is read.
        "21_passive_markout_by_ofi_decile":
            markout_by_ofi_decile(fills, config, horizon_tags,
                                  bucket_on="fill_aligned_ofi"),
        # Secondary: raw OFI, expected U-shaped. Kept because its near-zero
        # monotonicity is a diagnostic of symmetry, not of a missing signal.
        "21b_passive_markout_by_raw_ofi_decile":
            markout_by_ofi_decile(fills, config, horizon_tags,
                                  bucket_on="prev_ofi"),
        # Toxicity should rise with imbalance MAGNITUDE regardless of side.
        "21c_passive_markout_by_abs_ofi_decile":
            markout_by_ofi_decile(fills, config, horizon_tags,
                                  bucket_on="abs_ofi"),
        "22_passive_markout_by_side":
            side_breakdown(fills, config, horizon_tags),
    }
    if save is not None:
        for name, tab in tables.items():
            if tab is not None and len(tab):
                save(tab, config, name)
    return tables


def phase0_report_section(tables: Dict[str, pd.DataFrame],
                          config: Config, primary: str) -> str:
    """Render the screen as a go / no-go paragraph, not a table dump."""
    uncond = tables.get("20_passive_markout_unconditional")
    dec = tables.get("21_passive_markout_by_ofi_decile")
    if uncond is None or uncond.empty:
        return ""
    tick = config.costs.tick_size
    rebate_note = (
        f"Maker rebate applied: {config.costs.maker_rebate_per_unit:.5f} price "
        f"units ({config.costs.maker_rebate_per_unit / tick:.3f} ticks) per "
        "share."
        if config.costs.maker_rebate_per_unit > 0 else
        "**Maker rebate is 0** — set `costs.maker_rebate_per_unit` from the "
        "venue schedule; at a $0.01 tick a typical add rebate is worth 0.2-0.3 "
        "ticks per share, which is large relative to the edges below.")

    lines = ["\n## Phase 0 — front-of-queue passive markout (UPPER BOUND)\n",
             "\nEvery number here assumes our order is at the FRONT of the "
             "queue and fills on every trade at our price. Real queue position "
             "is strictly worse, so these are ceilings. There is no queue "
             "model, no fill probability and no inventory limit: this is a "
             "screen to decide whether market-by-order data is worth buying, "
             "not a backtest.\n",
             f"\n{rebate_note}\n"]

    prim = uncond[(uncond["horizon"] == primary)
                  & (uncond["variant"] == "markout")]
    lines.append("\n### Unconditional (no OFI conditioning)\n")
    lines.append("| Horizon | Markout (ticks) | day-clustered SE | t | fills | "
                 "days |\n|---|---|---|---|---|---|\n")
    for _, r in uncond[uncond["variant"] == "markout"].iterrows():
        lines.append(f"| {r['horizon']} | {r['mean_ticks']:.4f} | "
                     f"{r['se_ticks']:.4f} | {r['t_stat']:.2f} | "
                     f"{int(r['n_fills'])} | {int(r['n_days'])} |\n")

    if len(prim):
        v = float(prim["mean_ticks"].iloc[0])
        lines.append(
            f"\nAt the primary horizon `{primary}` the unconditional "
            f"front-of-queue maker earns {v:.4f} ticks per fill "
            + ("— positive, so the spread captured exceeds adverse selection "
               "even before OFI is used to be selective.\n"
               if v > 0 else
               "— NEGATIVE. Adverse selection already exceeds the spread "
               "captured at the best possible queue position, so OFI would "
               "have to rescue a business that loses money by default, not "
               "merely improve a profitable one.\n"))

    if dec is not None and not dec.empty:
        d = dec[dec["horizon"] == primary].sort_values("ofi_decile")
        if len(d):
            m = float(d["decile_monotonicity_spearman"].iloc[0])
            lo = float(d["markout_ticks"].iloc[0])
            hi = float(d["markout_ticks"].iloc[-1])
            lines.append("\n### Conditioned on fill-aligned OFI, known BEFORE "
                         "the fill\n")
            lines.append(
                "\nBuckets are cut on OFI signed by the position the fill "
                "leaves us in, because adverse selection is symmetric: raw OFI "
                "produces a U (both tails toxic) and its monotonicity would "
                "read ~0 even if OFI predicted toxicity perfectly. See "
                "`21b_` for that raw view and `21c_` for |OFI|.\n\n")
            lines.append("| Decile (worst->best flow) | markout (ticks) | SE | "
                         "fills |\n|---|---|---|---|\n")
            for _, r in d.iterrows():
                lines.append(f"| {int(r['ofi_decile'])} | "
                             f"{r['markout_ticks']:.4f} | "
                             f"{r['se_ticks']:.4f} | {int(r['n_fills'])} |\n")
            spread_ticks = abs(hi - lo)
            lines.append(
                f"\nDecile monotonicity (Spearman) = {m:.3f}; bottom-to-top "
                f"decile spread = {spread_ticks:.4f} ticks. "
                + ("OFI separates toxic from benign fills, which is the "
                   "result Phase 1 would be built on.\n" if abs(m) > 0.5
                   and spread_ticks > 0.01 else
                   "OFI does NOT meaningfully order fill quality here. On this "
                   "evidence the passive hypothesis fails its cheapest test "
                   "and market-by-order data should not be purchased yet.\n"))
            lines.append(
                "\nDecile edges are cut pooled over the whole sample, so they "
                "peek at the full period. A positive reading is a reason to "
                "run this walk-forward, not a result on its own.\n")
    return "".join(lines)
