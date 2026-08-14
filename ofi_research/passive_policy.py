"""Walk-forward passive quoting: queue position, latency, and a fitted gate.

Phase 0 (:mod:`ofi_research.passive`) is a ceiling, and it says two things at
once on INTC: the unconditional front-of-queue maker LOSES 0.054 ticks per fill
at ms1000, while fills sorted by fill-aligned OFI run from -0.150 to +0.089.
So the money is not in quoting, it is in NOT quoting into toxic flow. That
makes the filter the strategy, and a filter has to be tested the way a strategy
is tested — fitted on train days, applied to days it has never seen.

Four things here are strictly harder than the screen, and every one of them can
only move the answer down:

1. **Queue position is a dial.** The screen assumes we are first in line and
   fill on every trade at our price. Here an order joins behind
   ``alpha * displayed_depth`` shares and fills only once trades have eaten
   through them. ``alpha=0`` reproduces the screen exactly, which is what makes
   the two comparable; ``alpha=1`` is joining behind every displayed share.
   MBP-10 shows aggregate size and never our own place in it, so this is swept
   and reported as a bracket. Nothing in the data can pin it down.

2. **Cancels do not help us.** When displayed size falls without a trade,
   :attr:`PassiveConfig.cancels_leave_from_behind` assumes the departing shares
   were behind us, so our queue position never improves for free. With
   aggregate depth we cannot tell who left; assuming it was the people in front
   is how a maker backtest invents an edge.

3. **The gate is causal per side.** The screen buckets on OFI signed by the
   side the fill LEFT us in — known only once the fill happened. A quote does
   not have that luxury: a resting bid can only ever produce a long fill, so
   the bid is quoted when the model predicts UP and the ask when it predicts
   DOWN, each decided before either is hit.

4. **Placing and cancelling both take time.** A quote decided at t reaches the
   book at t + latency and joins the queue as it stands THEN; a withdrawal
   decided at t stops protecting us until t + latency, and trades in between
   still fill us.

Fills are simulated at event resolution from :mod:`ofi_research.passive_tape`,
because at a 200 ms grid two thirds of buckets trade more than the whole
displayed depth and every queue position would fill at once — see that module.
The GATE stays on the decision clock: deciding whether to quote every 200 ms is
a real strategy, and pretending to re-decide every event would be a different
and much stronger claim.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import Config
from .models import LinearModel, model_feature_sets
from .passive_tape import read_tape
from .splits import make_folds

logger = logging.getLogger(__name__)

#: Quoting a resting bid can only ever fill us LONG, and a resting ask SHORT.
#: The whole point of per-side gating is that this is known before the quote.
SIDES = ((1, "passive_buy"), (-1, "passive_sell"))


# --------------------------------------------------------------------------
# Queue simulation
# --------------------------------------------------------------------------
def _episode_bounds(px: np.ndarray, seg: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Index ranges over which our resting order survives untouched.

    A change in the best price on our side ends the episode: if the price moved
    away we are no longer at the touch, and if it moved through us the level we
    joined is gone. Either way the next quote joins a NEW queue at the back,
    which is the pessimistic and correct reading — an order does not keep its
    place across a price change.
    """
    n = len(px)
    if n == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    new_ep = np.ones(n, dtype=bool)
    new_ep[1:] = (px[1:] != px[:-1]) | (seg[1:] != seg[:-1])
    starts = np.flatnonzero(new_ep).astype(np.int64)
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:] - 1
    ends[-1] = n - 1
    return starts, ends


def _first_at_or_after(sorted_idx: np.ndarray, query: np.ndarray,
                       n: int) -> np.ndarray:
    """For each query index, the first member of ``sorted_idx`` that is >= it.

    Returns ``n`` where none exists, so callers can reject with a single
    comparison instead of masking twice.
    """
    if sorted_idx.size == 0:
        return np.full(query.shape, n, dtype=np.int64)
    pos = np.searchsorted(sorted_idx, query, side="left")
    out = np.where(pos < sorted_idx.size, sorted_idx[np.minimum(
        pos, sorted_idx.size - 1)], n)
    return out.astype(np.int64)


def precompute_side(px: np.ndarray, sz: np.ndarray, vol: np.ndarray,
                    seg: np.ndarray, cancels_leave_from_behind: bool = True
                    ) -> Dict[str, np.ndarray]:
    """The parts of the simulation that depend on the book, not the policy.

    Episode boundaries and cumulative consumption are functions of the tape
    alone, so recomputing them for each (gate, queue) pair would repeat an
    O(n) pass over millions of events a hundred times per fold for nothing.
    """
    starts, ends = _episode_bounds(px, seg)
    consumed = vol.astype(np.float64)
    if not cancels_leave_from_behind:
        n = len(px)
        d_sz = np.zeros(n, dtype=np.float64)
        d_sz[1:] = sz[:-1].astype(np.float64) - sz[1:].astype(np.float64)
        cancels = np.maximum(d_sz - vol, 0.0)
        new_ep = np.concatenate(([True], (px[1:] != px[:-1])
                                 | (seg[1:] != seg[:-1])))
        cancels[np.flatnonzero(new_ep)] = 0.0
        consumed = consumed + cancels
    return {"starts": starts, "ends": ends,
            "cum": np.concatenate(([0.0], np.cumsum(consumed)))}


def simulate_side_fills(ts: np.ndarray, px: np.ndarray, sz: np.ndarray,
                        vol: np.ndarray, seg: np.ndarray, gate: np.ndarray,
                        alpha: float, place_latency_ns: int,
                        cancel_latency_ns: int,
                        cancels_leave_from_behind: bool = True,
                        pre: Optional[Dict[str, np.ndarray]] = None
                        ) -> Dict[str, np.ndarray]:
    """Simulate one resting order per episode on one side of the book.

    ``vol`` is aggressor volume hitting OUR side, already non-negative. The
    order fills when volume traded since we joined strictly exceeds the shares
    ahead of us, which at ``alpha=0`` degenerates to "the next trade fills us"
    — the front-of-queue assumption, reproduced rather than special-cased.

    Returns arrays of equal length over filled episodes: the row index of the
    fill, the price we rested at, and the index we joined the queue at.
    """
    n = len(ts)
    if pre is None:
        pre = precompute_side(px, sz, vol, seg, cancels_leave_from_behind)
    starts, ends = pre["starts"], pre["ends"]
    if starts.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return {"fill_idx": empty, "join_idx": empty,
                "fill_price": np.empty(0, dtype=np.float64)}

    idx_on = np.flatnonzero(gate).astype(np.int64)
    idx_off = np.flatnonzero(~gate).astype(np.int64)

    # 1. The quote is decided at the first gated instant inside the episode.
    decide = _first_at_or_after(idx_on, starts, n)
    alive = decide <= ends

    # 2. It reaches the book one latency later, and joins the queue as the
    #    book stands at THAT moment, not as it stood when we decided.
    d = np.where(alive, decide, 0)
    join = np.searchsorted(ts, ts[d] + place_latency_ns, side="left")
    join = np.minimum(join, n - 1).astype(np.int64)
    alive &= (join <= ends) & (join >= d)

    # 3. Shares ahead of us at the moment we arrive.
    j = np.where(alive, join, 0)
    queue_ahead = alpha * sz[j].astype(np.float64)

    # 4. Consumption: trades always eat the queue, and cancellations only do
    #    when we assume the departing shares were in FRONT of us — see
    #    precompute_side, where that choice is made once per tape.
    cum = pre["cum"]                                     # cum[i] = sum(:i)
    # Fill at the first row whose cumulative consumption since joining exceeds
    # the queue ahead. cum is non-decreasing, so this is one binary search.
    target = cum[j] + queue_ahead
    fill = np.searchsorted(cum[1:], target, side="right").astype(np.int64)
    alive &= (fill < n)
    fill = np.where(alive, np.minimum(fill, n - 1), 0)
    alive &= (fill <= ends) & (fill >= j)

    # 5. A withdrawal decided inside the episode stops protecting us only after
    #    the cancel round trip; trades before that still fill us.
    off = _first_at_or_after(idx_off, j, n)
    has_off = off <= ends
    deadline = np.where(has_off, ts[np.minimum(off, n - 1)] + cancel_latency_ns,
                        np.iinfo(np.int64).max)
    alive &= ts[fill] <= deadline

    keep = np.flatnonzero(alive)
    return {"fill_idx": fill[keep], "join_idx": j[keep],
            "fill_price": px[j[keep]].astype(np.float64)}


def _markout(tape: pd.DataFrame, fill_idx: np.ndarray, fill_px: np.ndarray,
             side: int, horizon_ms: float) -> Dict[str, np.ndarray]:
    """Signed mid move from the fill price, ``horizon_ms`` after each fill.

    Fills whose horizon runs past the end of their segment are dropped rather
    than truncated: a markout measured over a shorter window than advertised
    would flatter every number, because adverse selection accumulates.
    """
    ts = tape["timestamp"].to_numpy()
    mid = tape["midprice"].to_numpy()
    seg = tape["segment_id"].to_numpy()
    spread = (tape["ask_price_1"].to_numpy() - tape["bid_price_1"].to_numpy())

    target = ts[fill_idx] + int(horizon_ms * 1_000_000)
    k = np.searchsorted(ts, target, side="left")
    ok = (k < len(ts))
    k = np.minimum(k, len(ts) - 1)
    ok &= (seg[k] == seg[fill_idx])

    mk = side * (mid[k] - fill_px)
    return {"markout": mk[ok], "cross_out": (mk - spread[k] / 2.0)[ok],
            "fill_idx": fill_idx[ok], "ok": ok}


# --------------------------------------------------------------------------
# Walk-forward policy
# --------------------------------------------------------------------------
def _gate_on_tape(tape_ts: np.ndarray, dec_ts: np.ndarray,
                  dec_gate: np.ndarray) -> np.ndarray:
    """Hold each decision until the next one — a step function, not a peek.

    Tape events before the first decision of the day are ungated (we are not
    quoting yet), which is why the search is right-sided and index 0 means
    "no decision has happened".
    """
    if dec_ts.size == 0:
        return np.zeros(len(tape_ts), dtype=bool)
    pos = np.searchsorted(dec_ts, tape_ts, side="right")
    out = np.zeros(len(tape_ts), dtype=bool)
    live = pos > 0
    out[live] = dec_gate[pos[live] - 1]
    return out


def _day_clustered(values: np.ndarray, days: np.ndarray) -> Dict[str, float]:
    """Mean with a day-clustered SE; markouts inside a day are not independent."""
    if values.size == 0:
        return {"mean": np.nan, "se": np.nan, "n": 0, "n_days": 0}
    df = pd.DataFrame({"x": values, "day": days})
    per_day = df.groupby("day")["x"].mean()
    nd = int(per_day.shape[0])
    se = float(per_day.std(ddof=1)) / np.sqrt(nd) if nd > 1 else np.nan
    return {"mean": float(df["x"].mean()), "se": se, "n": int(values.size),
            "n_days": nd}


def run_passive_walk_forward(feat: pd.DataFrame, config: Config,
                             horizon_tag: str,
                             model_name: Optional[str] = None
                             ) -> Dict[str, pd.DataFrame]:
    """Fit the gate on train days, quote on held-out days, sweep the unknowns.

    The sweep is over queue position and gate selectivity. The rebate sweep is
    applied afterwards in closed form: a rebate is a constant paid on every
    fill, so it shifts each markout by the same amount and cannot change which
    orders filled. Re-simulating per rebate would burn hours to reproduce
    addition.
    """
    pcfg = config.passive
    model_name = model_name or config.evaluation.reference_model
    horizon_ms = float(horizon_tag[2:]) if horizon_tag.startswith("ms") else None
    if horizon_ms is None:
        logger.warning("Passive policy needs a clock horizon; got %s",
                       horizon_tag)
        return {}

    target_col = f"future_mid_change_{horizon_tag}"
    fsets = model_feature_sets(config)
    if model_name not in fsets:
        logger.warning("Passive policy: unknown model %s", model_name)
        return {}
    feats = [f for f in fsets[model_name] if f in feat.columns]
    if not feats or target_col not in feat.columns:
        logger.warning("Passive policy: model %s unavailable in this frame",
                       model_name)
        return {}

    place_ns = int(config.costs.total_latency_ms() * 1_000_000)
    cancel_ms = (pcfg.cancel_latency_ms if pcfg.cancel_latency_ms is not None
                 else config.costs.total_latency_ms())
    cancel_ns = int(cancel_ms * 1_000_000)

    folds = make_folds(feat, config)
    rows: List[dict] = []
    per_day: List[dict] = []

    for fold in folds:
        tr = feat[fold.train_mask]
        te = feat[fold.test_mask]
        if tr.empty or te.empty:
            continue
        day = str(te["session_date"].iloc[0])
        tape = read_tape(config, day)
        if tape is None or tape.empty:
            logger.warning("Passive policy: no tape for %s; fold skipped", day)
            continue

        model = LinearModel(model_name, feats)
        try:
            model.fit(tr, target_col)
        except Exception as exc:                       # pragma: no cover
            logger.warning("Passive policy fold %d: fit failed: %s",
                           fold.index, exc)
            continue
        pred_tr = model.predict(tr)
        pred_te = model.predict(te)

        dec_ts = pd.to_datetime(te["timestamp"]).astype("int64").to_numpy()
        order = np.argsort(dec_ts, kind="stable")
        dec_ts = dec_ts[order]
        pred_te_sorted = np.asarray(pred_te)[order]

        ts = tape["timestamp"].to_numpy()
        seg = tape["segment_id"].to_numpy()
        svi = tape["signed_volume_increment"].to_numpy(dtype="float64")

        for side, side_name in SIDES:
            # A resting bid fills us long, so it is quoted when the model
            # predicts UP; the ask is its mirror. Both decisions predate the
            # fill they might receive.
            aligned_tr = side * np.asarray(pred_tr)
            aligned_te = side * pred_te_sorted
            px = (tape["bid_price_1"] if side > 0
                  else tape["ask_price_1"]).to_numpy(dtype="float64")
            sz = (tape["bid_size_1"] if side > 0
                  else tape["ask_size_1"]).to_numpy(dtype="float64")
            # Sell aggressors (negative signed volume) hit the bid; buy
            # aggressors lift the ask.
            vol = np.maximum(-svi, 0.0) if side > 0 else np.maximum(svi, 0.0)
            pre = precompute_side(px, sz, vol, seg,
                                  pcfg.cancels_leave_from_behind)

            for gq in pcfg.gate_quantiles:
                thr = (-np.inf if gq <= 0
                       else float(np.quantile(aligned_tr, gq)))
                gate = _gate_on_tape(ts, dec_ts, aligned_te >= thr)
                if not gate.any():
                    continue
                for alpha in pcfg.queue_ahead_fractions:
                    sim = simulate_side_fills(
                        ts, px, sz, vol, seg, gate, float(alpha),
                        place_ns, cancel_ns, pcfg.cancels_leave_from_behind,
                        pre=pre)
                    if sim["fill_idx"].size == 0:
                        continue
                    mk = _markout(tape, sim["fill_idx"], sim["fill_price"],
                                  side, horizon_ms)
                    if mk["markout"].size == 0:
                        continue
                    per_day.append({
                        "fold": fold.index, "session_date": day,
                        "side": side_name, "gate_quantile": float(gq),
                        "queue_ahead_fraction": float(alpha),
                        "n_fills": int(mk["markout"].size),
                        "gate_threshold": thr,
                        "mean_markout": float(mk["markout"].mean()),
                        "mean_cross_out": float(mk["cross_out"].mean()),
                        "sum_markout": float(mk["markout"].sum()),
                    })

    if not per_day:
        logger.warning("Passive policy: no fills in any fold")
        return {}

    daily = pd.DataFrame(per_day)
    tick = config.costs.tick_size

    # Sides are pooled by fill count: a maker runs both quotes, and the
    # business is the book, not the better half of it.
    for (gq, alpha), g in daily.groupby(["gate_quantile",
                                         "queue_ahead_fraction"]):
        w = g["n_fills"].to_numpy(dtype="float64")
        day_means = (g.groupby("session_date")
                     .apply(lambda x: np.average(x["mean_markout"],
                                                 weights=x["n_fills"])
                            if x["n_fills"].sum() else np.nan,
                            include_groups=False))
        nd = int(day_means.notna().sum())
        se = (float(day_means.std(ddof=1)) / np.sqrt(nd)) if nd > 1 else np.nan
        mean_mk = float(np.average(g["mean_markout"], weights=w)) if w.sum() \
            else np.nan
        cross = float(np.average(g["mean_cross_out"], weights=w)) if w.sum() \
            else np.nan
        n_fills = int(g["n_fills"].sum())
        for rebate in config.passive.rebate_sweep:
            m = mean_mk + float(rebate)
            rows.append({
                "horizon": horizon_tag, "model": model_name,
                "gate_quantile": float(gq),
                "queue_ahead_fraction": float(alpha),
                "maker_rebate_per_unit": float(rebate),
                "n_fills": n_fills, "n_days": nd,
                "fills_per_day": n_fills / max(nd, 1),
                "markout_ticks": m / tick if tick > 0 else np.nan,
                "markout_after_crossing_out_ticks":
                    (cross + float(rebate)) / tick if tick > 0 else np.nan,
                "se_ticks": (se / tick) if tick > 0 and se == se else np.nan,
                "t_stat": (m / se) if se == se and se and se > 0 else np.nan,
                "below_min_fills": n_fills < config.passive.min_fills_per_cell,
                # The rebate that would put this cell exactly at break-even:
                # the cleanest way to state what the venue schedule has to be
                # worth for the business to exist at all.
                "breakeven_rebate_per_unit": -mean_mk,
                "breakeven_rebate_ticks": (-mean_mk / tick) if tick > 0
                else np.nan,
            })

    grid = pd.DataFrame(rows)
    return {"23_passive_policy_grid": grid,
            "24_passive_policy_by_day": daily}


def policy_report_section(tables: Dict[str, pd.DataFrame],
                          config: Config) -> str:
    """Render the sweep as the two questions a maker actually has to answer.

    Those are: where does queue position stop being survivable, and how much
    of the answer is the venue paying us rather than the signal working. A
    grid of 100 cells does not say either out loud, so both are stated in
    prose and the grid is left for the reader who wants to check.
    """
    grid = tables.get("23_passive_policy_grid")
    if grid is None or grid.empty:
        return ""
    tick = config.costs.tick_size
    lat = config.costs.total_latency_ms()
    cancel = (config.passive.cancel_latency_ms
              if config.passive.cancel_latency_ms is not None else lat)
    zero = grid[grid["maker_rebate_per_unit"] == 0.0]

    out = ["\n## Passive policy — walk-forward, queued, latency-charged\n",
           "\nThe gate threshold is fitted on TRAIN days and applied to days "
           "the model has not seen. A resting bid is quoted when the model "
           "predicts UP and a resting ask when it predicts DOWN, so the side "
           "is chosen before the fill rather than read off it — which is the "
           "one thing the Phase-0 decile table above cannot claim.\n",
           f"\nQuotes reach the book {lat:.2f} ms after the decision and join "
           f"the queue as it stands then; cancels take {cancel:.2f} ms to bite "
           "and every trade inside that window still fills us. Fills are "
           "simulated per event, not on the decision clock.\n"]

    # --- queue position ---
    out.append("\n### How far back in the queue does the edge survive?\n")
    out.append("\nZero rebate, best gate per queue position.\n\n")
    out.append("| queue ahead | best gate | markout (ticks) | t | fills/day | "
               "days |\n|---|---|---|---|---|---|\n")
    for alpha, g in zero.groupby("queue_ahead_fraction"):
        g = g[~g["below_min_fills"]]
        if g.empty:
            out.append(f"| {alpha:.2f} × depth | — | too few fills | | | |\n")
            continue
        b = g.loc[g["markout_ticks"].idxmax()]
        out.append(f"| {alpha:.2f} × depth | q={b['gate_quantile']:.2f} | "
                   f"{b['markout_ticks']:.4f} | "
                   f"{b['t_stat']:.2f} | {b['fills_per_day']:.0f} | "
                   f"{int(b['n_days'])} |\n")

    # --- the gate itself ---
    ung = zero[(zero["gate_quantile"] == 0.0)
               & (zero["queue_ahead_fraction"] == 0.0)]
    gat = zero[(zero["gate_quantile"] > 0.0)
               & (zero["queue_ahead_fraction"] == 0.0)
               & (~zero["below_min_fills"])]
    if len(ung) and len(gat):
        u = float(ung["markout_ticks"].iloc[0])
        b = gat.loc[gat["markout_ticks"].idxmax()]
        delta = float(b["markout_ticks"]) - u
        out.append(
            f"\nAt the front of the queue, quoting indiscriminately earns "
            f"{u:.4f} ticks per fill. The best fitted gate "
            f"(q={b['gate_quantile']:.2f}) earns {b['markout_ticks']:.4f}, a "
            f"difference of {delta:+.4f} ticks — "
            + ("which is the filter doing its job: the edge is in declining "
               "to quote, exactly as Phase 0 suggested.\n" if delta > 0 else
               "so the filter does NOT pay for itself out of sample. The "
               "pooled decile ordering did not survive being fitted on one "
               "set of days and applied to another.\n"))

    # --- rebate ---
    out.append("\n### How much of this is the rebate?\n")
    best_zero = zero[~zero["below_min_fills"]]
    if len(best_zero):
        b = best_zero.loc[best_zero["markout_ticks"].idxmax()]
        need = float(b["breakeven_rebate_ticks"])
        out.append(
            f"\nThe best cell before any rebate is {b['markout_ticks']:.4f} "
            f"ticks per fill (queue {b['queue_ahead_fraction']:.2f}×, "
            f"gate q={b['gate_quantile']:.2f}). Break-even needs "
            f"{need:+.3f} ticks of rebate ({need * tick:+.5f} per share). "
            + ("The business therefore does not depend on the schedule.\n"
               if need <= 0 else
               "A typical US add tier is 0.20-0.30 ticks, so this sits "
               + ("INSIDE what a venue plausibly pays — meaning the strategy "
                  "is a rebate-capture business whose signal contributes at "
                  "the margin, and the real schedule decides it.\n"
                  if need <= 0.30 else
                  "BEYOND what any venue pays to add. No rebate rescues "
                  "it.\n")))
        out.append("\n| rebate (ticks) | markout (ticks) | after crossing out "
                   "|\n|---|---|---|\n")
        sel = grid[(grid["queue_ahead_fraction"] == b["queue_ahead_fraction"])
                   & (grid["gate_quantile"] == b["gate_quantile"])]
        for _, r in sel.sort_values("maker_rebate_per_unit").iterrows():
            out.append(
                f"| {r['maker_rebate_per_unit'] / tick:.2f} | "
                f"{r['markout_ticks']:.4f} | "
                f"{r['markout_after_crossing_out_ticks']:.4f} |\n")
        out.append("\nThese rebate tiers are ILLUSTRATIVE, not a schedule. "
                   "Replace `passive.rebate_sweep` with the venue's real "
                   "numbers before any of this is quoted to anyone.\n")

    thin = grid[grid["below_min_fills"]]
    if len(thin):
        out.append(f"\n{len(thin)} of {len(grid)} cells fell below "
                   f"{config.passive.min_fills_per_cell} fills and are "
                   "excluded from every 'best' above — a selective gate at a "
                   "deep queue position can fill so rarely that its mean is "
                   "one day of noise wearing a decimal point.\n")
    return "".join(out)


def policy_gate(grid: pd.DataFrame, config: Config,
                daily: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Decision gate for the PASSIVE path, replacing the taker one.

    The taker gate asked whether a strategy that executed one trade in twenty
    days was profitable. These ask the questions a maker actually faces: does
    the filter beat quoting indiscriminately, does the edge survive being at
    the back of the queue, and how much of the venue's rebate schedule is
    load-bearing.
    """
    if grid is None or grid.empty:
        return pd.DataFrame()
    tick = config.costs.tick_size
    zero_reb = grid[grid["maker_rebate_per_unit"] == 0.0]
    ungated = zero_reb[zero_reb["gate_quantile"] == 0.0]
    gated = zero_reb[(zero_reb["gate_quantile"] > 0.0)
                     & (~zero_reb["below_min_fills"])]

    def _best(df):
        return df.loc[df["markout_ticks"].idxmax()] if len(df) else None

    checks = []
    b = _best(gated)
    u = _best(ungated)
    if b is not None and u is not None:
        checks.append({
            "criterion": "filter_beats_unconditional", "status":
            "PASS" if b["markout_ticks"] > u["markout_ticks"] else "FAIL",
            "basis": f"best gated {b['markout_ticks']:.4f} ticks vs "
                     f"unconditional {u['markout_ticks']:.4f} at the same "
                     f"queue position sweep, zero rebate"})
    if b is not None:
        checks.append({
            "criterion": "positive_at_zero_rebate",
            "status": "PASS" if b["markout_ticks"] > 0 else "FAIL",
            "basis": f"best gated cell {b['markout_ticks']:.4f} ticks/fill "
                     f"before any rebate"})
        checks.append({
            "criterion": "significant_day_clustered",
            "status": "PASS" if (b["t_stat"] == b["t_stat"]
                                 and abs(b["t_stat"]) >= 2.0) else "FAIL",
            "basis": f"t = {b['t_stat']:.2f} over {int(b['n_days'])} days, "
                     "day-clustered"})
    back = zero_reb[(zero_reb["queue_ahead_fraction"] >= 1.0)
                    & (zero_reb["gate_quantile"] > 0.0)
                    & (~zero_reb["below_min_fills"])]
    bb = _best(back)
    if bb is not None:
        checks.append({
            "criterion": "survives_back_of_queue",
            "status": "PASS" if bb["markout_ticks"] > 0 else "FAIL",
            "basis": f"behind all displayed depth: {bb['markout_ticks']:.4f} "
                     f"ticks/fill on {int(bb['n_fills'])} fills"})
    if b is not None:
        need = float(b["breakeven_rebate_ticks"])
        checks.append({
            "criterion": "rebate_not_load_bearing",
            "status": "PASS" if need <= 0 else "FAIL",
            "basis": (f"break-even needs {need:.3f} ticks of rebate "
                      f"({need * tick:.5f} price units per share); a typical "
                      "US add tier is 0.20-0.30 ticks")})
    return pd.DataFrame(checks)
