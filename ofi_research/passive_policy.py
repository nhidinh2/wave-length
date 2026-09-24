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

    Every tape row carries the book AFTER its event, so the volume printed on
    row ``i`` traded against the book standing at row ``i - 1``. On INTC
    2026-07-29, 65% of sell prints arrive on a row whose bid has already
    dropped — the print cleared the level. Charging that volume to row ``i``
    credits the fill to the NEXT level, a tick better than it printed, and
    loses it outright when the price change ends the episode first: exactly
    the most toxic fills vanish. So consumption is indexed by the book it hit:
    ``cum`` advances at row ``r`` by what printed on row ``r + 1``.
    """
    n = len(px)
    starts, ends = _episode_bounds(px, seg)
    consumed = vol.astype(np.float64)
    if not cancels_leave_from_behind:
        d_sz = np.zeros(n, dtype=np.float64)
        d_sz[1:] = sz[:-1].astype(np.float64) - sz[1:].astype(np.float64)
        cancels = np.maximum(d_sz - vol, 0.0)
        new_ep = np.concatenate(([True], (px[1:] != px[:-1])
                                 | (seg[1:] != seg[:-1])))
        cancels[np.flatnonzero(new_ep)] = 0.0
        consumed = consumed + cancels
    # Re-index onto the pre-event book; never across a segment boundary.
    on_book = np.zeros(n, dtype=np.float64)
    if n > 1:
        on_book[:-1] = np.where(seg[1:] == seg[:-1], consumed[1:], 0.0)
    return {"starts": starts, "ends": ends,
            "cum": np.concatenate(([0.0], np.cumsum(on_book)))}


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
    cum = pre["cum"]          # cum[r] = volume that hit books 0..r-1
    # The book row whose consumption first exceeds the queue ahead is the one
    # we were resting in when filled; the print itself is the NEXT row, and
    # that is the fill's time. cum is non-decreasing, so one binary search.
    target = cum[j] + queue_ahead
    rest = np.searchsorted(cum[1:], target, side="right").astype(np.int64)
    alive &= (rest < n - 1)
    rest = np.where(alive, rest, 0)
    alive &= (rest <= ends) & (rest >= j)
    fill = rest + 1

    # 5. A withdrawal decided inside the episode stops protecting us only after
    #    the cancel round trip; trades before that still fill us.
    off = _first_at_or_after(idx_off, j, n)
    has_off = off <= ends
    deadline = np.where(has_off, ts[np.minimum(off, n - 1)] + cancel_latency_ns,
                        np.iinfo(np.int64).max)
    alive &= ts[np.minimum(fill, n - 1)] <= deadline

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


def simulate_unwind(ts: np.ndarray, seg: np.ndarray, fill_idx: np.ndarray,
                    fill_px: np.ndarray, side: int, exit_px: np.ndarray,
                    exit_sz: np.ndarray, cross_px: np.ndarray,
                    exit_pre: Dict[str, np.ndarray], alpha: float,
                    place_latency_ns: int, timeouts_ms: List[float]
                    ) -> Dict[str, np.ndarray]:
    """Flatten each fill with a pegged passive exit, crossing on timeout.

    A long fill rests an ask at the touch (``exit_px`` is the ask series,
    ``exit_pre`` its :func:`precompute_side` over BUY aggressor volume) and a
    short fill rests a bid. The exit joins behind ``alpha`` of displayed depth,
    one latency after the fill, and fills by the same consumption rule as the
    entry. When the touch moves the exit loses its place and re-joins at the
    new touch one latency later — a peg, and pessimistic in the same way the
    entry is. If it has not filled by ``timeout`` the position is crossed at
    ``cross_px`` (the bid for a long) one latency after the deadline.

    The simulation is run once to the longest timeout; a shorter one keeps only
    the passive exits that happened before its own deadline and crosses the
    rest, which is exactly what running it separately would produce.

    Returns, per timeout, the round-trip P&L in price units per share (NaN
    where the exit would leave the segment — dropped, as in ``_markout``) and
    whether the exit was passive, which decides whether it earns a rebate.
    """
    n = len(ts)
    nf = fill_idx.size
    starts, ends = exit_pre["starts"], exit_pre["ends"]
    cum = exit_pre["cum"]
    t_max = int(max(timeouts_ms) * 1_000_000)
    deadline = ts[fill_idx] + t_max
    exit_ts = np.full(nf, np.iinfo(np.int64).max, dtype=np.int64)
    exit_price = np.full(nf, np.nan)

    j = np.searchsorted(ts, ts[fill_idx] + place_latency_ns, side="left")
    active = (j < n)
    active[active] &= seg[j[active]] == seg[fill_idx[active]]
    while active.any():
        a = np.flatnonzero(active)
        ja = j[a]
        ep_end = ends[np.searchsorted(starts, ja, side="right") - 1]
        target = cum[ja] + alpha * exit_sz[ja]
        rest = np.searchsorted(cum[1:], target, side="right")
        hit = (rest <= ep_end) & (rest < n - 1)
        hit_ts = ts[np.minimum(rest + 1, n - 1)]
        hit &= hit_ts <= deadline[a]
        done = a[hit]
        exit_ts[done] = hit_ts[hit]
        exit_price[done] = exit_px[ja[hit]]
        # Not filled in this queue: re-peg at the next touch, if still in time.
        miss = a[~hit]
        nxt = ep_end[~hit] + 1
        ok = nxt < n
        nxt_c = np.minimum(nxt, n - 1)
        ok &= seg[nxt_c] == seg[fill_idx[miss]]
        nj = np.searchsorted(ts, ts[nxt_c] + place_latency_ns, side="left")
        ok &= (nj < n)
        nj_c = np.minimum(nj, n - 1)
        ok &= (ts[nj_c] <= deadline[miss]) & (seg[nj_c] == seg[fill_idx[miss]])
        j[miss] = nj_c
        active[:] = False
        active[miss[ok]] = True

    out: Dict[str, np.ndarray] = {}
    entry_ts = ts[fill_idx]
    for T in timeouts_ms:
        t_ns = int(T * 1_000_000)
        passive = exit_ts <= entry_ts + t_ns
        c = np.searchsorted(ts, entry_ts + t_ns + place_latency_ns,
                            side="left")
        c_ok = c < n
        c = np.minimum(c, n - 1)
        c_ok &= seg[c] == seg[fill_idx]
        px_out = np.where(passive, exit_price, cross_px[c])
        pnl = side * (px_out - fill_px)
        pnl = np.where(passive | c_ok, pnl, np.nan)
        out[f"pnl_{int(T)}"] = pnl
        out[f"passive_{int(T)}"] = passive
    return out


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


def _cell_day_means(g: pd.DataFrame, col: str,
                    weight_col: str = "n_fills") -> pd.Series:
    """One number per day for a cell, sides pooled by their fill counts."""
    def _one(x):
        w = x[weight_col].to_numpy(dtype="float64")
        v = x[col].to_numpy(dtype="float64")
        ok = np.isfinite(v) & (w > 0)
        return np.average(v[ok], weights=w[ok]) if ok.any() else np.nan
    return g.groupby("session_date").apply(_one, include_groups=False)


def _pooled_days(g: pd.DataFrame, col: str,
                 weight_col: str = "n_fills") -> Dict[str, float]:
    """Fill-weighted mean, with a SE clustered on days rather than fills."""
    w = g[weight_col].to_numpy(dtype="float64")
    v = g[col].to_numpy(dtype="float64")
    ok = np.isfinite(v) & (w > 0)
    mean = float(np.average(v[ok], weights=w[ok])) if ok.any() else np.nan
    dm = _cell_day_means(g, col, weight_col).dropna()
    nd = int(dm.size)
    se = float(dm.std(ddof=1)) / np.sqrt(nd) if nd > 1 else np.nan
    t = mean / se if se == se and se > 0 else np.nan
    return {"mean": mean, "se": se, "t": t, "n_days": nd}


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

        # Both books up front: each side's entry is the other side's exit.
        book = {}
        for side, _ in SIDES:
            px = (tape["bid_price_1"] if side > 0
                  else tape["ask_price_1"]).to_numpy(dtype="float64")
            sz = (tape["bid_size_1"] if side > 0
                  else tape["ask_size_1"]).to_numpy(dtype="float64")
            # Sell aggressors (negative signed volume) hit the bid; buy
            # aggressors lift the ask.
            vol = np.maximum(-svi, 0.0) if side > 0 else np.maximum(svi, 0.0)
            book[side] = (px, sz, vol, precompute_side(
                px, sz, vol, seg, pcfg.cancels_leave_from_behind))

        for side, side_name in SIDES:
            # A resting bid fills us long, so it is quoted when the model
            # predicts UP; the ask is its mirror. Both decisions predate the
            # fill they might receive.
            aligned_tr = side * np.asarray(pred_tr)
            aligned_te = side * pred_te_sorted
            px, sz, vol, pre = book[side]
            x_px, x_sz, _, x_pre = book[-side]

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
                    rec = {
                        "fold": fold.index, "session_date": day,
                        "side": side_name, "gate_quantile": float(gq),
                        "queue_ahead_fraction": float(alpha),
                        "n_fills": int(mk["markout"].size),
                        "gate_threshold": thr,
                        "mean_markout": float(mk["markout"].mean()),
                        "mean_cross_out": float(mk["cross_out"].mean()),
                        "sum_markout": float(mk["markout"].sum()),
                    }
                    # The long's exit crosses at the bid, which is its own
                    # quote series; the short's crosses at the ask.
                    unw = simulate_unwind(
                        ts, seg, mk["fill_idx"],
                        sim["fill_price"][mk["ok"]], side, x_px, x_sz, px,
                        x_pre, float(alpha), place_ns,
                        pcfg.unwind_timeouts_ms)
                    for T in pcfg.unwind_timeouts_ms:
                        p = unw[f"pnl_{int(T)}"]
                        ok = np.isfinite(p)
                        rec[f"n_unwind_{int(T)}"] = int(ok.sum())
                        rec[f"mean_unwind_{int(T)}"] = (
                            float(p[ok].mean()) if ok.any() else np.nan)
                        rec[f"frac_passive_exit_{int(T)}"] = (
                            float(unw[f"passive_{int(T)}"][ok].mean())
                            if ok.any() else np.nan)
                    per_day.append(rec)

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
            unwind = {}
            for T in config.passive.unwind_timeouts_ms:
                k = int(T)
                # A passive exit earns the rebate a second time; a crossed
                # one pays nothing extra here (taker fees are not modelled).
                val = (g[f"mean_unwind_{k}"]
                       + float(rebate) * (1.0 + g[f"frac_passive_exit_{k}"]))
                st = _pooled_days(g.assign(_v=val), "_v", f"n_unwind_{k}")
                unwind.update({
                    f"unwind_{k}ms_ticks": st["mean"] / tick,
                    f"unwind_{k}ms_se_ticks": st["se"] / tick,
                    f"unwind_{k}ms_t": st["t"],
                    f"unwind_{k}ms_passive_exit_frac": _pooled_days(
                        g, f"frac_passive_exit_{k}", f"n_unwind_{k}")["mean"],
                })
            rows.append({**unwind,
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
    out.append("\nZero rebate; per queue position, the gate with the largest "
               "day-clustered t (not the largest point estimate, which "
               "favours the noisiest cell).\n\n")
    out.append("| queue ahead | best gate | markout (ticks) | t | fills/day | "
               "days |\n|---|---|---|---|---|---|\n")
    for alpha, g in zero.groupby("queue_ahead_fraction"):
        g = g[~g["below_min_fills"]]
        if g.empty:
            out.append(f"| {alpha:.2f} × depth | — | too few fills | | | |\n")
            continue
        b = g.loc[g["t_stat"].fillna(-np.inf).idxmax()]
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

    # --- exit costs ---
    touts = [int(T) for T in config.passive.unwind_timeouts_ms
             if f"unwind_{int(T)}ms_ticks" in zero.columns]
    usable = zero[~zero["below_min_fills"]]
    if touts and len(usable):
        out.append("\n### What does getting out cost?\n")
        out.append(
            "\nMarkout values the position at mid, as if it could be sold "
            "there for free. Here every fill is flattened by a pegged passive "
            "exit at the opposite touch, crossed if it has not filled by the "
            "timeout. The two bracket columns are the ends this must land "
            "between. Zero rebate; best cell by t at each queue position.\n\n")
        out.append("| queue ahead | timeout | gate | markout | unwound | t | "
                   "passive exits | crossing every exit |\n"
                   "|---|---|---|---|---|---|---|---|\n")
        for alpha, g in usable.groupby("queue_ahead_fraction"):
            for k in touts:
                b = g.loc[g[f"unwind_{k}ms_t"].fillna(-np.inf).idxmax()]
                out.append(
                    f"| {alpha:.2f} × depth | {k} ms | "
                    f"q={b['gate_quantile']:.2f} | {b['markout_ticks']:.4f} | "
                    f"**{b[f'unwind_{k}ms_ticks']:.4f}** | "
                    f"{b[f'unwind_{k}ms_t']:.2f} | "
                    f"{100 * b[f'unwind_{k}ms_passive_exit_frac']:.0f}% | "
                    f"{b['markout_after_crossing_out_ticks']:.4f} |\n")

    # --- rebate ---
    out.append("\n### How much of this is the rebate?\n")
    best_zero = zero[~zero["below_min_fills"]]
    if len(best_zero):
        k = touts[-1] if touts else None
        tcol = f"unwind_{k}ms_t" if k else "t_stat"
        b = best_zero.loc[best_zero[tcol].fillna(-np.inf).idxmax()]
        if k:
            f = float(b[f"unwind_{k}ms_passive_exit_frac"])
            need = -float(b[f"unwind_{k}ms_ticks"]) / (1.0 + f)
            what = (f"round trip at the {k} ms timeout is "
                    f"{b[f'unwind_{k}ms_ticks']:.4f} ticks")
        else:
            need = float(b["breakeven_rebate_ticks"])
            what = f"fill is {b['markout_ticks']:.4f} ticks"
        out.append(
            f"\nThe best cell before any rebate (queue "
            f"{b['queue_ahead_fraction']:.2f}×, gate "
            f"q={b['gate_quantile']:.2f}): its {what}. Break-even needs "
            f"{need:+.3f} ticks of rebate per share "
            f"({need * tick:+.5f}), paid on the entry and on every passive "
            "exit. "
            + ("The business therefore does not depend on the schedule.\n"
               if need <= 0 else
               "A typical US add tier is 0.20-0.30 ticks, so this sits "
               + ("INSIDE what a venue plausibly pays — meaning the strategy "
                  "is a rebate-capture business whose signal contributes at "
                  "the margin, and the real schedule decides it.\n"
                  if need <= 0.30 else
                  "BEYOND what any venue pays to add. No rebate rescues "
                  "it.\n")))
        ucol = f"unwind_{k}ms_ticks" if k else "markout_ticks"
        out.append("\n| rebate (ticks) | markout (ticks) | unwound (ticks) | "
                   "after crossing out |\n|---|---|---|---|\n")
        sel = grid[(grid["queue_ahead_fraction"] == b["queue_ahead_fraction"])
                   & (grid["gate_quantile"] == b["gate_quantile"])]
        for _, r in sel.sort_values("maker_rebate_per_unit").iterrows():
            out.append(
                f"| {r['maker_rebate_per_unit'] / tick:.2f} | "
                f"{r['markout_ticks']:.4f} | {r[ucol]:.4f} | "
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


def _bonferroni_t(m: int, n_days: int, alpha: float) -> float:
    """Two-sided day-clustered t critical value after searching ``m`` cells."""
    from scipy import stats
    return float(stats.t.ppf(1.0 - alpha / (2.0 * max(m, 1)),
                             df=max(n_days - 1, 1)))


def policy_gate(grid: pd.DataFrame, config: Config,
                daily: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Decision gate for the PASSIVE path, replacing the taker one.

    Every criterion searches a family of (gate, queue[, timeout]) cells and
    keeps the one with the largest day-clustered t, so each is tested against
    a Bonferroni critical value over that family rather than against 2.0 —
    the previous gate passed ``survives_back_of_queue`` on a t = 0.75 cell
    picked as the largest of 100 point estimates.

    The verdict rests on ``positive_after_exit_costs``: markout alone assumes
    inventory unwinds free at mid, and every zero-rebate cell of the last run
    lost about a tick once the exit was paid for. The unwind is simulated
    (:func:`simulate_unwind`), and the basis reports it beside both ends of
    the bracket it lands in.
    """
    if grid is None or grid.empty:
        return pd.DataFrame()
    tick = config.costs.tick_size
    fam_alpha = config.passive.gate_family_alpha
    zero_reb = grid[grid["maker_rebate_per_unit"] == 0.0]
    gated = zero_reb[(zero_reb["gate_quantile"] > 0.0)
                     & (~zero_reb["below_min_fills"])]
    nd = int(zero_reb["n_days"].max()) if len(zero_reb) else 0

    def _row(name, ok, basis, evaluable=True):
        return {"criterion": name,
                "status": ("NOT_EVALUABLE" if not evaluable
                           else "PASS" if ok else "FAIL"),
                "basis": basis}

    def _best_t(df, tcol):
        d = df[df[tcol].notna()]
        return d.loc[d[tcol].idxmax()] if len(d) else None

    checks = []

    # 1. The gated markout, before any exit cost.
    b = _best_t(gated, "t_stat")
    crit = _bonferroni_t(len(gated), nd, fam_alpha)
    checks.append(_row(
        "markout_positive_significant",
        b is not None and b["markout_ticks"] > 0 and b["t_stat"] >= crit,
        "no evaluable gated cell" if b is None else
        f"best of {len(gated)} gated cells by t: q={b['gate_quantile']:.2f}, "
        f"queue {b['queue_ahead_fraction']:.2f}x, "
        f"{b['markout_ticks']:.4f} ticks, t = {b['t_stat']:.2f} vs "
        f"Bonferroni {crit:.2f} ({nd} days). Assumes free exit at mid",
        evaluable=b is not None))

    # 2. Does the filter beat quoting always, paired day by day?
    paired = []
    if daily is not None and len(daily):
        for (gq, a), g in daily.groupby(["gate_quantile",
                                         "queue_ahead_fraction"]):
            if gq <= 0:
                continue
            base = daily[(daily["gate_quantile"] == 0.0)
                         & (daily["queue_ahead_fraction"] == a)]
            if base.empty:
                continue
            d = (_cell_day_means(g, "mean_markout")
                 - _cell_day_means(base, "mean_markout")).dropna()
            if d.size < 2:
                continue
            se = float(d.std(ddof=1)) / np.sqrt(d.size)
            paired.append({"gq": gq, "alpha": a, "diff": float(d.mean()),
                           "t": float(d.mean()) / se if se > 0 else np.nan,
                           "n": int(d.size)})
    pdf = pd.DataFrame(paired)
    if len(pdf) and pdf["t"].notna().any():
        bp = pdf.loc[pdf["t"].idxmax()]
        crit_p = _bonferroni_t(len(pdf), int(bp["n"]), fam_alpha)
        checks.append(_row(
            "filter_beats_unconditional", bp["t"] >= crit_p,
            f"best of {len(pdf)} gated-minus-ungated day-paired differences: "
            f"q={bp['gq']:.2f} at queue {bp['alpha']:.2f}x, "
            f"{bp['diff'] / tick:+.4f} ticks, t = {bp['t']:.2f} vs "
            f"Bonferroni {crit_p:.2f}"))
    else:
        checks.append(_row("filter_beats_unconditional", False,
                           "no per-day table to pair against",
                           evaluable=False))

    # 3. Behind all displayed depth.
    back = gated[gated["queue_ahead_fraction"] >= 1.0]
    bb = _best_t(back, "t_stat")
    crit_b = _bonferroni_t(len(back), nd, fam_alpha)
    checks.append(_row(
        "survives_back_of_queue",
        bb is not None and bb["markout_ticks"] > 0 and bb["t_stat"] >= crit_b,
        "no evaluable back-of-queue cell" if bb is None else
        f"best of {len(back)} cells behind all displayed depth: "
        f"q={bb['gate_quantile']:.2f}, {bb['markout_ticks']:.4f} ticks on "
        f"{int(bb['n_fills'])} fills, t = {bb['t_stat']:.2f} vs Bonferroni "
        f"{crit_b:.2f}",
        evaluable=bb is not None))

    # 4. After paying to get out: the criterion the verdict rests on.
    long_rows = []
    for T in config.passive.unwind_timeouts_ms:
        k = int(T)
        col = f"unwind_{k}ms_ticks"
        if col not in gated.columns:
            continue
        for _, r in gated.iterrows():
            long_rows.append({**r.to_dict(), "timeout_ms": k,
                              "unwind": r[col], "unwind_t": r[f"unwind_{k}ms_t"],
                              "pexit": r[f"unwind_{k}ms_passive_exit_frac"]})
    ul = pd.DataFrame(long_rows)
    bu = _best_t(ul, "unwind_t") if len(ul) else None
    crit_u = _bonferroni_t(len(ul), nd, fam_alpha)
    checks.append(_row(
        "positive_after_exit_costs",
        bu is not None and bu["unwind"] > 0 and bu["unwind_t"] >= crit_u,
        "unwind not simulated" if bu is None else
        f"best of {len(ul)} (gate, queue, timeout) cells by t: "
        f"q={bu['gate_quantile']:.2f}, queue {bu['queue_ahead_fraction']:.2f}x, "
        f"{bu['timeout_ms']} ms timeout: {bu['unwind']:.4f} ticks per round "
        f"trip, t = {bu['unwind_t']:.2f} vs Bonferroni {crit_u:.2f}; "
        f"{100 * bu['pexit']:.0f}% of exits passive. Bracket for this cell: "
        f"[{bu['markout_after_crossing_out_ticks']:.4f} crossing every exit, "
        f"{bu['markout_ticks']:.4f} free exit at mid]",
        evaluable=bu is not None))

    # 5. How much of the round trip the venue has to pay for.
    if bu is not None:
        f = float(bu["pexit"]) if bu["pexit"] == bu["pexit"] else 0.0
        need = -float(bu["unwind"]) / (1.0 + f)
        checks.append(_row(
            "rebate_not_load_bearing", need <= 0,
            f"break-even needs {need:+.3f} ticks of rebate per share "
            f"({need * tick:+.5f}) on the best round trip, counting the "
            f"rebate on the entry and on the {100 * f:.0f}% of passive exits; "
            "a typical US add tier is 0.20-0.30 ticks"))
    else:
        checks.append(_row("rebate_not_load_bearing", False,
                           "unwind not simulated", evaluable=False))
    return pd.DataFrame(checks)
