"""Transaction-cost and execution model (section 18).

Two execution assumptions are supported:

A. **Aggressive market-order execution** (the realistic path). Entry and exit
   prices are computed EXPLICITLY from the full bid/ask, so a round trip pays
   the *entire* spread plus slippage, fees and impact -- never just half a
   spread.

       long : buy at ask (+slippage), later sell at bid (-slippage)
       short: sell at bid (-slippage), later buy at ask (+slippage)

B. **Mid-to-mid** P&L, returned as ``gross_mid_pnl`` and clearly labeled
   *gross theoretical* -- not executable.

``estimate_round_trip_cost`` gives the trade gate its threshold: a trade is
allowed only when ``abs(predicted_move) > round_trip_cost + minimum_edge_buffer``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .config import CostConfig, ExecutionConfig


@dataclass
class TradePnL:
    direction: int
    size: float
    entry_bid: float
    entry_ask: float
    exit_bid: float
    exit_ask: float
    entry_price: float          # executable, incl. slippage
    exit_price: float           # executable, incl. slippage
    entry_mid: float
    exit_mid: float
    gross_mid_pnl: float        # theoretical mid-to-mid (labeled gross)
    executable_gross: float     # bid/ask + slippage, before fees/impact
    spread_cost: float          # gross_mid - executable_gross
    fees: float
    impact: float
    slippage_cost: float
    net_pnl: float
    # Depth-walking diagnostics (P0.9). Defaults keep the single-level
    # ``compute_trade_pnl`` path unchanged.
    requested_size: float = float("nan")
    entry_levels_used: int = 1
    exit_levels_used: int = 1
    depth_exhausted: bool = False
    participation_capped: bool = False


def _fees_one_side(price: float, size: float, cfg: CostConfig) -> float:
    return cfg.fee_per_unit * size + (cfg.fee_bps / 1e4) * price * size


def compute_trade_pnl(direction: int, size: float,
                      entry_bid: float, entry_ask: float,
                      exit_bid: float, exit_ask: float,
                      cfg: CostConfig) -> TradePnL:
    """Full explicit aggressive-execution P&L for a round trip.

    ``direction`` is +1 (long) or -1 (short).
    """
    tick = cfg.tick_size
    slip = cfg.slippage_ticks * tick
    entry_mid = 0.5 * (entry_bid + entry_ask)
    exit_mid = 0.5 * (exit_bid + exit_ask)

    if direction == 1:
        entry_price = entry_ask + slip
        exit_price = exit_bid - slip
        executable_gross = (exit_price - entry_price) * size
    elif direction == -1:
        entry_price = entry_bid - slip
        exit_price = exit_ask + slip
        executable_gross = (entry_price - exit_price) * size
    else:
        raise ValueError("direction must be +1 or -1")

    gross_mid_pnl = direction * (exit_mid - entry_mid) * size
    # slippage cost isolated: what the extra ticks cost on both sides
    slippage_cost = 2.0 * slip * size
    spread_cost = gross_mid_pnl - executable_gross - 0.0  # crossing cost incl. slip
    fees = (_fees_one_side(entry_price, size, cfg)
            + _fees_one_side(exit_price, size, cfg))
    impact = cfg.impact_coefficient * size
    net_pnl = executable_gross - fees - impact

    return TradePnL(
        direction=direction, size=size,
        entry_bid=entry_bid, entry_ask=entry_ask,
        exit_bid=exit_bid, exit_ask=exit_ask,
        entry_price=entry_price, exit_price=exit_price,
        entry_mid=entry_mid, exit_mid=exit_mid,
        gross_mid_pnl=gross_mid_pnl,
        executable_gross=executable_gross,
        spread_cost=spread_cost,
        fees=fees, impact=impact, slippage_cost=slippage_cost,
        net_pnl=net_pnl,
    )


# --- Multi-level taker execution (P0.9) ---
@dataclass
class Fill:
    """Result of walking displayed depth for an aggressive order."""
    requested_size: float
    filled_size: float
    avg_price: float          # size-weighted, before slippage/fees
    levels_used: int
    exhausted: bool           # ran out of displayed depth
    capped_by_participation: bool


def walk_depth(prices: Sequence[float], sizes: Sequence[float], size: float,
               exec_cfg: ExecutionConfig) -> Fill:
    """Consume displayed liquidity level by level, worst price last.

    ``prices``/``sizes`` must already be ordered best-first (ascending asks for
    a buy, descending bids for a sell). Queue position is irrelevant here — a
    taker crosses immediately — but price *walking* is not: an order larger
    than the top level pays progressively worse prices, and pretending
    otherwise flatters every large trade.
    """
    requested = float(size)
    displayed = float(np.nansum([s for s in sizes if s is not None
                                 and not np.isnan(s)]))
    capped = False
    if exec_cfg.max_participation is not None and displayed > 0:
        cap = exec_cfg.max_participation * displayed
        if requested > cap:
            size = cap
            capped = True

    remaining = float(size)
    notional = 0.0
    filled = 0.0
    levels_used = 0
    for px, sz in zip(prices, sizes):
        if remaining <= 0:
            break
        if px is None or sz is None or np.isnan(px) or np.isnan(sz) or sz <= 0:
            continue
        take = min(remaining, float(sz))
        notional += take * float(px)
        filled += take
        remaining -= take
        levels_used += 1

    exhausted = remaining > 1e-12
    avg = notional / filled if filled > 0 else float("nan")
    return Fill(requested_size=requested, filled_size=filled, avg_price=avg,
                levels_used=levels_used, exhausted=exhausted,
                capped_by_participation=capped)


def compute_trade_pnl_walked(direction: int, size: float,
                             entry_bid_px: Sequence[float],
                             entry_bid_sz: Sequence[float],
                             entry_ask_px: Sequence[float],
                             entry_ask_sz: Sequence[float],
                             exit_bid_px: Sequence[float],
                             exit_bid_sz: Sequence[float],
                             exit_ask_px: Sequence[float],
                             exit_ask_sz: Sequence[float],
                             cfg: CostConfig,
                             exec_cfg: ExecutionConfig
                             ) -> Optional[TradePnL]:
    """Round-trip taker P&L using multi-level displayed depth on both sides.

    A long buys the ask ladder on entry and sells the bid ladder on exit; a
    short does the reverse. Returns ``None`` when depth is insufficient and
    partial fills are disallowed, so an untradeable signal never books P&L.
    """
    tick = cfg.tick_size
    slip = cfg.slippage_ticks * tick

    if direction == 1:
        entry_fill = walk_depth(entry_ask_px, entry_ask_sz, size, exec_cfg)
        exit_fill = walk_depth(exit_bid_px, exit_bid_sz, size, exec_cfg)
    elif direction == -1:
        entry_fill = walk_depth(entry_bid_px, entry_bid_sz, size, exec_cfg)
        exit_fill = walk_depth(exit_ask_px, exit_ask_sz, size, exec_cfg)
    else:
        raise ValueError("direction must be +1 or -1")

    # trade the size we can both open AND close
    filled = min(entry_fill.filled_size, exit_fill.filled_size)
    if filled <= 0:
        return None
    if not exec_cfg.allow_partial_fill and filled < size - 1e-12:
        return None
    if np.isnan(entry_fill.avg_price) or np.isnan(exit_fill.avg_price):
        return None

    entry_bid0, entry_ask0 = float(entry_bid_px[0]), float(entry_ask_px[0])
    exit_bid0, exit_ask0 = float(exit_bid_px[0]), float(exit_ask_px[0])
    entry_mid = 0.5 * (entry_bid0 + entry_ask0)
    exit_mid = 0.5 * (exit_bid0 + exit_ask0)

    if direction == 1:
        entry_price = entry_fill.avg_price + slip
        exit_price = exit_fill.avg_price - slip
        executable_gross = (exit_price - entry_price) * filled
    else:
        entry_price = entry_fill.avg_price - slip
        exit_price = exit_fill.avg_price + slip
        executable_gross = (entry_price - exit_price) * filled

    gross_mid_pnl = direction * (exit_mid - entry_mid) * filled
    slippage_cost = 2.0 * slip * filled
    spread_cost = gross_mid_pnl - executable_gross
    fees = (_fees_one_side(entry_price, filled, cfg)
            + _fees_one_side(exit_price, filled, cfg))
    impact = cfg.impact_coefficient * filled
    net_pnl = executable_gross - fees - impact

    return TradePnL(
        direction=direction, size=filled,
        entry_bid=entry_bid0, entry_ask=entry_ask0,
        exit_bid=exit_bid0, exit_ask=exit_ask0,
        entry_price=entry_price, exit_price=exit_price,
        entry_mid=entry_mid, exit_mid=exit_mid,
        gross_mid_pnl=gross_mid_pnl,
        executable_gross=executable_gross,
        spread_cost=spread_cost,
        fees=fees, impact=impact, slippage_cost=slippage_cost,
        net_pnl=net_pnl,
        requested_size=float(size),
        entry_levels_used=entry_fill.levels_used,
        exit_levels_used=exit_fill.levels_used,
        depth_exhausted=bool(entry_fill.exhausted or exit_fill.exhausted),
        participation_capped=bool(entry_fill.capped_by_participation
                                  or exit_fill.capped_by_participation),
    )


def estimate_round_trip_cost(mid: float, spread: float, size: float,
                             cfg: CostConfig) -> float:
    """Per-unit price threshold a predicted move must beat to justify a trade.

    Accounts for the full spread, both-side slippage, both-side fees and impact,
    expressed in price units per unit of size.
    """
    tick = cfg.tick_size
    slippage = 2.0 * cfg.slippage_ticks * tick
    fees_price = 2.0 * (cfg.fee_per_unit + (cfg.fee_bps / 1e4) * mid)
    impact = cfg.impact_coefficient
    return spread + slippage + fees_price + impact + cfg.minimum_edge_buffer
