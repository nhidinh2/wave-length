"""Transaction-cost arithmetic tests (section 18 / 28)."""

from ofi_research.config import CostConfig
from ofi_research.costs import compute_trade_pnl, estimate_round_trip_cost


def test_aggressive_long_full_spread():
    # entry ask=101, bid=100; exit bid=102, ask=103; size=1; no fees/slip
    cfg = CostConfig(tick_size=1.0)
    r = compute_trade_pnl(1, 1.0, entry_bid=100, entry_ask=101,
                          exit_bid=102, exit_ask=103, cfg=cfg)
    # buy at 101, sell at 102 => executable 1.0
    assert r.entry_price == 101
    assert r.exit_price == 102
    assert r.executable_gross == 1.0
    # mid-to-mid: entry mid 100.5, exit mid 102.5 => 2.0 gross theoretical
    assert r.gross_mid_pnl == 2.0
    # spread cost = full spread crossing = 2.0 - 1.0 = 1.0
    assert r.spread_cost == 1.0
    assert r.net_pnl == 1.0


def test_aggressive_short_full_spread():
    cfg = CostConfig(tick_size=1.0)
    # short: sell at bid=100, later buy at ask=99 (price fell) => profit 1
    r = compute_trade_pnl(-1, 1.0, entry_bid=100, entry_ask=101,
                          exit_bid=98, exit_ask=99, cfg=cfg)
    assert r.entry_price == 100  # sell at bid
    assert r.exit_price == 99    # buy back at ask
    assert r.executable_gross == 1.0
    # mid entry 100.5, exit 98.5 => short gross mid = +2.0
    assert r.gross_mid_pnl == 2.0
    assert r.net_pnl == 1.0


def test_slippage_and_fees_reduce_pnl():
    cfg = CostConfig(tick_size=1.0, slippage_ticks=1.0, fee_per_unit=0.5)
    r = compute_trade_pnl(1, 2.0, entry_bid=100, entry_ask=101,
                          exit_bid=102, exit_ask=103, cfg=cfg)
    # entry 101+1=102, exit 102-1=101 => executable per unit = -1 * size 2 = -2
    assert r.entry_price == 102
    assert r.exit_price == 101
    assert r.executable_gross == -2.0
    # fees = 0.5*2 (entry) + 0.5*2 (exit) = 2.0
    assert r.fees == 2.0
    assert r.slippage_cost == 4.0  # 2 * 1 tick * size 2
    assert r.net_pnl == -4.0  # -2 executable - 2 fees


def test_round_trip_cost_threshold():
    cfg = CostConfig(tick_size=1.0, slippage_ticks=1.0, fee_bps=0.0,
                     minimum_edge_buffer=0.5)
    # spread=2, slippage=2*1=2, buffer 0.5 => 4.5
    c = estimate_round_trip_cost(mid=100.0, spread=2.0, size=1.0, cfg=cfg)
    assert c == 4.5
