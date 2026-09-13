"""The P&L criteria must not be scored off a ledger too thin to score.

Regression cover for the 20-day M5 run, which executed a single trade, won
4 ticks on it, and reported "net-positive out of sample" beside a
concentration FAIL stating that 100% of P&L came from one day. Both sentences
described the same trade. The three-valued gate already existed to stop an
EMPTY ledger reading as a failure; these tests extend the same reasoning to a
ledger that is technically non-empty and just as uninformative.
"""

from ofi_research.config import Config
from ofi_research.run_experiment import pnl_gate_evaluable


def _cfg(min_trades=30, min_days=2):
    c = Config()
    c.evaluation.min_trades_for_pnl_gate = min_trades
    c.evaluation.min_trade_days_for_pnl_gate = min_days
    return c


def test_single_trade_is_not_evaluable():
    # The exact shape of the M5 run that produced the bad headline.
    assert not pnl_gate_evaluable(_cfg(), n_trades=1, n_trade_days=1)


def test_empty_ledger_is_not_evaluable():
    assert not pnl_gate_evaluable(_cfg(), n_trades=0, n_trade_days=0)


def test_enough_trades_but_all_on_one_day_is_not_evaluable():
    # Overlapping targets make same-day rows dependent, so every interval in
    # the report is day-blocked; 500 trades on one day is one observation.
    assert not pnl_gate_evaluable(_cfg(), n_trades=500, n_trade_days=1)


def test_enough_days_but_too_few_trades_is_not_evaluable():
    assert not pnl_gate_evaluable(_cfg(), n_trades=3, n_trade_days=3)


def test_thresholds_are_inclusive():
    assert pnl_gate_evaluable(_cfg(), n_trades=30, n_trade_days=2)


def test_comfortably_past_both_thresholds_is_evaluable():
    assert pnl_gate_evaluable(_cfg(), n_trades=250, n_trade_days=8)


def test_zero_minimum_restores_score_any_nonempty_ledger():
    # Documented escape hatch: min 0 reproduces the pre-fix behaviour.
    cfg = _cfg(min_trades=0, min_days=0)
    assert pnl_gate_evaluable(cfg, n_trades=1, n_trade_days=1)
    assert pnl_gate_evaluable(cfg, n_trades=0, n_trade_days=0)


def test_none_counts_are_treated_as_zero():
    # _sample_accounting can hand back None when the ledger column is absent.
    assert not pnl_gate_evaluable(_cfg(), n_trades=None, n_trade_days=None)
