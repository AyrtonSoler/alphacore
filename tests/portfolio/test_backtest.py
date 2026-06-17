"""RED tests for portfolio.backtest — research backtest of a weight panel.

A fast research backtest (the KaxaNuk engine is the official evaluator): given the
rebalance weight panel, the forward return of each name over each holding period, and
the benchmark's period returns, it computes the net portfolio return series and the
headline metrics (return, vol, Sharpe, and the relative-mandate active return,
tracking error and information ratio).

Acceptance
----------
- The gross period return is the weighted sum of forward returns.
- A turnover cost (bps of traded weight) is deducted; the first period's turnover is
  the initial build from cash.
- The summary reports Sharpe and information ratio by the standard formulas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphacore.portfolio.backtest import run_backtest

_DATES = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"])
_NAMES = ["AAA", "BBB"]


def _frame(rows: list[list[float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, index=_DATES, columns=_NAMES)


def test_gross_return_is_weighted_forward_return() -> None:
    """Each period's gross return is sum_i w_i * forward_return_i."""
    weights = _frame([[0.6, 0.4], [0.5, 0.5], [0.7, 0.3]])
    fwd = _frame([[0.10, -0.05], [0.02, 0.02], [-0.03, 0.04]])
    result = run_backtest(weights, fwd, pd.Series(0.0, index=_DATES), cost_bps=0.0)
    assert np.isclose(result.portfolio_returns.iloc[0], 0.6 * 0.10 + 0.4 * -0.05)
    assert np.isclose(result.portfolio_returns.iloc[1], 0.5 * 0.02 + 0.5 * 0.02)


def test_turnover_cost_is_deducted() -> None:
    """Net return subtracts turnover (traded weight) times the per-unit bps cost."""
    weights = _frame([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    fwd = _frame([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    result = run_backtest(weights, fwd, pd.Series(0.0, index=_DATES), cost_bps=100.0)
    # Period 0: initial build turnover = |1| + |0| = 1.0 -> cost 100bps = 0.01.
    assert np.isclose(result.turnover.iloc[0], 1.0)
    assert np.isclose(result.portfolio_returns.iloc[0], -0.01)
    # Period 1: rotate fully A->B, turnover = |0-1| + |1-0| = 2.0 -> cost 0.02.
    assert np.isclose(result.turnover.iloc[1], 2.0)
    assert np.isclose(result.portfolio_returns.iloc[1], -0.02)


def test_summary_reports_sharpe_and_information_ratio() -> None:
    """Sharpe and IR follow the standard annualized formulas."""
    weights = _frame([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    # Forward returns chosen so portfolio returns vary (non-zero volatility).
    fwd = _frame([[0.03, 0.01], [0.04, 0.02], [0.015, 0.005]])
    bench = pd.Series([0.01, 0.015, 0.005], index=_DATES)
    result = run_backtest(weights, fwd, bench, cost_bps=0.0)
    summary = result.summary(periods_per_year=12)

    port = result.portfolio_returns
    expected_sharpe = port.mean() * 12 / (port.std(ddof=1) * np.sqrt(12))
    assert np.isclose(summary["sharpe"], expected_sharpe)

    active = port - bench
    expected_ir = active.mean() * 12 / (active.std(ddof=1) * np.sqrt(12))
    assert np.isclose(summary["information_ratio"], expected_ir)
    assert "annualized_return" in summary
    assert "max_drawdown" in summary


def test_cumulative_compounds_returns() -> None:
    """The cumulative curve compounds the net period returns."""
    weights = _frame([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    fwd = _frame([[0.10, 0.0], [0.10, 0.0], [0.10, 0.0]])
    result = run_backtest(weights, fwd, pd.Series(0.0, index=_DATES), cost_bps=0.0)
    assert np.isclose(result.cumulative().iloc[-1], 1.10**3 - 1.0)
