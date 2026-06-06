"""Research backtest of a weight panel — fast metrics, not the official engine.

The KaxaNuk backtest engine is the official evaluator; this is a lightweight research
backtest for fast iteration. Given the rebalance weight panel, each name's forward
return over its holding period, and the benchmark's period returns, it computes the
net portfolio return series (after a turnover cost) and the headline metrics for both
the absolute and the relative (vs KN600) mandate.

References
----------
.. [Sharpe1994] Sharpe, W. F. (1994). "The Sharpe Ratio." Journal of Portfolio
   Management 21(1), 49-58.
.. [Grinold1989] Grinold, R. C. (1989). "The Fundamental Law of Active Management."
   Journal of Portfolio Management 15(3), 30-37. (Information ratio.)

"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

_BPS = 1e4
_MONTHS_PER_YEAR = 12


@dataclasses.dataclass(frozen=True)
class BacktestResult:
    """Outcome of a research backtest.

    Attributes
    ----------
    portfolio_returns:
        Net portfolio return per rebalancing period (after turnover cost).
    benchmark_returns:
        Benchmark return over the same periods.
    turnover:
        Traded weight per period (the first period is the build from cash).

    """

    portfolio_returns: pd.Series[float]
    benchmark_returns: pd.Series[float]
    turnover: pd.Series[float]

    def cumulative(self) -> pd.Series[float]:
        """Compounded growth of the net portfolio returns (starting at 0)."""
        return (1.0 + self.portfolio_returns).cumprod() - 1.0

    def summary(self, periods_per_year: int = _MONTHS_PER_YEAR) -> dict[str, float]:
        """Headline absolute and relative-mandate metrics.

        Returns
        -------
        dict
            ``total_return``, ``annualized_return``, ``annualized_volatility``,
            ``sharpe``, ``active_return``, ``tracking_error``, ``information_ratio``,
            ``max_drawdown`` and ``average_turnover``.

        """
        # Work in numpy so the reductions are cleanly-typed np.float64, not the
        # complex-inclusive unions the pandas stubs assign to Series reductions.
        port = self.portfolio_returns.to_numpy()
        active = (self.portfolio_returns - self.benchmark_returns).to_numpy()
        turnover = self.turnover.to_numpy()
        ann = float(periods_per_year)
        root = float(np.sqrt(ann))
        n = len(port)

        vol = float(np.std(port, ddof=1)) * root
        te = float(np.std(active, ddof=1)) * root
        growth = float(np.prod(1.0 + port))
        cumulative = np.cumprod(1.0 + port)
        drawdown = cumulative / np.maximum.accumulate(cumulative) - 1.0

        return {
            "total_return": growth - 1.0,
            "annualized_return": growth ** (ann / n) - 1.0,
            "annualized_volatility": vol,
            "sharpe": float(np.mean(port)) * ann / vol if vol > 0 else 0.0,
            "active_return": float(np.mean(active)) * ann,
            "tracking_error": te,
            "information_ratio": float(np.mean(active)) * ann / te if te > 0 else 0.0,
            "max_drawdown": float(np.min(drawdown)),
            "average_turnover": float(np.mean(turnover)),
        }


def run_backtest(
    weights: pd.DataFrame,
    forward_returns: pd.DataFrame,
    benchmark_returns: pd.Series[float],
    *,
    cost_bps: float = 0.0,
) -> BacktestResult:
    """Backtest a weight panel against per-name forward returns.

    Parameters
    ----------
    weights:
        Rebalance weight panel (date x ticker).
    forward_returns:
        Each name's return over the period following the rebalancing date
        (date x ticker), aligned to ``weights``. Missing names contribute 0.
    benchmark_returns:
        Benchmark return over each period, indexed by rebalancing date.
    cost_bps:
        Turnover cost in basis points of traded weight (round-trip). The first
        period's turnover is the initial build from cash.

    Returns
    -------
    BacktestResult
        Net portfolio returns, the aligned benchmark returns, and turnover.

    References
    ----------
    Sharpe ratio from [Sharpe1994]_; information ratio from [Grinold1989]_.

    """
    aligned = forward_returns.reindex(index=weights.index, columns=weights.columns).fillna(0.0)
    gross = (weights * aligned).sum(axis=1)

    turnover = weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = weights.iloc[0].abs().sum()  # build from cash
    cost = turnover * cost_bps / _BPS
    net = gross - cost

    return BacktestResult(
        portfolio_returns=net,
        benchmark_returns=benchmark_returns.reindex(net.index),
        turnover=turnover,
    )
