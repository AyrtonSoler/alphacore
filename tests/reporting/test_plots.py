"""RED tests for reporting.plots — backtest visualisation.

Every backtest run is plotted (cumulative growth vs the KN600 plus the drawdown)
so results can be eyeballed and compared. The plot is a deterministic side effect:
the test checks a non-empty PNG is written.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from alphacore.portfolio.backtest import BacktestResult
from alphacore.reporting.plots import plot_backtest


def test_plot_backtest_writes_a_png(tmp_path: Path) -> None:
    """plot_backtest saves a non-empty PNG of the strategy vs benchmark."""
    dates = pd.date_range("2020-01-31", periods=24, freq="ME")
    rng = np.random.default_rng(0)
    result = BacktestResult(
        portfolio_returns=pd.Series(rng.normal(0.012, 0.04, 24), index=dates),
        benchmark_returns=pd.Series(rng.normal(0.009, 0.03, 24), index=dates),
        turnover=pd.Series(0.5, index=dates),
    )
    out = tmp_path / "backtest.png"
    plot_backtest(result, out, title="Test run")
    assert out.exists()
    assert out.stat().st_size > 0
