"""Backtest visualisation — cumulative growth vs the KN600, plus drawdown.

Every backtest run is plotted so results can be eyeballed and compared across runs.
Uses the headless Agg backend so it works inside Docker without a display.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow backend selection)

from alphacore.portfolio.backtest import BacktestResult  # noqa: E402

_HEIGHT_RATIOS = [3, 1]


def plot_backtest(
    result: BacktestResult,
    path: Path | str,
    *,
    title: str = "v3 backtest",
) -> None:
    """Write a PNG of cumulative strategy vs benchmark growth and the drawdown.

    Parameters
    ----------
    result:
        The backtest outcome to visualise.
    path:
        Destination PNG path (parent directories are created).
    title:
        Figure title (e.g. the run configuration).

    """
    portfolio_growth = (1.0 + result.portfolio_returns).cumprod()
    benchmark_growth = (1.0 + result.benchmark_returns).cumprod()
    drawdown = portfolio_growth / portfolio_growth.cummax() - 1.0

    fig, (ax_growth, ax_drawdown) = plt.subplots(
        2, 1, figsize=(11, 7), height_ratios=_HEIGHT_RATIOS, sharex=True
    )

    ax_growth.plot(portfolio_growth.index, portfolio_growth.to_numpy(), label="Strategy", lw=1.8)
    ax_growth.plot(
        benchmark_growth.index, benchmark_growth.to_numpy(), label="KN600", lw=1.4, alpha=0.8
    )
    ax_growth.set_yscale("log")
    ax_growth.set_ylabel("Growth of 1 (log)")
    ax_growth.set_title(title)
    ax_growth.legend(loc="upper left")
    ax_growth.grid(visible=True, alpha=0.3)

    ax_drawdown.fill_between(drawdown.index, drawdown.to_numpy(), 0.0, color="tab:red", alpha=0.4)
    ax_drawdown.set_ylabel("Drawdown")
    ax_drawdown.grid(visible=True, alpha=0.3)

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destination, dpi=120)
    plt.close(fig)
