"""RED tests for portfolio.construction — the end-to-end rebalance loop.

Ties the whole strategy together: at each rebalancing date it scores the universe
(:func:`signals.pipeline.assemble_cross_section`), builds the optimizer instance, and
solves (:func:`portfolio.optimizer.optimize_portfolio`), chaining the previous
weights so the turnover penalty has a reference. The output is the date x ticker
weight panel the KaxaNuk backtest engine consumes.

Acceptance
----------
- The panel is indexed by the rebalancing dates with a column per held name.
- Each date's weights are long-only, within the cap, and sum to 1 - cash.
- The resulting portfolio is factor-neutral within the band on every date (the whole
  signal -> score -> optimize chain holds end-to-end).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from common.factor_model import factor_exposures_at
from portfolio.construction import BacktestConfig, BacktestData, construct_portfolios
from portfolio.optimizer import OptimizerParams

_TICKERS = [f"T{i:02d}" for i in range(12)]
_DATES = pd.to_datetime(["2020-01-31", "2020-02-28"])
_SLEEVES = {"quality": ["q1", "q2"], "momentum": ["m1"]}
_SIGNAL_COLS = ["q1", "q2", "m1"]
_FACTORS = ["beta", "size"]
_CASH = 0.01


def _build() -> tuple[dict[str, pd.DataFrame], BacktestData]:
    rng = np.random.default_rng(0)
    # Per-ticker signal frames: constant over the two dates, varying across tickers.
    values = pd.DataFrame(rng.normal(size=(12, 3)), index=_TICKERS, columns=_SIGNAL_COLS)
    vols = pd.Series(rng.uniform(0.1, 0.6, 12), index=_TICKERS)  # per-name realized vol
    frames = {
        t: pd.DataFrame(
            {c: [values.loc[t, c]] * len(_DATES) for c in _SIGNAL_COLS}
            | {"risk_realized_vol": [vols[t]] * len(_DATES)},
            index=_DATES,
        )
        for t in _TICKERS
    }
    sectors = pd.Series(["A", "B", "C"] * 4, index=_TICKERS)
    # Factor matrices: date x ticker, covering both rebalancing dates.
    factor_matrices = {
        f: pd.DataFrame(rng.normal(size=(2, 12)), index=_DATES, columns=_TICKERS) for f in _FACTORS
    }
    benchmark = pd.DataFrame(np.full((2, 12), 1.0 / 12), index=_DATES, columns=_TICKERS)
    data = BacktestData(
        sectors=sectors, factor_matrices=factor_matrices, benchmark_weights=benchmark
    )
    return frames, data


def test_panel_is_dates_by_held_names() -> None:
    """The weight panel is indexed by rebalancing dates with name columns."""
    frames, data = _build()
    panel = construct_portfolios(frames, _DATES, data, BacktestConfig(sleeves=_SLEEVES, cash=_CASH))
    assert list(panel.index) == list(_DATES)
    assert set(panel.columns) <= set(_TICKERS)
    assert len(panel.columns) >= 1


def test_each_date_is_long_only_capped_and_budgeted() -> None:
    """Per date: weights in [0, cap] and summing to 1 - cash."""
    frames, data = _build()
    panel = construct_portfolios(frames, _DATES, data, BacktestConfig(sleeves=_SLEEVES, cash=_CASH))
    for date in _DATES:
        row = panel.loc[date]
        assert (row >= -1e-6).all()
        assert (row <= 0.125 + 1e-6).all()
        assert np.isclose(row.sum(), 1.0 - _CASH, atol=1e-4)


def test_uses_realized_vol_risk_model_from_frames() -> None:
    """The per-date realized-vol risk model changes the portfolio vs a flat risk model."""
    frames, data = _build()
    # Low cardinality floor + meaningful risk aversion so the risk model can act.
    cfg = BacktestConfig(
        sleeves=_SLEEVES, cash=_CASH, params=OptimizerParams(min_holdings=3, gamma_risk=5.0)
    )
    realized = construct_portfolios(frames, _DATES, data, cfg)
    flat = dataclasses.replace(data, volatility=pd.Series(0.3, index=_TICKERS))
    flat_risk = construct_portfolios(frames, _DATES, flat, cfg)
    aligned = flat_risk.reindex(columns=realized.columns).fillna(0.0)
    assert not np.allclose(realized.to_numpy(), aligned.to_numpy())


def test_handles_partial_benchmark_coverage() -> None:
    """A universe holding only a slice of benchmark weight is renormalized, not infeasible."""
    frames, data = _build()
    # Benchmark sums to ~0.1 over the universe (a small replicable subset).
    data = dataclasses.replace(data, benchmark_weights=data.benchmark_weights * 0.1)
    panel = construct_portfolios(frames, _DATES, data, BacktestConfig(sleeves=_SLEEVES, cash=_CASH))
    for date in _DATES:
        assert np.isclose(panel.loc[date].sum(), 1.0 - _CASH, atol=1e-4)


def test_portfolio_is_factor_neutral_each_date() -> None:
    """The end-to-end portfolio stays within the factor-neutrality band every date."""
    frames, data = _build()
    params = OptimizerParams(eps_factor=0.05, gamma_risk=0.1)
    panel = construct_portfolios(
        frames, _DATES, data, BacktestConfig(sleeves=_SLEEVES, cash=_CASH, params=params)
    )
    for date in _DATES:
        weights = panel.loc[date]
        design = factor_exposures_at(data.factor_matrices, date, tickers=list(panel.columns))
        bench = data.benchmark_weights.loc[date].reindex(panel.columns).fillna(0.0)
        active = (weights - bench).to_numpy()
        exposure = design.to_numpy().T @ active
        assert (np.abs(exposure) <= 0.05 + 1e-3).all()
