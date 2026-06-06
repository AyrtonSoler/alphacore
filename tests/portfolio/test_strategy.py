"""RED tests for portfolio.strategy — the definitive strategy's alpha + construction.

Team decision: Quality + Momentum sleeves, **market-wide** z-score (NOT sector-neutral),
residualized against the factor design (§5.4), quarterly rebalance, gamma 0, no bands,
min-15. This module pins those choices in a typed ``StrategyParams`` and builds the
date x ticker weight panel via the tested optimizer.

Acceptance
----------
- ``StrategyParams`` defaults encode the team decision (Q+M, market_wide, gamma 0,
  quarterly cadence).
- ``select_rebalance_dates`` subsamples month-ends to the configured cadence.
- The weight panel is indexed by the rebalancing dates, long-only, within the cap, and
  sums to ``1 - cash`` each date.
- The ``zscore_grouping`` knob actually changes the portfolio (market-wide != sector).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.allocation import AllocatorParams
from portfolio.strategy import StrategyParams, build_strategy_weights, select_rebalance_dates

_TICKERS = [f"T{i:02d}" for i in range(12)]
_DATES = pd.to_datetime(["2020-01-31", "2020-02-29"])
_SIGNAL_COLS = ["quality_a", "quality_b", "momentum_a"]
_SLEEVES = {"quality": ["quality_a", "quality_b"], "momentum": ["momentum_a"]}
_FACTORS = ["beta", "size"]
_CASH = 0.01


def _build() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(42)
    values = pd.DataFrame(rng.normal(size=(12, 3)), index=_TICKERS, columns=_SIGNAL_COLS)
    vols = pd.Series(rng.uniform(0.1, 0.6, 12), index=_TICKERS)
    frames = {
        t: pd.DataFrame(
            {c: [values.loc[t, c]] * len(_DATES) for c in _SIGNAL_COLS}
            | {"risk_realized_vol": [vols[t]] * len(_DATES)},
            index=_DATES,
        )
        for t in _TICKERS
    }
    sectors = pd.Series(["A", "B", "C"] * 4, index=_TICKERS)
    factor_matrices = {
        f: pd.DataFrame(rng.normal(size=(2, 12)), index=_DATES, columns=_TICKERS) for f in _FACTORS
    }
    benchmark = pd.DataFrame(np.full((2, 12), 1.0 / 12), index=_DATES, columns=_TICKERS)
    return frames, factor_matrices, sectors, benchmark


def test_default_params_encode_team_decision() -> None:
    """StrategyParams defaults pin Q+M, market-wide, gamma 0, quarterly."""
    params = StrategyParams()
    assert set(params.sleeves) == {"quality", "momentum"}
    assert params.zscore_grouping == "market_wide"
    assert params.gamma_risk == 0.0
    assert params.cadence_months == 3
    assert params.min_holdings == 15
    assert params.cap_per_name == 0.125
    assert params.allocator is not None
    assert params.allocator.lambda_max == 0.30


def test_select_rebalance_dates_quarterly() -> None:
    """Quarterly cadence keeps every third month-end."""
    month_ends = pd.date_range("2020-01-31", periods=12, freq="ME").tolist()
    selected = select_rebalance_dates(month_ends, StrategyParams())
    assert selected == month_ends[::3]
    assert len(selected) == 4


def test_panel_is_dates_by_held_names() -> None:
    """The weight panel is indexed by the rebalancing dates with name columns."""
    frames, factor_matrices, sectors, benchmark = _build()
    params = StrategyParams(sleeves=_SLEEVES, min_holdings=3)
    panel = build_strategy_weights(frames, factor_matrices, sectors, benchmark, _DATES, params)
    assert list(panel.index) == list(_DATES)
    assert set(panel.columns) <= set(_TICKERS)


def test_delisted_name_is_not_held_after_its_data_ends() -> None:
    """Asset eligibility: a name whose price series ends before a rebalance date carries
    zero weight there (no stale ``asof`` carry-forward into a date with no execution price)."""
    frames, factor_matrices, sectors, benchmark = _build()
    # T00 is delisted after the first date: its frame only reaches _DATES[0].
    frames["T00"] = frames["T00"].loc[[_DATES[0]]]
    params = StrategyParams(sleeves=_SLEEVES, min_holdings=3, cash=_CASH)
    panel = build_strategy_weights(frames, factor_matrices, sectors, benchmark, _DATES, params)
    assert float(panel.loc[_DATES[1]].get("T00", 0.0)) == 0.0


def test_each_date_is_long_only_capped_and_budgeted() -> None:
    """Per date: weights in [0, cap] summing to 1 - cash."""
    frames, factor_matrices, sectors, benchmark = _build()
    params = StrategyParams(sleeves=_SLEEVES, min_holdings=3, cash=_CASH)
    panel = build_strategy_weights(frames, factor_matrices, sectors, benchmark, _DATES, params)
    for date in _DATES:
        row = panel.loc[date]
        assert (row >= -1e-6).all()
        assert (row <= params.cap_per_name + 1e-6).all()
        assert np.isclose(row.sum(), 1.0 - _CASH, atol=1e-4)


def test_ml_allocator_retilts_held_weights_without_changing_selection() -> None:
    """The ML overlay re-sizes the held names but never changes which names are held."""
    frames, factor_matrices, sectors, benchmark = _build()
    rng = np.random.default_rng(7)
    predictions = pd.DataFrame(
        rng.normal(size=(len(_DATES), len(_TICKERS))), index=_DATES, columns=_TICKERS
    )
    lam = pd.Series(0.3, index=_DATES)

    base = build_strategy_weights(
        frames,
        factor_matrices,
        sectors,
        benchmark,
        _DATES,
        StrategyParams(sleeves=_SLEEVES, min_holdings=3),
    )
    tilted = build_strategy_weights(
        frames,
        factor_matrices,
        sectors,
        benchmark,
        _DATES,
        StrategyParams(sleeves=_SLEEVES, min_holdings=3, allocator=AllocatorParams()),
        ml_predictions=predictions,
        ml_lambda=lam,
    )

    # Same names held each date (selection is Q+M's, untouched by the overlay).
    for date in _DATES:
        assert set(base.loc[date][base.loc[date] > 1e-6].index) == set(
            tilted.loc[date][tilted.loc[date] > 1e-6].index
        )
    # But the money split differs, and stays long-only / capped / budgeted.
    assert not np.allclose(base.to_numpy(), tilted.reindex_like(base).to_numpy())
    for date in _DATES:
        row = tilted.loc[date]
        assert (row >= -1e-6).all()
        assert (row <= StrategyParams().cap_per_name + 1e-6).all()
        assert np.isclose(row.sum(), 1.0 - StrategyParams().cash, atol=1e-4)


def test_tilt_needs_predictions_and_spread() -> None:
    """The overlay is on by default, but is a no-op without predictions and flat with no spread."""
    frames, factor_matrices, sectors, benchmark = _build()
    assert StrategyParams().allocator is not None  # on by default

    # No predictions supplied => the overlay does nothing (identical to explicitly off).
    no_preds = build_strategy_weights(
        frames,
        factor_matrices,
        sectors,
        benchmark,
        _DATES,
        StrategyParams(sleeves=_SLEEVES, min_holdings=3),
    )
    explicit_off = build_strategy_weights(
        frames,
        factor_matrices,
        sectors,
        benchmark,
        _DATES,
        StrategyParams(sleeves=_SLEEVES, min_holdings=3, allocator=None),
    )
    pd.testing.assert_frame_equal(no_preds, explicit_off)

    # Flat predictions (zero cross-sectional spread) => the tilt collapses to equal weight.
    flat_predictions = pd.DataFrame(1.0, index=_DATES, columns=_TICKERS)
    flat_preds = build_strategy_weights(
        frames,
        factor_matrices,
        sectors,
        benchmark,
        _DATES,
        StrategyParams(sleeves=_SLEEVES, min_holdings=3),
        ml_predictions=flat_predictions,
        ml_lambda=pd.Series(0.3, index=_DATES),
    )
    for date in _DATES:
        row = flat_preds.loc[date]
        held = row[row > 1e-6]
        assert np.allclose(held.to_numpy(), held.mean())  # all held weights equal


def test_zscore_grouping_changes_weights() -> None:
    """Market-wide vs sector-neutral z-score grouping yields a different portfolio."""
    frames, factor_matrices, sectors, benchmark = _build()
    base = StrategyParams(sleeves=_SLEEVES, min_holdings=3)
    mw = build_strategy_weights(
        frames,
        factor_matrices,
        sectors,
        benchmark,
        _DATES,
        StrategyParams(sleeves=_SLEEVES, min_holdings=3, zscore_grouping="market_wide"),
    )
    sn = build_strategy_weights(
        frames,
        factor_matrices,
        sectors,
        benchmark,
        _DATES,
        StrategyParams(sleeves=_SLEEVES, min_holdings=3, zscore_grouping="sector"),
    )
    assert base.zscore_grouping == "market_wide"
    aligned = sn.reindex(columns=mw.columns).fillna(0.0)
    assert not np.allclose(mw.to_numpy(), aligned.to_numpy())
