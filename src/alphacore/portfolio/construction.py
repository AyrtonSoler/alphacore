"""End-to-end portfolio construction — the rebalance loop (v3 §4 pipeline).

Ties the whole strategy together. For each rebalancing date it:

1. scores the universe with :func:`signals.pipeline.assemble_cross_section` (the
   factor-neutral residual-alpha score, §5.3+§5.4);
2. assembles the per-date :class:`portfolio.optimizer.OptimizerInputs` — alpha from
   the score, a diagonal risk model from per-name volatility, the factor design
   (§5.4 regressors), the benchmark weights as-of the date, and the previous
   period's weights (so the turnover penalty has a reference); and
3. solves with :func:`portfolio.optimizer.optimize_portfolio` (§5.8).

The output is the ``date x ticker`` weight panel the KaxaNuk backtest engine
consumes. The risk model defaults to a diagonal of per-name variance (a Ledoit-Wolf
full covariance is a later refinement) and liquidity to "deep" unless an ADV series
is supplied, keeping the loop runnable before those inputs are wired.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
import pandas as pd

from common.factor_model import factor_exposures_at
from signals.pipeline import assemble_cross_section

from .optimizer import OptimizerInputs, OptimizerParams, optimize_portfolio

_DEFAULT_VOLATILITY = 1.0
_DEEP_ADV = 1e15  # effectively unlimited liquidity when no ADV is supplied
_RISK_VOL_COLUMN = "risk_realized_vol"
_VOL_FLOOR = 1e-3  # keep the diagonal risk model strictly positive-definite


@dataclasses.dataclass(frozen=True)
class BacktestData:
    """The cross-date market data the rebalance loop reads.

    Attributes
    ----------
    sectors:
        Ticker -> GICS sector for the sector-neutral z-score.
    factor_matrices:
        Mapping from factor name to its date x ticker matrix
        (:func:`common.factor_model.load_factor_matrix`).
    benchmark_weights:
        KN600 daily constituent weights (date x ticker), the optimizer's ``w_bench``.
    volatility:
        Optional ticker -> volatility for the diagonal risk model. Defaults to 1.
    adv:
        Optional ticker -> average dollar volume for the liquidity cap. Defaults to
        deep liquidity (no binding cap).
    cash_schedule:
        Optional date -> cash weight from the regime overlay (§5.7); read as-of each
        rebalancing date. When absent, the fixed ``BacktestConfig.cash`` is used.

    """

    sectors: pd.Series[str]
    factor_matrices: Mapping[str, pd.DataFrame]
    benchmark_weights: pd.DataFrame
    volatility: pd.Series[float] | None = None
    adv: pd.Series[float] | None = None
    cash_schedule: pd.Series[float] | None = None


@dataclasses.dataclass(frozen=True)
class BacktestConfig:
    """Run configuration for the rebalance loop.

    Attributes
    ----------
    params:
        Optimizer configuration (:class:`portfolio.optimizer.OptimizerParams`).
    sleeves:
        Alpha sleeves; defaults to the canonical quality/value/momentum.
    cash:
        Cash weight held out of the invested budget each date.
    aum:
        Assets under management, for the liquidity cap and market impact.

    """

    params: OptimizerParams = dataclasses.field(default_factory=OptimizerParams)
    sleeves: Mapping[str, Sequence[str]] | None = None
    cash: float = 0.01
    aum: float = 100_000_000.0


def construct_portfolios(
    signal_frames: Mapping[str, pd.DataFrame],
    rebalance_dates: Sequence[pd.Timestamp],
    data: BacktestData,
    config: BacktestConfig | None = None,
    alpha_override: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run the rebalance loop and return the date x ticker weight panel.

    Parameters
    ----------
    signal_frames:
        Ticker -> per-ticker signal frame (:func:`signals.pipeline.compute_ticker_signals`).
    rebalance_dates:
        Ordered rebalancing dates.
    data:
        The cross-date market data (:class:`BacktestData`).
    config:
        Run configuration (:class:`BacktestConfig`); defaults are used if omitted.
    alpha_override:
        Optional pre-computed expected-alpha panel (date x ticker), e.g. the
        ML-governed score. When given, it replaces the rules score from
        ``assemble_cross_section`` as the optimizer's alpha.

    Returns
    -------
    pandas.DataFrame
        Indexed by rebalancing date, one column per name ever held, weights summing
        to ``1 - cash`` per date (missing entries are 0).

    """
    config = config or BacktestConfig()
    weights_by_date: dict[pd.Timestamp, pd.Series[float]] = {}
    previous: pd.Series[float] | None = None

    for date in rebalance_dates:
        full_design = factor_exposures_at(data.factor_matrices, date)
        if alpha_override is not None:
            empty = pd.Series(dtype=float)
            row = alpha_override.loc[date] if date in alpha_override.index else empty
            alpha = cast("pd.Series[float]", row).dropna()
        else:
            scores = assemble_cross_section(
                signal_frames, date, data.sectors, full_design, sleeves=config.sleeves
            )
            alpha = scores["score"].dropna()
        # Restrict to names with a factor design and a known sector (the optimizer
        # needs both); for the assemble path this intersection is already satisfied.
        universe = alpha.index.intersection(full_design.index).intersection(
            data.sectors.dropna().index
        )
        alpha = alpha.reindex(universe)
        if alpha.empty:
            continue

        cash = _cash_for(data.cash_schedule, date, config.cash)
        design = full_design.reindex(universe)
        vol = _risk_volatility(signal_frames, universe, date, data.volatility)
        covariance = pd.DataFrame(np.diag(vol.to_numpy() ** 2), index=universe, columns=universe)
        # asof(scalar) returns the row as a Series; the stub widens it to a union.
        bench_row = cast("pd.Series[float]", data.benchmark_weights.asof(date))
        bench = bench_row.reindex(universe).fillna(0.0)
        # Renormalize the benchmark to the *invested* budget (1 - cash) over the
        # investable universe, so active weights reflect stock selection only — the
        # cash held by the regime overlay is a deliberate top-down bet, not active
        # stock risk, and would otherwise blow through the sector/factor bands.
        bench_total = bench.sum()
        if bench_total > 0:
            bench = bench / bench_total * (1.0 - cash)
        prev = previous.reindex(universe).fillna(0.0) if previous is not None else bench

        inputs = OptimizerInputs(
            alpha=alpha,
            covariance=covariance,
            factor_exposures=design,
            benchmark_weights=bench,
            previous_weights=prev,
            sectors=data.sectors.reindex(universe),
            adv=_series_for(data.adv, universe, _DEEP_ADV),
            cash=cash,
            aum=config.aum,
        )
        weights = optimize_portfolio(inputs, config.params)
        weights_by_date[date] = weights
        previous = weights

    return pd.DataFrame(weights_by_date).T.fillna(0.0)


def _series_for(
    source: pd.Series[float] | None, universe: pd.Index[str], default: float
) -> pd.Series[float]:
    """Reindex ``source`` to ``universe`` (missing -> default); all-default if None."""
    if source is None:
        return pd.Series(default, index=universe, dtype=float)
    return source.reindex(universe).fillna(default)


def _cash_for(schedule: pd.Series[float] | None, date: pd.Timestamp, default: float) -> float:
    """Cash weight for ``date`` from the regime schedule (as-of), else the default."""
    if schedule is None:
        return default
    value = schedule.asof(date)
    return float(value) if pd.notna(value) else default


def _risk_volatility(
    signal_frames: Mapping[str, pd.DataFrame],
    universe: pd.Index[str],
    date: pd.Timestamp,
    override: pd.Series[float] | None,
) -> pd.Series[float]:
    """Per-name volatility for the diagonal risk model.

    Uses a static ``override`` when supplied; otherwise reads each name's realized
    volatility as-of ``date`` from its signal frame (point-in-time — only data up to
    the rebalancing date). Missing values fall back to the cross-sectional median so
    a name without an estimate carries typical, not zero, risk.
    """
    if override is not None:
        source = override.reindex(universe)
    else:
        values = {
            ticker: cast("float", frame[_RISK_VOL_COLUMN].asof(date))
            for ticker, frame in signal_frames.items()
            if ticker in universe and _RISK_VOL_COLUMN in frame.columns
        }
        source = pd.Series(values, dtype=float).reindex(universe)
    median = source.median()
    fallback = float(median) if pd.notna(median) else _DEFAULT_VOLATILITY
    return source.fillna(fallback).clip(lower=_VOL_FLOOR)
