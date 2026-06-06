"""The definitive strategy — alpha definition + weight construction (team decision).

After three ablation rounds (and the correction of a sector-map bug that had made the
§5.3 z-score accidentally market-wide), the team fixed the production strategy:

- **Sleeves:** Quality + Momentum (Value dropped — it diluted the blend).
- **Cross-sectional standardization:** a **market-wide** robust z-score, applied
  deliberately. This is NOT sector-neutral: each signal is standardized across the
  whole investable cross-section, so a name is ranked against the entire universe, not
  only its sector peers. (The earlier "sector-neutral §5.3" label does not apply to
  this strategy — see ``zscore_grouping``.)
- **Factor neutralization (§5.4):** the market-wide composite is still residualized
  against the KN factor design, so the alpha is the part orthogonal to the style
  factors — the auto-correlation defence is preserved at the signal level.
- **Construction:** the convex optimizer with no risk term (``gamma_risk = 0``), no
  factor/sector bands, the 12.5% cap and the 15-name floor — which in this regime
  reduces to a top-15 equal-weight selection by alpha.
- **Cadence:** quarterly.

The market-wide choice trades a little information ratio versus a sector-neutral
momentum sleeve for the robustness of a two-sleeve (Quality + Momentum) blend; it is a
deliberate, pre-registered decision, implemented honestly here (a constant grouping
label), never via the total-return-file bug that originally produced it.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Literal

import pandas as pd

from common.factor_model import factor_exposures_at
from ml.allocation import AllocatorParams, tilt_weights
from signals.cross_section import c_zscore_sector_neutral
from signals.neutralization import residualize_signal
from signals.pipeline import DEFAULT_SLEEVES

from .construction import BacktestConfig, BacktestData, construct_portfolios
from .optimizer import OptimizerParams

ZScoreGrouping = Literal["sector", "market_wide"]

# A single market-wide group label: standardize across the whole cross-section.
_MARKET_WIDE_LABEL = "ALL"
# A name counts as held above this weight (the optimizer leaves ~0 on non-selected names).
_HELD_THRESHOLD = 1e-6


def _quality_momentum_sleeves() -> dict[str, list[str]]:
    """Return the canonical Quality + Momentum sleeve definitions (Value excluded)."""
    return {sleeve: list(DEFAULT_SLEEVES[sleeve]) for sleeve in ("quality", "momentum")}


@dataclasses.dataclass(frozen=True)
class StrategyParams:
    """Pre-registered configuration of the definitive strategy (team decision).

    Attributes
    ----------
    sleeves:
        Sleeve name -> member signal columns. Defaults to Quality + Momentum.
    zscore_grouping:
        ``"market_wide"`` (the decision) standardizes each signal across the whole
        cross-section; ``"sector"`` standardizes within each GICS sector.
    gamma_risk, eps_factor, eps_sector, cap_per_name, min_holdings:
        Optimizer settings (:class:`portfolio.optimizer.OptimizerParams`). The defaults
        switch the risk term and the factor/sector bands off and keep the official
        12.5% cap and 15-name floor.
    cash:
        Cash weight held out of the invested budget each date.
    aum:
        Assets under management, for the liquidity cap and market impact.
    cadence_months:
        Rebalance every this many months (3 = quarterly).
    allocator:
        Optional ML allocation overlay. ``None`` (the default) keeps the equal-weight
        sizing of the definitive strategy; when set, the held names are re-sized from the
        frozen ML predictions passed to :func:`build_strategy_weights` (selection is never
        changed). See :mod:`ml.allocation`.

    """

    sleeves: Mapping[str, Sequence[str]] = dataclasses.field(
        default_factory=_quality_momentum_sleeves
    )
    zscore_grouping: ZScoreGrouping = "market_wide"
    gamma_risk: float = 0.0
    eps_factor: float = 1e9
    eps_sector: float = 1e9
    cap_per_name: float = 0.125
    min_holdings: int = 15
    cash: float = 0.01
    aum: float = 100_000_000.0
    cadence_months: int = 3
    allocator: AllocatorParams | None = None


# Frozen, so a single shared default instance is safe to reuse as a parameter default.
_DEFAULT_PARAMS = StrategyParams()


def select_rebalance_dates(
    month_ends: Sequence[pd.Timestamp], params: StrategyParams
) -> list[pd.Timestamp]:
    """Subsample ordered month-end dates to the configured cadence.

    Parameters
    ----------
    month_ends:
        Ordered month-end candidate dates.
    params:
        Strategy configuration; ``cadence_months`` sets the stride (3 = quarterly).

    Returns
    -------
    list of pandas.Timestamp
        Every ``cadence_months``-th date, starting from the first.

    """
    return list(month_ends)[:: params.cadence_months]


def _alpha_panel(  # noqa: PLR0913 - cohesive per-date alpha inputs
    signal_frames: Mapping[str, pd.DataFrame],
    factor_matrices: Mapping[str, pd.DataFrame],
    sectors: pd.Series[str],
    rebalance_dates: Sequence[pd.Timestamp],
    params: StrategyParams,
) -> pd.DataFrame:
    """Build the date x ticker alpha: market-wide z-score, then §5.4 residualization.

    For each date and sleeve, every member signal is standardized (across the whole
    cross-section when ``zscore_grouping == "market_wide"``, else within sector), the
    sleeve members are averaged into a composite, and the composite is residualized
    against the factor design. The per-date alpha is the mean of the sleeve residuals.
    """
    rows: dict[pd.Timestamp, pd.Series[float]] = {}
    for date in rebalance_dates:
        design = factor_exposures_at(factor_matrices, date)
        cross = pd.DataFrame({t: f.asof(date) for t, f in signal_frames.items()}).T
        common = cross.index.intersection(sectors.dropna().index).intersection(design.index)
        cross = cross.loc[common].dropna(how="all")
        if cross.empty:
            continue
        if params.zscore_grouping == "market_wide":
            labels = pd.Series(_MARKET_WIDE_LABEL, index=cross.index)
        else:
            labels = sectors.loc[cross.index]
        sleeve_design = design.reindex(cross.index)
        residuals: list[pd.Series[float]] = []
        for columns in params.sleeves.values():
            present = [column for column in columns if column in cross.columns]
            if not present:
                continue
            z_scores = pd.DataFrame(
                {column: c_zscore_sector_neutral(cross[column], labels) for column in present}
            )
            residuals.append(residualize_signal(z_scores.mean(axis=1), sleeve_design))
        if not residuals:
            continue
        # Mean of the sleeve residuals (concat keeps the result typed as a Series).
        rows[date] = pd.concat(residuals, axis=1).mean(axis=1)
    return pd.DataFrame(rows).T


def build_strategy_weights(  # noqa: PLR0913 - cohesive strategy inputs
    signal_frames: Mapping[str, pd.DataFrame],
    factor_matrices: Mapping[str, pd.DataFrame],
    sectors: pd.Series[str],
    benchmark_weights: pd.DataFrame,
    rebalance_dates: Sequence[pd.Timestamp],
    params: StrategyParams = _DEFAULT_PARAMS,
    ml_predictions: pd.DataFrame | None = None,
    ml_lambda: pd.Series[float] | None = None,
) -> pd.DataFrame:
    """Build the date x ticker weight panel of the definitive strategy.

    Parameters
    ----------
    signal_frames:
        Ticker -> per-ticker signal frame (:func:`signals.pipeline.compute_ticker_signals`).
    factor_matrices:
        Factor name -> date x ticker matrix (:func:`common.factor_model.load_factor_matrix`).
    sectors:
        Ticker -> GICS sector (used for the optional sector grouping and the optimizer).
    benchmark_weights:
        KN600 date x ticker constituent weights (the optimizer's ``w_bench``).
    rebalance_dates:
        Ordered rebalancing dates (use :func:`select_rebalance_dates` for the cadence).
    params:
        Strategy configuration (:class:`StrategyParams`); defaults encode the team decision.
    ml_predictions:
        Optional frozen Phase-A prediction panel (date x ticker forward returns). Only used
        when ``params.allocator`` is set; the held names are re-sized from it.
    ml_lambda:
        Optional per-date tilt strength (:func:`ml.allocation.earned_lambda`), paired with
        ``ml_predictions``.

    Returns
    -------
    pandas.DataFrame
        Date x ticker weights, long-only, within the cap, summing to ``1 - cash`` per
        date. Selection (which names are held) is always Q+M's; the ML overlay only
        changes the money split among held names.

    """
    alpha = _alpha_panel(signal_frames, factor_matrices, sectors, rebalance_dates, params)
    data = BacktestData(
        sectors=sectors, factor_matrices=factor_matrices, benchmark_weights=benchmark_weights
    )
    config = BacktestConfig(
        params=OptimizerParams(
            gamma_risk=params.gamma_risk,
            eps_factor=params.eps_factor,
            eps_sector=params.eps_sector,
            cap_per_name=params.cap_per_name,
            min_holdings=params.min_holdings,
        ),
        cash=params.cash,
        aum=params.aum,
    )
    weights = construct_portfolios(
        signal_frames, rebalance_dates, data, config, alpha_override=alpha
    )
    if params.allocator is None or ml_predictions is None or ml_lambda is None:
        return weights
    # Keep the overlay's cap/cash consistent with the strategy's.
    allocator = dataclasses.replace(
        params.allocator, cash=params.cash, cap_per_name=params.cap_per_name
    )
    return _apply_ml_allocation(weights, ml_predictions, ml_lambda, allocator)


def _apply_ml_allocation(
    weights: pd.DataFrame,
    ml_predictions: pd.DataFrame,
    ml_lambda: pd.Series[float],
    allocator: AllocatorParams,
) -> pd.DataFrame:
    """Re-size each date's held names from the frozen ML predictions (selection unchanged).

    For every rebalance date, the held names are re-weighted by :func:`ml.allocation.
    tilt_weights` using that date's frozen predictions and earned tilt. Dates with no
    earned tilt, missing predictions, or no holdings keep the equal-weight allocation.
    """
    out = weights.copy()
    for date in out.index:
        row = out.loc[date]
        held = row[row > _HELD_THRESHOLD]
        if held.empty or date not in ml_predictions.index or date not in ml_lambda.index:
            continue
        lam = float(ml_lambda.loc[date])
        predicted = ml_predictions.loc[date].reindex(held.index)
        if lam <= 0.0 or bool(predicted.isna().any()):
            continue
        tilted = tilt_weights(predicted, lam, allocator)
        out.loc[date, :] = 0.0
        out.loc[date, tilted.index] = tilted.to_numpy()
    return out
