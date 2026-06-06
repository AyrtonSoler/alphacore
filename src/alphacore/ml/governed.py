"""Governed alpha panel — the ML overlay, wired and boxed in (§5.5/§5.6).

Walk-forward, point-in-time: an ElasticNet ML model is trained on the sleeve-residual
features to predict each name's residual forward return, and the Trust Governor blends
its score with the rules score *only to the extent it beat the rules out-of-sample*
(expanding-window information coefficient). The blend weight degrades to pure rules
(``lambda = 1``) when the ML does not clear the ``min_edge`` margin — the distrust the
governor exists to encode (§5.6). Influence is then bounded so the ML cannot move a
name's rank by more than ``k_max`` percentiles.

The output is a date x name governed-alpha panel in cross-sectional **rank units**
(``[0, 1]``), so its scale is consistent across dates — the optimizer (§5.8) consumes
it as expected alpha via ``construct_portfolios(alpha_override=...)``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
import pandas as pd

from .metrics import information_coefficient
from .scorers import LinearScorer
from .trust_governor import blend_scores, bound_rank_influence, trust_weight

_MIN_NAMES = 2
_MIN_IC_HISTORY = 3
_PURGE = 1


def build_governed_alpha(  # noqa: PLR0913 - ML hyperparameters, each meaningful
    rules: pd.DataFrame,
    features: Mapping[str, pd.DataFrame],
    target: pd.DataFrame,
    *,
    min_train_periods: int = 24,
    refit_every: int = 6,
    min_edge: float = 0.0,
    lambda_min: float = 0.5,
    k_max_pctile: float = 20.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """Build the ML-governed alpha panel from the rules score and ML features.

    Parameters
    ----------
    rules:
        Rules residual-alpha score (date x name) — the governor's baseline.
    features:
        Mapping from feature name to its (date x name) panel (the sleeve residuals).
    target:
        Residual forward return (date x name): the value over the period *following*
        each date, which the ML predicts.
    min_train_periods:
        Periods of history before the ML starts predicting.
    refit_every:
        Refit the ML every this many periods (cost control).
    min_edge:
        Minimum OOS IC margin the ML must beat the rules by to earn any weight.
    lambda_min:
        Floor on the rules weight (the ML never exceeds ``1 - lambda_min``).
    k_max_pctile:
        Maximum percentile the ML may move a name's rank from its rules rank.
    random_state:
        Seed for the ElasticNet.

    Returns
    -------
    pandas.DataFrame
        Governed alpha (date x name) in ``[0, 1]`` rank units. Dates before the
        warm-up, or where the ML earns no weight, equal the rules ranking.

    """
    dates = list(rules.index)
    rules_rank = rules.rank(axis=1, pct=True)
    governed = rules_rank.copy()

    ml_pred = _walk_forward_predictions(
        features, target, dates, min_train_periods, refit_every, random_state
    )

    for k, date in enumerate(dates):
        ml_today = ml_pred.loc[date].dropna()
        if len(ml_today) < _MIN_NAMES:
            continue
        lam = _lambda_at(ml_pred, rules, target, dates, k, min_edge, lambda_min)
        if lam >= 1.0:
            continue
        rules_today = rules.loc[date].dropna()
        common = rules_today.index.intersection(ml_today.index)
        if len(common) < _MIN_NAMES:
            continue
        rules_pct = rules_today[common].rank(pct=True) * 100.0
        ml_pct = ml_today[common].rank(pct=True) * 100.0
        blended = blend_scores(rules_pct, ml_pct, lam)
        bounded = bound_rank_influence(blended, rules_today[common], k_max_pctile)
        governed.loc[date, common] = (bounded / 100.0).to_numpy()

    return governed


def _walk_forward_predictions(  # noqa: PLR0913 - cohesive walk-forward parameters
    features: Mapping[str, pd.DataFrame],
    target: pd.DataFrame,
    dates: Sequence[pd.Timestamp],
    min_train_periods: int,
    refit_every: int,
    random_state: int,
) -> pd.DataFrame:
    """Out-of-sample ML score per (date, name): trained only on prior periods."""
    names = target.columns
    predictions = pd.DataFrame(index=pd.Index(dates), columns=names, dtype=float)
    scorer: LinearScorer | None = None
    for k, date in enumerate(dates):
        if k < min_train_periods:
            continue
        if scorer is None or (k - min_train_periods) % refit_every == 0:
            x_train, y_train = _training_matrix(features, target, dates[: k - _PURGE])
            if len(x_train) >= _MIN_NAMES:
                scorer = LinearScorer(random_state=random_state).fit(x_train, y_train)
        if scorer is None:
            continue
        x_today = _features_at(features, date)
        if not x_today.empty:
            predictions.loc[date, x_today.index] = scorer.predict(x_today).to_numpy()
    return predictions


def _training_matrix(
    features: Mapping[str, pd.DataFrame],
    target: pd.DataFrame,
    train_dates: Sequence[pd.Timestamp],
) -> tuple[pd.DataFrame, pd.Series[float]]:
    """Stack the feature/target panels over ``train_dates`` into an (X, y) design."""
    columns = {name: panel.loc[list(train_dates)].stack() for name, panel in features.items()}
    design = pd.DataFrame(columns)
    design["__target__"] = target.loc[list(train_dates)].stack()
    design = design.dropna()
    return design[list(features)], design["__target__"]


def _features_at(features: Mapping[str, pd.DataFrame], date: pd.Timestamp) -> pd.DataFrame:
    """Per-name feature matrix on ``date`` (complete rows only)."""
    return pd.DataFrame({name: panel.loc[date] for name, panel in features.items()}).dropna()


def _lambda_at(  # noqa: PLR0913 - cohesive expanding-IC parameters
    ml_pred: pd.DataFrame,
    rules: pd.DataFrame,
    target: pd.DataFrame,
    dates: Sequence[pd.Timestamp],
    k: int,
    min_edge: float,
    lambda_min: float,
) -> float:
    """Rules weight at ``dates[k]`` from the expanding-window OOS IC of ML vs rules."""
    ic_ml: list[float] = []
    ic_rules: list[float] = []
    for j in range(k):
        past = dates[j]
        ml_past = cast("pd.Series[float]", ml_pred.loc[past]).dropna()
        if len(ml_past) < _MIN_NAMES:
            continue
        realized = cast("pd.Series[float]", target.loc[past])
        rules_past = cast("pd.Series[float]", rules.loc[past]).dropna()
        ic_ml.append(information_coefficient(ml_past, realized))
        ic_rules.append(information_coefficient(rules_past, realized))
    if len(ic_ml) < _MIN_IC_HISTORY:
        return 1.0
    return trust_weight(
        float(np.mean(ic_ml)), float(np.mean(ic_rules)), min_edge=min_edge, lambda_min=lambda_min
    )
