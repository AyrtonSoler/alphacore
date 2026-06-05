"""Cross-sectional scorers — linear baseline and glass-box EBM (v3 §5.5).

Two production models, both trained on the **residual** forward return so they can
only learn what is orthogonal to the factor model (training on raw return would just
re-learn the factors v3 neutralizes):

- :class:`LinearScorer` — ElasticNet, the null hypothesis ("does exploitable
  non-linearity even exist?"). Cheap, mandatory, defensible.
- :class:`EbmScorer` — the Explainable Boosting Machine (InterpretML), a glass-box
  GA²M whose per-feature shape functions and pairwise-interaction heatmaps *are* the
  explanation. It captures the interactions the additive linear model cannot, and
  supports monotone constraints (Trust Governor lock 1, enforced at training).

Both share a ``fit`` / ``predict`` surface so the §8 experiment and the Trust
Governor (§5.6) can treat them interchangeably. Every estimator is seeded
(``random_state=42``) for reproducibility.

References
----------
.. [ZHT2005] Zou, H. and Hastie, T. (2005). "Regularization and Variable Selection
   via the Elastic Net." JRSS B 67(2), 301-320.
.. [Lou2013] Lou, Y., Caruana, R., Gehrke, J. and Hooker, G. (2013). "Accurate
   Intelligible Models with Pairwise Interactions." KDD 2013. (GA²M / EBM.)

"""

from __future__ import annotations

from typing import Self

import pandas as pd
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.linear_model import ElasticNet

_DEFAULT_SEED = 42


class LinearScorer:
    """ElasticNet linear baseline — the non-linearity null hypothesis (§5.5).

    Parameters
    ----------
    alpha:
        Overall regularization strength. A small default keeps the baseline a
        genuine fit rather than a shrink-to-zero.
    l1_ratio:
        Elastic-net mix between L1 (1.0) and L2 (0.0) penalties.
    random_state:
        Seed for reproducibility.

    References
    ----------
    Elastic net from [ZHT2005]_.

    """

    def __init__(
        self,
        alpha: float = 0.01,
        l1_ratio: float = 0.5,
        random_state: int = _DEFAULT_SEED,
    ) -> None:
        """Configure the ElasticNet baseline (see class docstring for parameters)."""
        self._model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)

    def fit(self, features: pd.DataFrame, target: pd.Series[float]) -> Self:
        """Fit on a feature matrix and residual-return target."""
        self._model.fit(features.to_numpy(dtype=float), target.to_numpy(dtype=float))
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series[float]:
        """Predict a per-name score aligned to ``features.index``."""
        scores = self._model.predict(features.to_numpy(dtype=float))
        return pd.Series(scores, index=features.index, dtype=float)


class EbmScorer:
    """Explainable Boosting Machine — glass-box non-linear scorer (§5.5).

    Parameters
    ----------
    interactions:
        Number of pairwise interaction terms to fit (0 = a purely additive GAM).
    outer_bags:
        Number of outer bagging rounds. Fewer is faster (used small in tests);
        production uses the library default for stability.
    monotone_constraints:
        Optional mapping from feature name to ``+1`` (non-decreasing) or ``-1``
        (non-increasing). Enforced isotonically at training — Trust Governor lock 1.
    random_state:
        Seed for reproducibility.

    References
    ----------
    GA²M / EBM from [Lou2013]_.

    """

    def __init__(
        self,
        interactions: int = 10,
        outer_bags: int = 8,
        monotone_constraints: dict[str, int] | None = None,
        random_state: int = _DEFAULT_SEED,
    ) -> None:
        """Configure the EBM (see class docstring for parameters)."""
        self._interactions = interactions
        self._outer_bags = outer_bags
        self._monotone_constraints = monotone_constraints
        self._random_state = random_state
        self._model: ExplainableBoostingRegressor | None = None

    def fit(self, features: pd.DataFrame, target: pd.Series[float]) -> Self:
        """Fit on a feature matrix and residual-return target."""
        monotone = None
        if self._monotone_constraints is not None:
            monotone = [self._monotone_constraints.get(column, 0) for column in features.columns]
        self._model = ExplainableBoostingRegressor(
            interactions=self._interactions,
            outer_bags=self._outer_bags,
            monotone_constraints=monotone,
            random_state=self._random_state,
        )
        self._model.fit(features.to_numpy(dtype=float), target.to_numpy(dtype=float))
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series[float]:
        """Predict a per-name score aligned to ``features.index``."""
        if self._model is None:
            msg = "EbmScorer.predict called before fit"
            raise ValueError(msg)
        scores = self._model.predict(features.to_numpy(dtype=float))
        return pd.Series(scores, index=features.index, dtype=float)
