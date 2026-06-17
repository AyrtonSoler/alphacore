"""Walk-forward model evaluation — the governed ML decision (v3 §5.5/§8).

Runs a scorer across the purged walk-forward folds (§8), scores each held-out fold
by information coefficient (§5.5), and averages. Comparing the EBM's mean OOS IC to
the linear baseline's is exactly what the Trust Governor (§5.6) consumes to decide
the ML's blend weight: the ML earns weight only when it genuinely beats the linear
null hypothesis out-of-sample, and is benched (lambda = 1) otherwise.

The harness is generic over any scorer with a ``fit`` / ``predict`` surface and is
deterministic given seeded scorers and fixed splits.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .metrics import information_coefficient


class Scorer(Protocol):
    """Structural type for a cross-sectional scorer (LinearScorer, EbmScorer)."""

    def fit(self, features: pd.DataFrame, target: pd.Series[float]) -> Scorer:
        """Fit the model and return self."""
        ...

    def predict(self, features: pd.DataFrame) -> pd.Series[float]:
        """Predict a per-name score aligned to ``features.index``."""
        ...


_Splits = Sequence[tuple[NDArray[np.intp], NDArray[np.intp]]]


def walk_forward_ic(
    make_scorer: Callable[[], Scorer],
    features: pd.DataFrame,
    target: pd.Series[float],
    splits: _Splits,
) -> list[float]:
    """Out-of-sample information coefficient on each walk-forward fold.

    Parameters
    ----------
    make_scorer:
        Zero-argument factory returning a fresh scorer, so each fold is fit from
        scratch (no leakage of one fold's fit into the next).
    features:
        Feature matrix, one row per sample, aligned to ``target``.
    target:
        Residual forward return (or its rank) per sample.
    splits:
        Train/test index pairs from
        :func:`alphacore.robustness.cross_validation.purged_walk_forward_splits`.

    Returns
    -------
    list of float
        One held-out IC per fold, in fold order.

    """
    ics: list[float] = []
    for train_index, test_index in splits:
        scorer = make_scorer()
        scorer.fit(features.iloc[train_index], target.iloc[train_index])
        prediction = scorer.predict(features.iloc[test_index])
        ics.append(information_coefficient(prediction, target.iloc[test_index]))
    return ics


def mean_oos_ic(
    make_scorer: Callable[[], Scorer],
    features: pd.DataFrame,
    target: pd.Series[float],
    splits: _Splits,
) -> float:
    """Mean out-of-sample IC across folds; 0.0 when there are no folds.

    This is the scalar the Trust Governor compares between the EBM and the linear
    baseline to set the ML's blend weight.
    """
    ics = walk_forward_ic(make_scorer, features, target, splits)
    if not ics:
        return 0.0
    return float(np.mean(ics))
