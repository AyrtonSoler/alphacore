"""RED tests for ml.metrics — the information coefficient (v3 §5.5/§8).

The IC (cross-sectional Spearman rank correlation between predicted score and
realized residual return) is how the linear-vs-non-linear experiment is judged OOS
and what the Trust Governor (§5.6) compares to decide the ML's blend weight.

Acceptance
----------
- Perfectly rank-aligned predictions score IC = +1; reversed score -1.
- Independent predictions score ~0.
- It is rank-based, so any monotone transform of the prediction is invariant.
- Misaligned indices intersect; fewer than two paired points score 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphacore.ml.metrics import information_coefficient


def test_perfectly_aligned_ranks_score_one() -> None:
    """Predictions that rank identically to outcomes have IC = 1."""
    realized = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])
    predicted = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.isclose(information_coefficient(predicted, realized), 1.0)


def test_reversed_ranks_score_minus_one() -> None:
    """Predictions that rank in reverse have IC = -1."""
    realized = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])
    predicted = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0])
    assert np.isclose(information_coefficient(predicted, realized), -1.0)


def test_independent_predictions_score_near_zero() -> None:
    """Independent random predictions have IC near zero over many names."""
    rng = np.random.default_rng(0)
    realized = pd.Series(rng.normal(size=2000))
    predicted = pd.Series(rng.normal(size=2000))
    assert abs(information_coefficient(predicted, realized)) < 0.1


def test_ic_is_invariant_to_monotone_transform() -> None:
    """Being rank-based, the IC is unchanged by any increasing transform."""
    rng = np.random.default_rng(1)
    realized = pd.Series(rng.normal(size=200))
    predicted = pd.Series(rng.normal(size=200))
    base = information_coefficient(predicted, realized)
    squashed = information_coefficient(np.tanh(predicted * 3.0), realized)
    assert np.isclose(base, squashed)


def test_too_few_points_score_zero() -> None:
    """A single paired observation cannot define a correlation -> 0."""
    realized = pd.Series([0.01], index=["A"])
    predicted = pd.Series([1.0], index=["A"])
    assert information_coefficient(predicted, realized) == 0.0
