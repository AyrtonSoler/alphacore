"""RED tests for ml.scorers — linear baseline vs glass-box EBM (v3 §5.5).

Two production models trained on the residual forward return:

- ``LinearScorer`` (ElasticNet) — the null hypothesis: "does non-linearity even
  exist?". Cheap, mandatory, defensible.
- ``EbmScorer`` (Explainable Boosting Machine) — glass-box, with optional monotone
  constraints, captures pairwise interactions an additive linear model cannot.

The headline test is the falsifiable experiment of §8: on a pure interaction the EBM
beats the linear baseline out-of-sample, while the linear baseline scores ~0.

Acceptance
----------
- The linear scorer recovers a linear signal (high OOS IC) and is deterministic.
- The EBM recovers a linear signal too (sanity).
- On a pure x0*x1 interaction the EBM beats the linear baseline OOS, and the linear
  baseline's IC is ~0 — the experiment that justifies (or refutes) the ML.
- A monotone-increasing constraint makes EBM predictions non-decreasing in that
  feature (Trust Governor lock 1, enforced at training).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.metrics import information_coefficient
from ml.scorers import EbmScorer, LinearScorer

_COLS = ["x0", "x1", "x2"]


def _features(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.uniform(-1.0, 1.0, size=(n, 3)), columns=_COLS)


def _split(x: pd.DataFrame, y: pd.Series, cut: int):
    return x.iloc[:cut], y.iloc[:cut], x.iloc[cut:], y.iloc[cut:]


def test_linear_scorer_recovers_linear_signal() -> None:
    """ElasticNet recovers a linear target with high out-of-sample IC."""
    x = _features(400, seed=0)
    rng = np.random.default_rng(10)
    y = pd.Series(x.to_numpy() @ [2.0, -1.0, 0.5] + 0.05 * rng.normal(size=400))
    x_tr, y_tr, x_te, y_te = _split(x, y, 300)
    pred = LinearScorer().fit(x_tr, y_tr).predict(x_te)
    assert information_coefficient(pred, y_te) > 0.7


def test_linear_scorer_is_deterministic() -> None:
    """A fixed seed makes predictions reproducible."""
    x = _features(200, seed=1)
    y = pd.Series(x["x0"].to_numpy() * 1.5)
    a = LinearScorer().fit(x, y).predict(x)
    b = LinearScorer().fit(x, y).predict(x)
    pd.testing.assert_series_equal(a, b)


def test_ebm_recovers_linear_signal() -> None:
    """The EBM also recovers a linear signal (sanity check)."""
    x = _features(400, seed=2)
    rng = np.random.default_rng(11)
    y = pd.Series(x.to_numpy() @ [1.5, -0.8, 0.3] + 0.05 * rng.normal(size=400))
    x_tr, y_tr, x_te, y_te = _split(x, y, 300)
    pred = EbmScorer(interactions=0, outer_bags=2).fit(x_tr, y_tr).predict(x_te)
    assert information_coefficient(pred, y_te) > 0.5


@pytest.mark.slow
def test_ebm_beats_linear_on_pure_interaction() -> None:
    """On y = x0*x1 the EBM beats the linear baseline OOS; the linear scores ~0."""
    # A larger test block keeps the linear baseline's finite-sample IC near its true
    # zero, so the separation reflects model capability, not sampling noise.
    x = _features(1200, seed=3)
    rng = np.random.default_rng(12)
    y = pd.Series(x["x0"].to_numpy() * x["x1"].to_numpy() + 0.02 * rng.normal(size=1200))
    x_tr, y_tr, x_te, y_te = _split(x, y, 900)

    linear_pred = LinearScorer().fit(x_tr, y_tr).predict(x_te)
    ebm_pred = EbmScorer(interactions=3, outer_bags=4).fit(x_tr, y_tr).predict(x_te)

    ic_linear = information_coefficient(linear_pred, y_te)
    ic_ebm = information_coefficient(ebm_pred, y_te)

    assert abs(ic_linear) < 0.15  # an additive model is blind to the interaction
    assert ic_ebm > 0.30  # the EBM sees it
    assert ic_ebm > ic_linear


def test_ebm_monotone_constraint_is_enforced() -> None:
    """A +1 monotone constraint makes predictions non-decreasing in that feature."""
    x = _features(400, seed=4)
    rng = np.random.default_rng(13)
    y = pd.Series(x["x0"].to_numpy() + 0.05 * rng.normal(size=400))
    model = EbmScorer(interactions=0, outer_bags=2, monotone_constraints={"x0": 1}).fit(x, y)

    grid = pd.DataFrame({"x0": np.linspace(-1.0, 1.0, 25), "x1": 0.0, "x2": 0.0}, columns=_COLS)
    preds = model.predict(grid).to_numpy()
    assert (np.diff(preds) >= -1e-9).all()  # non-decreasing in x0
