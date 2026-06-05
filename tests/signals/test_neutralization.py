"""RED tests for signals.neutralization — factor residualization (v3 §5.4, CORE).

This is the heart of v3: every composite signal is regressed cross-sectionally on
the five KN style factors + 11 GICS dummies and replaced by its **residual**, so
the ranking expresses the spread *orthogonal* to the factors rather than the
factors themselves. The mathematical guarantee (normal equations) is that the
residual is orthogonal to every column of the design matrix.

Acceptance
----------
- The residual is orthogonal to every factor column (zero correlation) and, with
  an intercept, has ~zero mean — the §5.4 guarantee.
- A signal that is exactly a linear combination of the factors residualizes to ~0
  (it carried no orthogonal information).
- A signal already orthogonal to the factors is returned essentially unchanged.
- Weighted residualization is orthogonal under the weighted inner product.
- Names with a NaN signal get a NaN residual and do not corrupt the fit.
- A rank-deficient design (the GICS dummy trap: intercept + all dummies) is handled
  gracefully (least-squares projection), still yielding an orthogonal residual.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from signals.neutralization import residualize_signal


def _synthetic_factors(n: int = 200, k: int = 5, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = [f"f{j}" for j in range(k)]
    return pd.DataFrame(rng.normal(size=(n, k)), columns=cols)


def _max_abs_factor_correlation(residual: pd.Series, factors: pd.DataFrame) -> float:
    r = residual.to_numpy()
    mask = np.isfinite(r)
    return max(
        abs(np.corrcoef(r[mask], factors[col].to_numpy()[mask])[0, 1]) for col in factors.columns
    )


def test_residual_is_orthogonal_to_every_factor() -> None:
    """The residual has ~zero correlation with each factor column."""
    factors = _synthetic_factors()
    rng = np.random.default_rng(1)
    signal = pd.Series(
        factors.to_numpy() @ rng.normal(size=factors.shape[1]) + rng.normal(size=len(factors))
    )
    residual = residualize_signal(signal, factors)
    assert _max_abs_factor_correlation(residual, factors) < 1e-8


def test_residual_has_zero_mean_with_intercept() -> None:
    """With the intercept in the design, the residual is mean-zero."""
    factors = _synthetic_factors()
    rng = np.random.default_rng(2)
    signal = pd.Series(rng.normal(size=len(factors)) + 5.0)  # non-zero mean
    residual = residualize_signal(signal, factors, add_intercept=True)
    assert abs(residual.mean()) < 1e-8


def test_pure_factor_signal_residualizes_to_zero() -> None:
    """A signal that is exactly a factor combination carries no residual alpha."""
    factors = _synthetic_factors()
    rng = np.random.default_rng(3)
    signal = pd.Series(factors.to_numpy() @ rng.normal(size=factors.shape[1]))
    residual = residualize_signal(signal, factors)
    assert np.abs(residual.to_numpy()).max() < 1e-8


def test_orthogonal_signal_survives_residualization() -> None:
    """A signal orthogonal to the factors is left essentially unchanged."""
    factors = _synthetic_factors(seed=10)
    # Build a signal orthogonal to the factor span via its own residual.
    rng = np.random.default_rng(4)
    raw = pd.Series(rng.normal(size=len(factors)))
    orthogonal = residualize_signal(raw, factors)
    again = residualize_signal(orthogonal, factors)
    pd.testing.assert_series_equal(again, orthogonal, check_names=False, atol=1e-8)


def test_weighted_residual_is_orthogonal_under_weights() -> None:
    """Weighted residualization satisfies the weighted normal equations."""
    factors = _synthetic_factors(seed=11)
    rng = np.random.default_rng(5)
    signal = pd.Series(rng.normal(size=len(factors)))
    weights = pd.Series(rng.uniform(0.5, 2.0, size=len(factors)))
    residual = residualize_signal(signal, factors, weights=weights, add_intercept=True)
    r = residual.to_numpy()
    w = weights.to_numpy()
    for col in factors.columns:
        assert abs(np.sum(w * factors[col].to_numpy() * r)) < 1e-6
    assert abs(np.sum(w * r)) < 1e-6  # orthogonal to the (weighted) intercept


def test_nan_signal_rows_get_nan_residual_without_corrupting_fit() -> None:
    """A NaN signal yields a NaN residual; the rest stays orthogonal."""
    factors = _synthetic_factors(seed=12)
    rng = np.random.default_rng(6)
    signal = pd.Series(rng.normal(size=len(factors)))
    signal.iloc[:10] = np.nan
    residual = residualize_signal(signal, factors)
    assert residual.iloc[:10].isna().all()
    assert _max_abs_factor_correlation(residual, factors) < 1e-8


def test_rank_deficient_design_is_handled() -> None:
    """Intercept + all GICS dummies (collinear) still gives an orthogonal residual."""
    n = 120
    rng = np.random.default_rng(7)
    sectors = rng.integers(0, 11, size=n)
    dummies = pd.get_dummies(sectors, prefix="gics").astype(float)  # 11 collinear cols
    signal = pd.Series(rng.normal(size=n))
    # add_intercept=True makes the design rank-deficient (dummies sum to the intercept).
    residual = residualize_signal(signal, dummies, add_intercept=True)
    assert _max_abs_factor_correlation(residual, dummies) < 1e-8
