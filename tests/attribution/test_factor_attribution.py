"""RED tests for attribution.factor_attribution — residual alpha as metric (§5.9).

Because the official KN model hands us, per name/day, the return *contributions* of
each factor plus the idiosyncratic return, the portfolio attribution is exact (no
Fama-MacBeth beta estimation needed): it is a weighted sum. The residual alpha is
the v3 **primary objective metric**, reported with its t-stat (target t > 2).

Acceptance
----------
- Residual alpha at t is the portfolio-weighted idiosyncratic return.
- Each factor's contribution at t is the portfolio-weighted factor contribution.
- The decomposition closes: residual alpha + sum of factor contributions equals the
  portfolio's total factor-model return (mirrors the exact identity
  total_excess = total_factor + idio).
- A factor-neutral portfolio (zero active exposure) contributes ~0 from that factor.
- The alpha t-statistic is sqrt(n) * mean / sample-std.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from attribution.factor_attribution import (
    alpha_t_statistic,
    factor_contributions,
    residual_alpha,
)

_DATES = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"])
_NAMES = ["AAA", "BBB", "CCC", "DDD"]


def _panel(values: list[list[float]]) -> pd.DataFrame:
    return pd.DataFrame(values, index=_DATES, columns=_NAMES)


def test_residual_alpha_is_weighted_idiosyncratic_return() -> None:
    """Residual alpha at each date is sum_i w_i * idio_i."""
    weights = _panel([[0.25, 0.25, 0.25, 0.25]] * 3)
    idio = _panel(
        [
            [0.02, -0.01, 0.00, 0.03],
            [0.01, 0.01, -0.02, 0.00],
            [-0.03, 0.04, 0.01, -0.02],
        ]
    )
    alpha = residual_alpha(weights, idio)
    expected = (weights * idio).sum(axis=1)
    pd.testing.assert_series_equal(alpha, expected, check_names=False)
    assert np.isclose(alpha.iloc[0], 0.25 * (0.02 - 0.01 + 0.00 + 0.03))


def test_factor_contributions_are_weighted_per_factor() -> None:
    """Each factor's contribution series is sum_i w_i * f_k[t, i]."""
    weights = _panel([[0.5, 0.5, 0.0, 0.0]] * 3)
    f_size = _panel([[0.01, 0.02, 0.0, 0.0]] * 3)
    f_value = _panel([[-0.01, 0.03, 0.0, 0.0]] * 3)
    contribs = factor_contributions(weights, {"size": f_size, "value": f_value})
    assert list(contribs.columns) == ["size", "value"]
    assert np.isclose(contribs["size"].iloc[0], 0.5 * 0.01 + 0.5 * 0.02)
    assert np.isclose(contribs["value"].iloc[0], 0.5 * -0.01 + 0.5 * 0.03)


def test_decomposition_closes_to_total_factor_model_return() -> None:
    """residual alpha + sum of factor contributions == weighted total return."""
    rng = np.random.default_rng(0)
    weights = _panel(rng.dirichlet(np.ones(4), size=3).tolist())  # rows sum to 1
    idio = _panel(rng.normal(0, 0.02, size=(3, 4)).tolist())
    f1 = _panel(rng.normal(0, 0.02, size=(3, 4)).tolist())
    f2 = _panel(rng.normal(0, 0.02, size=(3, 4)).tolist())
    total = idio + f1 + f2  # total_excess = total_factor + idio, by construction

    alpha = residual_alpha(weights, idio)
    contribs = factor_contributions(weights, {"f1": f1, "f2": f2})
    reconstructed = alpha + contribs.sum(axis=1)
    expected_total = (weights * total).sum(axis=1)
    pd.testing.assert_series_equal(reconstructed, expected_total, check_names=False)


def test_factor_neutral_portfolio_contributes_zero() -> None:
    """A portfolio with zero active factor exposure contributes ~0 from it."""
    # Long/short-style weights that cancel against an antisymmetric exposure.
    weights = _panel([[0.5, 0.5, -0.5, -0.5]] * 3)
    f = _panel([[1.0, 1.0, 1.0, 1.0]] * 3)  # sum_i w_i f_i = 0.5+0.5-0.5-0.5 = 0
    contribs = factor_contributions(weights, {"f": f})
    assert np.allclose(contribs["f"].to_numpy(), 0.0, atol=1e-12)


def test_alpha_t_statistic_matches_formula() -> None:
    """t = sqrt(n) * mean / sample-std (ddof=1)."""
    alpha = pd.Series([0.01, 0.02, -0.005, 0.015, 0.008, 0.012])
    n = len(alpha)
    expected = np.sqrt(n) * alpha.mean() / alpha.std(ddof=1)
    assert np.isclose(alpha_t_statistic(alpha), expected)


def test_alpha_t_statistic_zero_for_zero_mean() -> None:
    """A zero-mean alpha series has a zero t-statistic."""
    alpha = pd.Series([0.01, -0.01, 0.02, -0.02])
    assert np.isclose(alpha_t_statistic(alpha), 0.0)
