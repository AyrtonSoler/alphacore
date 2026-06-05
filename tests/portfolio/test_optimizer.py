"""RED tests for portfolio.optimizer — convex portfolio construction (v3 §5.8).

Replaces v2's accidental inverse-vol sizing with a convex program that maximises
expected residual alpha subject to factor-neutrality vs the benchmark, sector
bands, the 12.5% cap, a liquidity cap and turnover/variance penalties. The
factor-neutrality constraint is the *portfolio-level* defence of the residual
alpha (the signal-level defence is the §5.4 neutralization).

Acceptance
----------
- Weights are long-only, within the per-name cap, and sum to the invested budget
  (1 - cash).
- With loose constraints the solution tilts toward high-alpha names.
- Active factor exposure |B'(w - w_b)| and active sector weights stay within their
  bands by construction.
- A name with thin liquidity is held below its ADV-based cap.
- A large turnover penalty pulls the solution toward the previous weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio.optimizer import OptimizerInputs, OptimizerParams, optimize_portfolio

_N = 20
_NAMES = [f"S{i:02d}" for i in range(_N)]
_SECTORS = pd.Series(["A", "B", "C", "D"] * 5, index=_NAMES)
_AUM = 100_000_000.0


def _setup(seed: int = 0) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    return {
        "alpha": pd.Series(rng.normal(0, 0.02, _N), index=_NAMES),
        "cov": pd.DataFrame(np.eye(_N) * 0.04, index=_NAMES, columns=_NAMES),
        "factors": pd.DataFrame(
            rng.normal(size=(_N, 3)), index=_NAMES, columns=["beta", "size", "value"]
        ),
        "benchmark": pd.Series(np.full(_N, 1.0 / _N), index=_NAMES),
        "adv": pd.Series(np.full(_N, 1e9), index=_NAMES),  # deep unless overridden
    }


def _optimize(
    s: dict[str, object], *, cash: float = 0.0, prev: object = None, params: object = None
) -> pd.Series:
    inputs = OptimizerInputs(
        alpha=s["alpha"],
        covariance=s["cov"],
        factor_exposures=s["factors"],
        benchmark_weights=s["benchmark"],
        previous_weights=s["benchmark"] if prev is None else prev,
        sectors=_SECTORS,
        adv=s["adv"],
        cash=cash,
        aum=_AUM,
    )
    return optimize_portfolio(inputs, params or OptimizerParams())


def test_weights_sum_to_invested_budget() -> None:
    """Invested weight equals 1 - cash (the regime cash sits out)."""
    w = _optimize(_setup(), cash=0.05)
    assert np.isclose(w.sum(), 0.95, atol=1e-4)


def test_weights_are_long_only_and_capped() -> None:
    """Every weight is in [0, cap] — long-only, no leverage, name cap enforced."""
    w = _optimize(_setup(), params=OptimizerParams(cap_per_name=0.125))
    assert (w >= -1e-6).all()
    assert (w <= 0.125 + 1e-6).all()


def test_tilts_toward_high_alpha() -> None:
    """With identity risk and loose bands, weight correlates with alpha."""
    s = _setup()
    # Loosen factor/sector bands so the alpha tilt is what drives weights.
    params = OptimizerParams(eps_factor=10.0, eps_sector=10.0, gamma_risk=0.1)
    w = _optimize(s, params=params)
    assert np.corrcoef(w.to_numpy(), s["alpha"].to_numpy())[0, 1] > 0.3


def test_factor_neutrality_within_band() -> None:
    """Active factor exposure |B'(w - w_b)| stays within eps_factor."""
    s = _setup()
    w = _optimize(s, params=OptimizerParams(eps_factor=0.05))
    active = w - s["benchmark"]
    exposure = s["factors"].to_numpy().T @ active.to_numpy()
    assert (np.abs(exposure) <= 0.05 + 1e-4).all()


def test_sector_bands_within_limit() -> None:
    """Active sector weight stays within eps_sector for every GICS sector."""
    s = _setup()
    w = _optimize(s, params=OptimizerParams(eps_sector=0.03))
    active = w - s["benchmark"]
    for sector in _SECTORS.unique():
        in_sector = _SECTORS.eq(sector)
        assert abs(active[in_sector].sum()) <= 0.03 + 1e-4


def test_liquidity_cap_limits_thin_name() -> None:
    """A name with thin ADV is held below kappa * ADV / AUM."""
    s = _setup(seed=1)
    kappa = 0.10
    s["adv"] = s["adv"].copy()
    s["adv"].iloc[0] = 5_000_000.0  # thin -> cap at 0.10 * 5e6 / 1e8 = 0.005
    params = OptimizerParams(kappa_liquidity=kappa, eps_factor=10.0, eps_sector=10.0)
    w = _optimize(s, params=params)
    assert w.iloc[0] <= kappa * s["adv"].iloc[0] / _AUM + 1e-5


def test_turnover_penalty_pulls_toward_previous() -> None:
    """A large turnover penalty keeps the solution close to the previous weights."""
    s = _setup(seed=2)
    prev = s["benchmark"].copy()
    loose = {"eps_factor": 10.0, "eps_sector": 10.0}
    low_tau = _optimize(s, prev=prev, params=OptimizerParams(tau_turnover=0.0, **loose))
    high_tau = _optimize(s, prev=prev, params=OptimizerParams(tau_turnover=5.0, **loose))
    assert (high_tau - prev).abs().sum() < (low_tau - prev).abs().sum()
