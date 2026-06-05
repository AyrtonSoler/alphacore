"""Convex portfolio optimizer (v3 §5.8).

Replaces v2's ``score / vol`` sizing — which was an accidental inverse-vol weighting
(opiniones #2/#3) — with a disciplined convex program. Per rebalancing date it
maximises expected residual alpha net of risk and turnover, subject to the official
constraints and a portfolio-level factor-neutrality band:

    maximise   w'a  -  gamma * w'Sigma w  -  tau * ||w - w_prev||_1
    s.t.       sum_i w_i = 1 - cash                       (budget; regime cash sits out)
               0 <= w_i <= cap                            (long-only, 12.5% name cap)
               | B'(w - w_b) | <= eps_factor              (factor-neutral vs benchmark)
               | sum_{i in g}(w_i - w_b,i) | <= eps_sector (sector bands vs KN600)
               w_i <= kappa * ADV_i / AUM                 (liquidity: fillable position)

The factor-neutrality constraint is the **portfolio-level** half of the two-layer
defence of the residual alpha (the signal-level half is §5.4 neutralization): even if
a residual signal correlated with a factor in return space, the optimizer drives the
portfolio's active factor exposure to ~0, so the attribution (§5.9) shows style
contributions near zero *by design*.

The box cap resolves the 12.5% "waterfall" natively (no manual redistribution), and
the ``tau`` turnover term replaces v2's Top-30/45 buffer with a principled cost.
Market impact and the min-15 cardinality floor are layered on in follow-ups.

References
----------
.. [Markowitz1952] Markowitz, H. (1952). "Portfolio Selection." Journal of Finance
   7(1), 77-91. (Mean-variance objective.)
.. [LedoitWolf2004] Ledoit, O. and Wolf, M. (2004). "Honey, I Shrunk the Sample
   Covariance Matrix." Journal of Portfolio Management 30(4), 110-119. (The shrunk
   Sigma fed to the risk term.)

"""

from __future__ import annotations

import dataclasses

import cvxpy as cp
import numpy as np
import pandas as pd


@dataclasses.dataclass(frozen=True)
class OptimizerParams:
    """Pre-registered optimizer configuration (§5.8, §7)."""

    cap_per_name: float = 0.125
    eps_factor: float = 0.05
    eps_sector: float = 0.03
    gamma_risk: float = 1.0
    tau_turnover: float = 0.0
    kappa_liquidity: float = 0.10
    min_holdings: int = 15


@dataclasses.dataclass(frozen=True)
class OptimizerInputs:
    """One rebalancing date's optimization instance.

    Bundling the per-name data and scalars keeps :func:`optimize_portfolio` to two
    arguments and makes the instance a single typed object. All ``Series`` /
    ``DataFrame`` share the same name index/columns.
    """

    alpha: pd.Series[float]
    covariance: pd.DataFrame
    factor_exposures: pd.DataFrame
    benchmark_weights: pd.Series[float]
    previous_weights: pd.Series[float]
    sectors: pd.Series[str]
    adv: pd.Series[float]
    cash: float
    aum: float


_DEFAULT_PARAMS = OptimizerParams()


def optimize_portfolio(
    inputs: OptimizerInputs,
    params: OptimizerParams = _DEFAULT_PARAMS,
) -> pd.Series[float]:
    """Solve the convex portfolio program for one rebalancing date.

    Parameters
    ----------
    inputs:
        The optimization instance (alpha, covariance, factor exposures, benchmark
        and previous weights, sectors, ADV, cash and AUM).
    params:
        Pre-registered :class:`OptimizerParams`.

    Returns
    -------
    pandas.Series
        Optimal long-only weights indexed by name, summing to ``1 - cash``, within
        the per-name cap, factor- and sector-neutral within their bands, and below
        each name's liquidity cap.

    Raises
    ------
    ValueError
        If the program is infeasible or the solver returns no solution.

    Notes
    -----
    Solved with the deterministic CLARABEL conic solver. Tiny negative weights from
    solver tolerance are clipped to 0; the budget is preserved to solver accuracy.

    References
    ----------
    Mean-variance objective from [Markowitz1952]_; shrunk covariance from
    [LedoitWolf2004]_.

    """
    names = inputs.alpha.index
    n = len(names)
    a = inputs.alpha.to_numpy(dtype=float)
    sigma = inputs.covariance.to_numpy(dtype=float)
    exposures = inputs.factor_exposures.to_numpy(dtype=float)  # n x k
    w_bench = inputs.benchmark_weights.to_numpy(dtype=float)
    w_prev = inputs.previous_weights.to_numpy(dtype=float)
    adv = inputs.adv.to_numpy(dtype=float)
    liquidity_cap = params.kappa_liquidity * adv / inputs.aum

    w = cp.Variable(n)
    active = w - w_bench
    objective = (
        a @ w
        - params.gamma_risk * cp.quad_form(w, cp.psd_wrap(sigma))
        - params.tau_turnover * cp.norm1(w - w_prev)
    )

    constraints = [
        cp.sum(w) == 1.0 - inputs.cash,
        w >= 0.0,
        w <= params.cap_per_name,
        cp.abs(exposures.T @ active) <= params.eps_factor,
        w <= liquidity_cap,
    ]
    # Sector bands via 0/1 selector dot products (avoids fancy-indexing a cvxpy var).
    for sector in pd.unique(inputs.sectors):
        selector = (inputs.sectors == sector).to_numpy(dtype=float)
        constraints.append(cp.abs(selector @ active) <= params.eps_sector)

    problem = cp.Problem(cp.Maximize(objective), constraints)
    problem.solve(solver=cp.CLARABEL)

    if w.value is None:
        msg = f"optimizer returned no solution (status={problem.status!r})"
        raise ValueError(msg)

    weights = np.clip(np.asarray(w.value, dtype=float), 0.0, None)
    return pd.Series(weights, index=names, dtype=float)
