"""Factor neutralization — residualize a signal against the factor model (§5.4).

The core of v3. Each composite signal is regressed **cross-sectionally** on the
design matrix of factor exposures (the five KN styles + 11 GICS dummies, with the
market/country factor entering as the intercept) and replaced by its residual:

    beta_hat = argmin_b (s - X b)' W (s - X b)
    s_tilde  = s - X beta_hat = (I - H) s,   H = X (X'WX)^+ X'W

By the (weighted) normal equations ``X' W s_tilde = 0``, so the residual is
orthogonal to every factor and every sector dummy: the residual ranking expresses
the spread *within* the factors, not the factors themselves. This is the signal-
level half of the two-layer defence of the residual alpha (the portfolio-level
half is the optimizer's factor-neutrality constraint, §5.8).

The fit uses a least-squares (pseudo-inverse) projection, so a rank-deficient
design — e.g. the dummy-variable trap of an intercept plus all 11 GICS dummies — is
handled gracefully: the projection onto the column span is unique regardless of how
the span is parameterised, so the residual is well-defined and still orthogonal.

Names whose signal is NaN are excluded from the fit (so missing data cannot corrupt
the betas) and receive a NaN residual, honouring the v3 NaN policy (§3): missingness
is never imputed.

References
----------
.. [Fama1973] Fama, E. F. and MacBeth, J. D. (1973). "Risk, Return, and
   Equilibrium: Empirical Tests." Journal of Political Economy 81(3), 607-636.
   (Cross-sectional regression of returns/signals on exposures.)

"""

from __future__ import annotations

import numpy as np
import pandas as pd


def residualize_signal(
    signal: pd.Series[float],
    factor_exposures: pd.DataFrame,
    weights: pd.Series[float] | None = None,
    *,
    add_intercept: bool = True,
) -> pd.Series[float]:
    """Residualize a cross-sectional signal against factor exposures.

    Parameters
    ----------
    signal:
        The composite signal for one rebalancing cross-section, indexed by name.
    factor_exposures:
        Factor / sector-dummy exposures, one row per name aligned to ``signal``,
        one column per factor. Proportional KN return contributions may be passed
        directly — the projection is invariant to per-column scaling.
    weights:
        Optional regression weights per name (e.g. sqrt market cap). Default: equal
        weights.
    add_intercept:
        When True (default), prepend a constant column so the residual is also
        orthogonal to the intercept (mean-zero), representing the market/country
        factor of §5.4.

    Returns
    -------
    pandas.Series
        The residual ``s - X beta_hat`` aligned to ``signal.index``. Orthogonal to
        every factor column (and the intercept) under the chosen weights. Names with
        a NaN signal, NaN weight or NaN in any factor exposure receive NaN.

    Notes
    -----
    The fit whitens by ``sqrt(weights)`` and solves the resulting ordinary least
    squares with :func:`numpy.linalg.lstsq` (``rcond=None``), which returns the
    minimum-norm solution for rank-deficient designs while leaving the residual —
    the unique projection onto the column span — well-defined.

    References
    ----------
    Cross-sectional exposure regression in the spirit of [Fama1973]_.

    """
    s = signal.to_numpy(dtype=float)
    x = factor_exposures.to_numpy(dtype=float)
    if add_intercept:
        x = np.column_stack([np.ones(len(s)), x])
    w = np.ones(len(s)) if weights is None else weights.to_numpy(dtype=float)

    # Only fully-observed rows enter the fit, so missing data never moves the betas.
    complete = np.isfinite(s) & np.isfinite(w) & np.isfinite(x).all(axis=1)

    residual = np.full(len(s), np.nan)
    if complete.any():
        x_c, s_c, w_c = x[complete], s[complete], w[complete]
        root_w = np.sqrt(w_c)
        # Whitened OLS: minimise ||sqrt(W)(s - X b)||^2.
        beta, *_ = np.linalg.lstsq(root_w[:, None] * x_c, root_w * s_c, rcond=None)
        residual[complete] = s_c - x_c @ beta

    return pd.Series(residual, index=signal.index, dtype=float)
