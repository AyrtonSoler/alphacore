"""Exact factor attribution — residual alpha as the primary metric (v3 §5.9).

The official KN model provides, per name/day, the return *contribution* of each
factor ``f_k[t, i]`` and the idiosyncratic return ``f_idyo[t, i]``. Portfolio
attribution is therefore **exact** — a weighted sum, with no Fama-MacBeth beta
estimation — and decomposes cleanly:

    alpha_residual(t)       = sum_i w_i(t) * f_idyo[t, i]
    contrib_factor_k(t)     = sum_i w_i(t) * f_k[t, i]
    total_factor_model(t)   = alpha_residual(t) + sum_k contrib_factor_k(t)

The residual alpha is not a post-hoc diagnostic: it is the number the v3 strategy
maximises by construction (§5.4 signal neutralization + §5.8 portfolio factor
neutrality), reported with its t-statistic (the design target is ``t > 2``). The
factor contributions should hover near zero — evidence the neutralization worked.

References
----------
.. [BLP2014] Bailey, D. H. and Lopez de Prado, M. (2014). "The Deflated Sharpe
   Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality."
   Journal of Portfolio Management 40(5), 94-107. (t-stat / Sharpe inference.)

"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

# A sample standard deviation (ddof=1) needs at least two observations.
_MIN_OBSERVATIONS = 2


def residual_alpha(
    weights: pd.DataFrame,
    idiosyncratic_returns: pd.DataFrame,
) -> pd.Series[float]:
    """Portfolio residual alpha per date: the weighted idiosyncratic return.

    Parameters
    ----------
    weights:
        Portfolio weights, one row per rebalancing date, one column per name.
    idiosyncratic_returns:
        The factor model's idiosyncratic return ``f_idyo[t, i]``, aligned to
        ``weights`` (date index, name columns).

    Returns
    -------
    pandas.Series
        ``sum_i w_i(t) * f_idyo[t, i]`` indexed by date — the return no factor
        explains, i.e. the residual alpha the strategy targets.

    """
    return (weights * idiosyncratic_returns).sum(axis=1)


def factor_contributions(
    weights: pd.DataFrame,
    factor_returns: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Per-factor portfolio return contribution per date.

    Parameters
    ----------
    weights:
        Portfolio weights, one row per date, one column per name.
    factor_returns:
        Mapping from factor name to its per name/day return contribution
        ``f_k[t, i]`` (date index, name columns), each aligned to ``weights``.

    Returns
    -------
    pandas.DataFrame
        One column per factor, indexed by date, where column ``k`` is
        ``sum_i w_i(t) * f_k[t, i]``. These should sit near zero for a factor-
        neutral portfolio — the proof that neutralization held.

    """
    return pd.DataFrame(
        {
            name: (weights * contribution).sum(axis=1)
            for name, contribution in factor_returns.items()
        },
    )


def alpha_t_statistic(alpha: pd.Series[float]) -> float:
    """Newey-free t-statistic of a residual-alpha series: sqrt(n) * mean / std.

    Parameters
    ----------
    alpha:
        The residual-alpha time series from :func:`residual_alpha`.

    Returns
    -------
    float
        ``sqrt(n) * mean / sample_std`` (``ddof=1``). The v3 acceptance bar for a
        genuine, non-factor strategy is ``t > 2``. Returns 0.0 for a zero-mean or
        zero-dispersion series.

    Notes
    -----
    This is the i.i.d. t-statistic of the mean. Serial-correlation-robust standard
    errors (Newey-West) are a later refinement; the design target ``t > 2`` is read
    off this baseline.

    References
    ----------
    Significance testing of backtested means from [BLP2014]_.

    """
    clean = alpha.dropna()
    n = len(clean)
    if n < _MIN_OBSERVATIONS:
        return 0.0
    std = float(clean.std(ddof=1))
    if std == 0.0:
        return 0.0
    return float(np.sqrt(n) * clean.mean() / std)
