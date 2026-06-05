"""Risk signals — realized vol, idiosyncratic (CAPM) vol, drawdown (v3 §5.2).

These quantities measure risk, not edge. In v3 they are deliberately **kept out of
the alpha**: low-volatility *is* the KN residual-volatility style, so ranking on it
would re-import a factor the strategy neutralizes and double-count it (opiniones
#2). Instead they feed only:

- the regime overlay (§5.7), which raises cash when realized risk spikes; and
- the optimizer's risk model (§5.8), which penalises portfolio variance.

All three are pure functions of return / price series, indexed by trading day, so
windows are expressed in sessions (~252 per year). Annualization uses sqrt(252).

References
----------
.. [Sharpe1964] Sharpe, W. F. (1964). "Capital Asset Prices: A Theory of Market
   Equilibrium under Conditions of Risk." Journal of Finance 19(3), 425-442.
   (The CAPM market model whose residual defines idiosyncratic volatility.)
.. [AHXZ2006] Ang, A., Hodrick, R. J., Xing, Y. and Zhang, X. (2006). "The Cross-
   Section of Volatility and Expected Returns." Journal of Finance 61(1), 259-299.
   (Idiosyncratic volatility relative to the market model.)

"""

from __future__ import annotations

import numpy as np
import pandas as pd

_SESSIONS_PER_YEAR = 252
_ANNUALIZATION = float(np.sqrt(_SESSIONS_PER_YEAR))


def c_realized_vol(
    returns: pd.Series[float],
    window: int = _SESSIONS_PER_YEAR,
) -> pd.Series[float]:
    """Annualized realized volatility: trailing return standard deviation.

    Parameters
    ----------
    returns:
        Daily returns indexed by trading day, ascending.
    window:
        Trailing window length in sessions. Default 252 (one year).

    Returns
    -------
    pandas.Series
        ``returns.rolling(window).std(ddof=1) * sqrt(252)``. NaN until the window
        is full.

    Notes
    -----
    The sample standard deviation (``ddof=1``) over a trailing window, scaled to an
    annual figure by the square-root-of-time rule, is the standard realized-vol
    estimator and the headline input to the regime overlay.

    """
    return returns.rolling(window=window).std() * _ANNUALIZATION


def c_idiosyncratic_vol(
    stock_returns: pd.Series[float],
    market_returns: pd.Series[float],
    window: int = _SESSIONS_PER_YEAR,
) -> pd.Series[float]:
    """Annualized idiosyncratic volatility: the CAPM market-model residual vol.

    Parameters
    ----------
    stock_returns:
        Daily stock returns indexed by trading day, ascending.
    market_returns:
        Daily market returns aligned to ``stock_returns``.
    window:
        Trailing window length in sessions over which the market model is fit.
        Default 252.

    Returns
    -------
    pandas.Series
        Annualized standard deviation of the residual from regressing the stock on
        the market over the trailing window. NaN until the window is full.

    Notes
    -----
    Rather than fit an explicit rolling regression, this uses the closed form
    ``sigma_resid = sigma_stock * sqrt(1 - rho^2)`` where ``rho`` is the trailing
    stock-market correlation — algebraically the residual standard deviation of the
    single-factor market model. The ``1 - rho^2`` term is floored at 0 so float
    noise (``|rho|`` a hair above 1) cannot produce a NaN from a negative root.

    References
    ----------
    Market model from [Sharpe1964]_; idiosyncratic volatility from [AHXZ2006]_.

    """
    rolling_std = stock_returns.rolling(window=window).std()
    rolling_corr = stock_returns.rolling(window=window).corr(market_returns)
    # pandas ``** 0.5`` (not np.sqrt) keeps the result typed as Series[float].
    residual_fraction = (1.0 - rolling_corr**2).clip(lower=0.0) ** 0.5
    return rolling_std * residual_fraction * _ANNUALIZATION


def c_drawdown(
    close: pd.Series[float],
    window: int = _SESSIONS_PER_YEAR,
) -> pd.Series[float]:
    """Trailing drawdown: price relative to its trailing peak, minus one.

    Parameters
    ----------
    close:
        Closing prices indexed by trading day, ascending.
    window:
        Trailing window over which the running peak is taken. Default 252.

    Returns
    -------
    pandas.Series
        ``close / close.rolling(window).max() - 1``: 0 at a new trailing high and
        negative below it (e.g. -0.25 at 25% below the peak). NaN until the window
        is full.

    Notes
    -----
    Drawdown is the risk measure that matters to a relative-return mandate during
    selloffs; it informs the regime overlay's defensive cash level rather than the
    stock-selection score.

    """
    trailing_peak = close.rolling(window=window).max()
    return close / trailing_peak - 1.0
