"""Momentum signals — trend (SMA cross), cross-sectional return (ROC 12-1), RSI.

Momentum is the most-replicated anomaly in the asset-pricing literature: past
relative winners keep outperforming past losers over 3-12 month horizons. This
module exposes the three momentum primitives the research lab consumes, all as
pure functions of a single ``close`` price series so the Data Curator can
register them by their ``c_`` prefix.

Notes
-----
The series are indexed by *trading days* (see :mod:`alphacore._utils.time`), so window
lengths are expressed in sessions, not calendar days: a "month" is ~21 sessions
and a "year" ~252. These conventional values live in the module constants below
rather than as inline literals.

References
----------
.. [JT1993] Jegadeesh, N. and Titman, S. (1993). "Returns to Buying Winners and
   Selling Losers: Implications for Stock Market Efficiency." Journal of Finance
   48(1), 65-91.
.. [Wilder1978] Wilder, J. W. (1978). New Concepts in Technical Trading Systems.
   Trend Research. (Original definition of the RSI and its smoothing.)
.. [AMP2013] Asness, C. S., Moskowitz, T. J. and Pedersen, L. H. (2013). "Value
   and Momentum Everywhere." Journal of Finance 68(3), 929-985. (The 12-1
   formation window that skips the most recent month to avoid short-term
   reversal.)

"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Trading-session conventions (see module Notes; sessions, not calendar days).
_SESSIONS_PER_MONTH = 21
_SESSIONS_PER_HALF_YEAR = 126
_SESSIONS_PER_YEAR = 252
_SMA_FAST = 50
_SMA_SLOW = 200
_RSI_PERIOD = 14


def c_sma_50d(close: pd.Series[float]) -> pd.Series[float]:
    """50-session simple moving average of closing prices.

    Parameters
    ----------
    close:
        Closing prices indexed by trading day, ascending.

    Returns
    -------
    pandas.Series
        Trailing 50-session arithmetic mean. The first 49 entries are NaN
        because the window is not yet full.

    Notes
    -----
    The 50-day average is the standard medium-term trend proxy; paired with the
    200-day average it defines the golden/death cross in
    :func:`c_sma_50_200_signal`.

    References
    ----------
    See [JT1993]_ for the medium-horizon momentum motivation.

    """
    return close.rolling(window=_SMA_FAST).mean()


def c_sma_200d(close: pd.Series[float]) -> pd.Series[float]:
    """200-session simple moving average of closing prices.

    Parameters
    ----------
    close:
        Closing prices indexed by trading day, ascending.

    Returns
    -------
    pandas.Series
        Trailing 200-session arithmetic mean. The first 199 entries are NaN
        because the window is not yet full.

    Notes
    -----
    The 200-day average is the canonical long-term trend filter separating bull
    from bear regimes.

    References
    ----------
    See [JT1993]_ for the trend-following momentum motivation.

    """
    return close.rolling(window=_SMA_SLOW).mean()


def c_sma_50_200_signal(close: pd.Series[float]) -> pd.Series[int]:
    """Golden/death-cross trend signal from the 50- vs 200-session averages.

    Parameters
    ----------
    close:
        Closing prices indexed by trading day, ascending.

    Returns
    -------
    pandas.Series
        Integer trit per day: ``+1`` when the fast (50) average is above the
        slow (200) average (uptrend / golden cross), ``-1`` when it is below
        (downtrend / death cross), and ``0`` when they are equal or either
        average is undefined (warm-up period).

    Notes
    -----
    Encoding the cross as a sign keeps the signal scale-free and composable with
    other momentum signals. The warm-up window collapses to a neutral ``0``
    rather than NaN so downstream weighting never has to special-case it.

    References
    ----------
    See [JT1993]_ for trend persistence underpinning the cross.

    """
    fast = c_sma_50d(close)
    slow = c_sma_200d(close)
    # NaN comparisons are False on both sides, so the warm-up window yields 0.
    return (fast > slow).astype(int) - (fast < slow).astype(int)


def c_roc_12m(close: pd.Series[float]) -> pd.Series[float]:
    """12-1 momentum: trailing 12-month return skipping the most recent month.

    Parameters
    ----------
    close:
        Closing prices indexed by trading day, ascending.

    Returns
    -------
    pandas.Series
        Rate of change measured from the price ~12 months ago to the price ~1
        month ago: ``close[t-21] / close[t-252] - 1``. NaN until both lagged
        prices are available.

    Notes
    -----
    The most recent month is deliberately excluded ("12-1" formation) because
    1-month returns exhibit short-term reversal that contaminates the momentum
    signal. Consequently the value at ``t`` depends only on prices at or before
    ``t-21`` and never on the trailing month — making the signal usable without
    look-ahead at rebalancing.

    References
    ----------
    Formation window from [AMP2013]_; momentum effect from [JT1993]_.

    """
    formation_start = close.shift(_SESSIONS_PER_YEAR)
    formation_end = close.shift(_SESSIONS_PER_MONTH)
    return formation_end / formation_start - 1.0


def c_roc_6m(close: pd.Series[float]) -> pd.Series[float]:
    """6-1 momentum: trailing 6-month return skipping the most recent month.

    Parameters
    ----------
    close:
        Closing prices indexed by trading day, ascending.

    Returns
    -------
    pandas.Series
        Rate of change measured from the price ~6 months ago to the price ~1
        month ago: ``close[t-21] / close[t-126] - 1``. NaN until both lagged
        prices are available.

    Notes
    -----
    A shorter-horizon companion to :func:`c_roc_12m` used as confirmation rather
    than the primary momentum signal. It shares the "skip the last month" (``-1``)
    convention so the most recent month's short-term reversal never contaminates
    it, keeping the value at ``t`` free of look-ahead at rebalancing.

    References
    ----------
    Formation window from [AMP2013]_; momentum effect from [JT1993]_.

    """
    formation_start = close.shift(_SESSIONS_PER_HALF_YEAR)
    formation_end = close.shift(_SESSIONS_PER_MONTH)
    return formation_end / formation_start - 1.0


def c_rsi_14d(close: pd.Series[float]) -> pd.Series[float]:
    """14-session Relative Strength Index using Wilder's smoothing.

    Parameters
    ----------
    close:
        Closing prices indexed by trading day, ascending.

    Returns
    -------
    pandas.Series
        RSI bounded in [0, 100]. Values near 70 flag overbought conditions and
        near 30 oversold. NaN during the initial warm-up.

    Notes
    -----
    Average gains and losses are smoothed with Wilder's recursive moving average
    (an EMA with ``alpha = 1/period``). RSI is computed as
    ``100 * avg_gain / (avg_gain + avg_loss)`` — algebraically identical to the
    textbook ``100 - 100/(1+RS)`` but free of the ``avg_loss == 0`` division by
    zero: a flat or strictly-rising window has zero average loss and resolves to
    100 without raising. The denominator is masked to NaN only when both averages
    are zero (a perfectly flat window), which is undefined.

    References
    ----------
    Definition and smoothing from [Wilder1978]_.

    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    alpha = 1.0 / _RSI_PERIOD
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

    denominator = (avg_gain + avg_loss).replace(0.0, np.nan)
    return 100.0 * avg_gain / denominator
