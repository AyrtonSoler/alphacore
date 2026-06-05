"""Regime overlay — a continuous, defensive risk-on signal (v3 §5.7).

The overlay reincorporates the market-timing defence the v2 draft had dropped. It
is deliberately **soft and continuous**: a sigmoid of standardized macro inputs
maps to a risk-on probability in (0, 1), which a linear rule turns into a cash
weight between ``cash_min`` and ``cash_max``. A continuous overlay avoids the
whipsaw of a binary 50% switch and keeps tracking error against the long-only
KN600 benchmark controlled, while still defending in selloffs (2018-Q4, 2020,
2022).

The market-timing return this produces is **not** one of the five static KN style
factors, so it is a legitimate, orthogonal source of return rather than a hidden
factor exposure.

All inputs are standardized over a **trailing** window (point-in-time: only data
available at the decision date enters), so the signal never looks ahead.

References
----------
.. [Lo2004] Lo, A. W. (2004). "The Adaptive Markets Hypothesis: Market Efficiency
   from an Evolutionary Perspective." Journal of Portfolio Management 30(5), 15-29.
   (Time-varying risk premia motivating regime-conditional exposure.)

"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd


@dataclasses.dataclass(frozen=True)
class RegimeParams:
    """Pre-registered tunables for the regime overlay (§5.7, §7).

    Bundling the configuration keeps :func:`c_market_regime_signal` to its four
    market-data inputs and makes the pre-registered values a single, typed,
    immutable object rather than scattered magic numbers.

    Attributes
    ----------
    z_window:
        Trailing window over which each macro input is standardized.
    trend_sma_window:
        Window of the moving average defining the trend gauge.
    weight_vix, weight_curve, weight_trend:
        Non-negative blend weights ``a``, ``b``, ``c`` of §5.7.

    """

    z_window: int = 252
    trend_sma_window: int = 200
    weight_vix: float = 1.0
    weight_curve: float = 1.0
    weight_trend: float = 1.0


# Frozen, so a single shared default instance is safe to reuse as a parameter default.
_DEFAULT_REGIME_PARAMS = RegimeParams()


def _trailing_z(series: pd.Series[float], window: int) -> pd.Series[float]:
    """Trailing (point-in-time) z-score; flat windows score 0, warm-up is NaN."""
    mean = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    # Replace a zero standard deviation with NaN before dividing to avoid 0/0, then
    # set those flat (no-dispersion) positions to a neutral 0. Warm-up rows keep a
    # NaN std (not 0), so they remain NaN.
    std_nonzero = std.where(std != 0.0)
    z = (series - mean) / std_nonzero
    result: pd.Series[float] = z.mask(std == 0.0, 0.0)
    return result


def c_market_regime_signal(
    vix: pd.Series[float],
    spy_close: pd.Series[float],
    yield_10y: pd.Series[float],
    yield_2y: pd.Series[float],
    params: RegimeParams = _DEFAULT_REGIME_PARAMS,
) -> pd.Series[float]:
    """Continuous risk-on probability from VIX, curve slope and trend.

    Parameters
    ----------
    vix:
        Implied-volatility index (or realized SPY vol proxy), indexed by date.
    spy_close:
        Benchmark closing level, aligned to ``vix``, used for the trend gauge.
    yield_10y, yield_2y:
        Treasury yields whose difference is the curve slope, aligned to ``vix``.
    params:
        Pre-registered :class:`RegimeParams` (windows and blend weights).

    Returns
    -------
    pandas.Series
        ``sigmoid( a*(-vix_z) + b*curve_z + c*trend_z )`` in (0, 1): the
        probability the market is risk-on. High VIX lowers it; a steep curve and a
        price above its trailing average raise it. NaN until every input's trailing
        window is full.

    Notes
    -----
    The trend gauge is the standardized *relative distance* from the trailing
    average (``spy/sma - 1``) rather than a raw above/below boolean, so the overlay
    stays continuous — the design intent of §5.7.

    References
    ----------
    Regime-conditional exposure motivated by [Lo2004]_.

    """
    curve = yield_10y - yield_2y
    sma = spy_close.rolling(window=params.trend_sma_window).mean()
    trend_distance = spy_close / sma - 1.0

    score = (
        params.weight_vix * (-_trailing_z(vix, params.z_window))
        + params.weight_curve * _trailing_z(curve, params.z_window)
        + params.weight_trend * _trailing_z(trend_distance, params.z_window)
    )

    # Logistic squashing into (0, 1); NaN in score (warm-up) propagates to NaN.
    sigmoid = 1.0 / (1.0 + np.exp(-score.to_numpy()))
    result: pd.Series[float] = pd.Series(sigmoid, index=score.index, dtype=float)
    return result


def c_regime_cash_weight(
    risk_on: pd.Series[float],
    cash_min: float = 0.01,
    cash_max: float = 0.35,
) -> pd.Series[float]:
    """Map a risk-on probability to a portfolio cash weight (§5.7).

    Parameters
    ----------
    risk_on:
        Risk-on probability in [0, 1] from :func:`c_market_regime_signal`.
    cash_min, cash_max:
        Pre-registered cash band. Defaults: 1% floor, 35% defensive cap
        (``cash_max`` is tuned by information ratio within [0.30, 0.40] and fixed
        ex-ante, never chosen ex-post).

    Returns
    -------
    pandas.Series
        ``cash_min + (cash_max - cash_min) * (1 - risk_on)``: ``cash_min`` when
        fully risk-on, ``cash_max`` when fully risk-off, linear in between and
        always within ``[cash_min, cash_max]``.

    """
    result: pd.Series[float] = cash_min + (cash_max - cash_min) * (1.0 - risk_on)
    return result
