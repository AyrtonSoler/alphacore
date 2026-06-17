"""RED tests for signals.regime — continuous defensive regime overlay (v3 §5.7).

The overlay is deliberately *soft and continuous* (a sigmoid of standardized
macro inputs), not a binary 50% switch, to avoid whipsaw and control tracking
error against the long-only KN600 benchmark. It raises cash when risk rises.

Acceptance
----------
- The risk-on probability is bounded in (0, 1) for any input (it is a sigmoid).
- All inputs use trailing (point-in-time) standardization: a higher *current* VIX
  lowers risk-on; a steeper *current* curve and a stronger uptrend raise it.
- Warm-up (before the z-score window is full) is NaN.
- The cash mapping is linear and monotone: full risk-on -> cash_min, full
  risk-off -> cash_max, and always within [cash_min, cash_max].
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphacore.signals.regime import RegimeParams, c_market_regime_signal, c_regime_cash_weight

_Z = 20
_SMA = 5
_PARAMS = RegimeParams(z_window=_Z, trend_sma_window=_SMA)


def _flat(value: float, n: int = 50) -> list[float]:
    return [value] * n


def _regime(vix: list[float], spy: list[float], y10: list[float], y2: list[float]) -> pd.Series:
    return c_market_regime_signal(
        pd.Series(vix),
        pd.Series(spy),
        pd.Series(y10),
        pd.Series(y2),
        params=_PARAMS,
    )


def test_risk_on_probability_is_bounded_unit_interval() -> None:
    """The sigmoid output never leaves (0, 1) for any finite input."""
    rng = np.random.default_rng(42)
    n = 80
    r = _regime(
        list(15.0 + rng.normal(0, 3, n)),
        list(100.0 + np.cumsum(rng.normal(0, 1, n))),
        list(3.0 + rng.normal(0, 0.2, n)),
        list(2.0 + rng.normal(0, 0.2, n)),
    ).dropna()
    assert ((r > 0.0) & (r < 1.0)).all()


def test_higher_current_vix_lowers_risk_on() -> None:
    """Holding history fixed, a higher last VIX print pushes risk-on down."""
    base_vix, spy, y10, y2 = _flat(10.0), _flat(100.0), _flat(3.0), _flat(2.0)
    calm = base_vix[:-1] + [10.0]
    spike = base_vix[:-1] + [30.0]
    r_calm = _regime(calm, spy, y10, y2).iloc[-1]
    r_spike = _regime(spike, spy, y10, y2).iloc[-1]
    assert r_spike < r_calm


def test_steeper_current_curve_raises_risk_on() -> None:
    """A steeper 10y-2y slope at the current date raises risk-on."""
    vix, spy, y2 = _flat(15.0), _flat(100.0), _flat(2.0)
    flat_y10 = _flat(3.0)  # curve = 1.0 everywhere
    steep_y10 = _flat(3.0)[:-1] + [5.0]  # curve jumps to 3.0 at the last date
    r_flat = _regime(vix, spy, flat_y10, y2).iloc[-1]
    r_steep = _regime(vix, spy, steep_y10, y2).iloc[-1]
    assert r_steep > r_flat


def test_uptrend_raises_risk_on_relative_to_downtrend() -> None:
    """A price spiking above its trailing SMA beats one dropping below it."""
    vix, y10, y2 = _flat(15.0), _flat(3.0), _flat(2.0)
    rising = _flat(100.0)[:-1] + [130.0]
    falling = _flat(100.0)[:-1] + [70.0]
    r_up = _regime(vix, rising, y10, y2).iloc[-1]
    r_down = _regime(vix, falling, y10, y2).iloc[-1]
    assert r_up > r_down


def test_regime_warmup_is_nan() -> None:
    """Before the z-score window fills, the regime is undefined (NaN)."""
    n = 30
    r = _regime(_flat(15.0, n), _flat(100.0, n), _flat(3.0, n), _flat(2.0, n))
    assert np.isnan(r.iloc[_Z - 2])  # window not yet full
    assert not np.isnan(r.iloc[-1])  # full window -> defined


def test_cash_weight_maps_risk_on_to_cash_band() -> None:
    """Full risk-on -> cash_min, full risk-off -> cash_max, linear in between."""
    risk_on = pd.Series([1.0, 0.0, 0.5])
    cash = c_regime_cash_weight(risk_on, cash_min=0.01, cash_max=0.35)
    assert np.isclose(cash.iloc[0], 0.01)  # fully risk-on -> minimum cash
    assert np.isclose(cash.iloc[1], 0.35)  # fully risk-off -> maximum cash
    assert np.isclose(cash.iloc[2], 0.18)  # halfway


def test_cash_weight_is_within_band_and_monotone() -> None:
    """Cash stays within [cash_min, cash_max] and falls as risk-on rises."""
    risk_on = pd.Series([0.0, 0.25, 0.5, 0.75, 1.0])
    cash = c_regime_cash_weight(risk_on, cash_min=0.01, cash_max=0.35)
    assert ((cash >= 0.01) & (cash <= 0.35)).all()
    assert (cash.diff().dropna() < 0).all()  # strictly decreasing in risk-on
