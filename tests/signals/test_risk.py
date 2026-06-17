"""RED tests for signals.risk — realized vol, idiosyncratic (CAPM) vol, drawdown.

These are **not** alpha sleeves (v3 removes low-vol from the alpha to stop the
double-counting of opiniones #2). They feed only the regime overlay (§5.7) and the
optimizer's risk model (§5.8), so they are computed but never ranked into the score.

Acceptance
----------
- Realized vol is the trailing return standard deviation, annualized by sqrt(252).
- Idiosyncratic vol is the CAPM residual vol: a stock that is a perfect affine
  function of the market has ~zero idiosyncratic vol, and idio vol can never
  exceed total realized vol (the residual is a component of the whole).
- Drawdown is close / trailing-peak - 1: zero at new highs, negative below them.
- Every signal is NaN during its warm-up window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from alphacore.signals.risk import c_drawdown, c_idiosyncratic_vol, c_realized_vol

_ANNUALIZE = np.sqrt(252.0)


def _random_returns(n: int, seed: int = 42, scale: float = 0.01) -> pd.Series:
    """Deterministic synthetic daily returns."""
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0, scale, size=n))


def test_realized_vol_matches_rolling_std_annualized() -> None:
    """Realized vol is exactly rolling(window).std(ddof=1) * sqrt(252)."""
    returns = _random_returns(300)
    window = 21
    expected = returns.rolling(window).std() * _ANNUALIZE
    pd.testing.assert_series_equal(
        c_realized_vol(returns, window=window), expected, check_names=False
    )


def test_realized_vol_warmup_is_nan() -> None:
    """The first window-1 observations cannot fill the window, so they are NaN."""
    returns = _random_returns(100)
    vol = c_realized_vol(returns, window=21)
    assert vol.iloc[:20].isna().all()
    assert not np.isnan(vol.iloc[20])


def test_idiosyncratic_vol_is_zero_for_pure_market_stock() -> None:
    """A stock that is an affine function of the market has ~zero idio vol."""
    market = _random_returns(300, seed=7)
    stock = 2.0 * market + 0.5  # perfectly correlated -> residual is zero
    idio = c_idiosyncratic_vol(stock, market, window=63).dropna()
    # The closed-form sqrt(1 - rho^2) loses precision as rho -> 1, leaving float
    # noise ~1e-8 — seven orders of magnitude below a typical annual vol (~0.3).
    assert np.allclose(idio.to_numpy(), 0.0, atol=1e-6)


@settings(max_examples=40, deadline=None)
@given(seed_s=st.integers(0, 10_000), seed_m=st.integers(0, 10_000))
def test_idiosyncratic_vol_never_exceeds_total_vol(seed_s: int, seed_m: int) -> None:
    """The CAPM residual is a part of total variance, so idio vol <= realized vol."""
    stock = _random_returns(200, seed=seed_s)
    market = _random_returns(200, seed=seed_m)
    window = 63
    idio = c_idiosyncratic_vol(stock, market, window=window).dropna()
    total = c_realized_vol(stock, window=window).reindex(idio.index)
    assert (idio <= total + 1e-9).all()


def test_idiosyncratic_vol_warmup_is_nan() -> None:
    """Idio vol needs a full window of paired observations before it is defined."""
    stock = _random_returns(100, seed=1)
    market = _random_returns(100, seed=2)
    idio = c_idiosyncratic_vol(stock, market, window=63)
    assert idio.iloc[:62].isna().all()
    assert not np.isnan(idio.iloc[62])


def test_drawdown_is_zero_at_new_highs() -> None:
    """A strictly rising price path is always at a new peak -> drawdown 0."""
    close = pd.Series([float(i) for i in range(1, 50)])
    dd = c_drawdown(close, window=10).dropna()
    assert np.allclose(dd.to_numpy(), 0.0)


def test_drawdown_is_negative_below_peak() -> None:
    """A price below its trailing peak shows a negative drawdown of the right size."""
    close = pd.Series([100.0, 110.0, 120.0, 90.0])  # peak 120, last is 90
    dd = c_drawdown(close, window=4)
    assert np.isclose(dd.iloc[3], 90.0 / 120.0 - 1.0)  # -0.25
    assert dd.iloc[3] < 0.0


def test_drawdown_never_positive() -> None:
    """Drawdown is bounded above by 0 by construction (price <= trailing peak)."""
    close = pd.Series(100.0 + np.cumsum(_random_returns(200).to_numpy()))
    dd = c_drawdown(close, window=21).dropna()
    assert (dd <= 1e-12).all()
