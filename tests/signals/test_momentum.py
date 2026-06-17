"""RED tests for signals.momentum — SMA, golden-cross signal, ROC 12-1, RSI.

Acceptance
----------
- SMA(50) equals pandas ``rolling(50).mean()``; the first 49 values are NaN.
- The 50/200 golden-cross signal only ever emits {-1, 0, +1}.
- ROC 12-1 at ``t`` depends only on prices at or before ``t-21`` (skips the most
  recent month), so perturbing the last 21 prices leaves the value unchanged.
- RSI(14) stays within [0, 100]; a strictly-increasing series yields RSI >= 50.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from alphacore.signals.momentum import (
    c_roc_6m,
    c_roc_12m,
    c_rsi_14d,
    c_sma_50_200_signal,
    c_sma_50d,
    c_sma_200d,
)


def _random_walk(n: int, seed: int = 42) -> pd.Series:
    """Synthetic positive price path: geometric random walk, deterministic seed."""
    rng = np.random.default_rng(seed)
    daily_returns = rng.normal(0.0, 0.01, size=n)
    prices = 100.0 * np.exp(np.cumsum(daily_returns))
    return pd.Series(prices)


def test_sma_50d_matches_pandas_rolling_mean() -> None:
    """c_sma_50d is exactly the 50-period trailing arithmetic mean."""
    close = _random_walk(300)
    expected = close.rolling(window=50).mean()
    pd.testing.assert_series_equal(c_sma_50d(close), expected, check_names=False)


def test_sma_first_49_are_nan() -> None:
    """A 50-day window needs 50 observations: indices 0..48 are NaN, 49 is defined."""
    close = _random_walk(300)
    sma = c_sma_50d(close)
    assert sma.iloc[:49].isna().all()
    assert not np.isnan(sma.iloc[49])


def test_sma_200d_first_199_are_nan() -> None:
    """A 200-day window needs 200 observations: indices 0..198 are NaN."""
    close = _random_walk(400)
    sma = c_sma_200d(close)
    assert sma.iloc[:199].isna().all()
    assert not np.isnan(sma.iloc[199])


def test_sma_50_200_signal_returns_only_neg1_zero_pos1() -> None:
    """The golden/death-cross signal is a clean trit: only -1, 0, or +1."""
    close = _random_walk(600)
    signal = c_sma_50_200_signal(close)
    assert set(signal.unique()).issubset({-1, 0, 1})


def test_roc_12m_excludes_last_month() -> None:
    """ROC at t uses prices <= t-21, so doubling the last 21 prices changes nothing."""
    close = _random_walk(400)
    t = len(close) - 1
    baseline = c_roc_12m(close).iloc[t]

    perturbed = close.copy()
    perturbed.iloc[t - 20 :] *= 2.0  # touch only indices > t-21 (the recent month)
    after = c_roc_12m(perturbed).iloc[t]

    assert not np.isnan(baseline)
    assert after == baseline


def test_roc_6m_excludes_last_month() -> None:
    """ROC 6-1 at t uses prices <= t-21, so doubling the last 21 prices changes nothing."""
    close = _random_walk(300)
    t = len(close) - 1
    baseline = c_roc_6m(close).iloc[t]

    perturbed = close.copy()
    perturbed.iloc[t - 20 :] *= 2.0  # touch only the most recent month
    after = c_roc_6m(perturbed).iloc[t]

    assert not np.isnan(baseline)
    assert after == baseline


def test_roc_6m_matches_definition() -> None:
    """ROC 6-1 equals close[t-21] / close[t-126] - 1 (6 months, skip last month)."""
    close = _random_walk(300)
    t = len(close) - 1
    expected = close.iloc[t - 21] / close.iloc[t - 126] - 1.0
    assert np.isclose(c_roc_6m(close).iloc[t], expected)


def test_rsi_14d_in_zero_to_hundred_range() -> None:
    """RSI is bounded in [0, 100] by construction for any price path."""
    close = _random_walk(300)
    rsi = c_rsi_14d(close).dropna()
    assert ((rsi >= 0.0) & (rsi <= 100.0)).all()


@settings(max_examples=50, deadline=None)
@given(
    increments=st.lists(
        st.floats(
            min_value=0.01,
            max_value=10.0,
            allow_nan=False,
            allow_infinity=False,
        ),
        min_size=30,
        max_size=300,
    )
)
def test_rsi_ge_50_for_monotonic_increasing(increments: list[float]) -> None:
    """A strictly increasing positive series has zero losses, so RSI >= 50 (==100)."""
    prices = 100.0 + np.cumsum(np.array(increments))
    close = pd.Series(prices)
    rsi = c_rsi_14d(close).dropna()
    assert (rsi >= 50.0).all()
