"""RED tests for signals.pipeline — per-ticker signal computation (data wiring).

Bridges the loaded curator frame (common.panel) to the pure signal functions
(quality, momentum, risk). It builds the year-over-year "previous" fundamentals
(annual values are carried daily, so a 252-session shift recovers the prior filing),
applies the point-in-time guard via release_date, and emits one tidy per-ticker
frame of named signals — the input the cross-sectional stages (§5.3/§5.4) consume.

Acceptance
----------
- The output carries the expected family-prefixed signal columns.
- Momentum is defined once its formation window has filled.
- Quality signals honour point-in-time: NaN before the fundamentals are public.
- Realized volatility is non-negative where defined.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from signals.pipeline import compute_ticker_signals

_N = 320  # enough for the 12-1 momentum and 252-day risk windows to fill


def _synthetic_frame(release_date: pd.Timestamp) -> pd.DataFrame:
    idx = pd.bdate_range("2019-01-01", periods=_N)
    rng = np.random.default_rng(7)
    close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, _N))), index=idx)
    # Fundamentals carried daily, stepping up once midway (a new annual filing).
    step = np.where(np.arange(_N) < _N // 2, 1.0, 1.1)
    return pd.DataFrame(
        {
            "close": close,
            "gross_profit": 300.0 * step,
            "revenue": 1000.0 * step,
            "net_income": 100.0 * step,
            "total_assets": 2000.0 * step,
            "long_term_debt": 400.0 * step,
            "current_assets": 800.0 * step,
            "current_liabilities": 500.0 * step,
            "shares_outstanding": 50.0 * np.where(np.arange(_N) < _N // 2, 1.0, 0.98),
            "cash_flow_operations": 150.0 * step,
            "release_date": release_date,
        },
        index=idx,
    )


def test_produces_expected_signal_columns() -> None:
    """The per-ticker frame exposes the quality, momentum and risk signals."""
    frame = _synthetic_frame(pd.Timestamp("2019-01-01"))
    signals = compute_ticker_signals(frame)
    expected = {
        "quality_gross_profitability",
        "quality_accruals",
        "quality_asset_growth",
        "quality_net_issuance",
        "quality_piotroski",
        "momentum_roc12",
        "momentum_roc6",
        "momentum_trend",
        "risk_realized_vol",
        "risk_drawdown",
    }
    assert expected <= set(signals.columns)
    assert signals.index.equals(frame.index)


def test_momentum_defined_after_warmup() -> None:
    """12-1 momentum is NaN during warm-up and defined once the window fills."""
    frame = _synthetic_frame(pd.Timestamp("2019-01-01"))
    signals = compute_ticker_signals(frame)
    assert np.isnan(signals["momentum_roc12"].iloc[0])
    assert not np.isnan(signals["momentum_roc12"].iloc[-1])


def test_quality_respects_point_in_time() -> None:
    """Quality signals are NaN before the fundamentals' release date."""
    release = pd.Timestamp("2019-12-31")  # public only late in the sample
    frame = _synthetic_frame(release)
    signals = compute_ticker_signals(frame)
    gp = signals["quality_gross_profitability"]
    before = frame.index < release
    after = frame.index >= release
    assert gp[before].isna().all()
    assert gp[after].notna().any()


def test_realized_vol_is_non_negative() -> None:
    """Volatility is non-negative wherever it is defined."""
    frame = _synthetic_frame(pd.Timestamp("2019-01-01"))
    vol = compute_ticker_signals(frame)["risk_realized_vol"].dropna()
    assert (vol >= 0).all()
