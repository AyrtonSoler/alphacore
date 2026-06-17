"""RED tests for signals.value — P/E, P/B, EV/EBITDA, FCF yield, PIT guard.

Acceptance
----------
- P/E with zero EPS is NaN, never +/-inf (the multiple is undefined).
- P/B with negative book value is NaN (negative equity has no meaningful multiple).
- EV/EBITDA reproduces a known worked example (Damodaran-style fixture).
- FCF yield is positive whenever free cash flow is positive.
- Every function honours point-in-time: a value at t only reflects fundamentals
  whose release_date is on or before t; later releases are masked to NaN.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from alphacore.signals.value import c_ev_ebitda, c_fcf_yield, c_pb_ratio, c_pe_ratio

_FIXTURE = Path("tests/fixtures/value/damodaran_example.parquet")


def test_pe_ratio_handles_zero_eps_as_nan_not_inf() -> None:
    """Zero (or sign-flipping) EPS yields NaN — the P/E multiple is undefined."""
    close = pd.Series([100.0, 100.0, 100.0])
    eps_ttm = pd.Series([5.0, 0.0, -2.0])
    pe = c_pe_ratio(close, eps_ttm)
    assert pe.iloc[0] == 20.0
    assert np.isnan(pe.iloc[1])  # zero EPS -> undefined, not inf
    assert np.isnan(pe.iloc[2])  # negative EPS -> undefined
    assert not np.isinf(pe).any()


def test_pb_ratio_handles_negative_book_value_returns_nan() -> None:
    """Negative book value per share has no meaningful multiple -> NaN."""
    close = pd.Series([50.0, 50.0, 50.0])
    bvps = pd.Series([25.0, 0.0, -10.0])
    pb = c_pb_ratio(close, bvps)
    assert pb.iloc[0] == 2.0
    assert np.isnan(pb.iloc[1])
    assert np.isnan(pb.iloc[2])
    assert not np.isinf(pb).any()


def test_ev_ebitda_matches_known_example() -> None:
    """EV/EBITDA reproduces the worked fixture rows exactly."""
    df = pd.read_parquet(_FIXTURE)
    result = c_ev_ebitda(df["ev"], df["ebitda_ttm"])
    pd.testing.assert_series_equal(
        result.reset_index(drop=True),
        df["expected_ev_ebitda"].reset_index(drop=True),
        check_names=False,
    )


def test_fcf_yield_is_positive_for_positive_fcf() -> None:
    """Positive free cash flow over positive market cap gives a positive yield."""
    fcf_ttm = pd.Series([200.0, 50.0, 0.0])
    market_cap = pd.Series([4000.0, 1000.0, 1000.0])
    fcfy = c_fcf_yield(fcf_ttm, market_cap)
    assert (fcfy.iloc[:2] > 0).all()
    assert fcfy.iloc[0] == 0.05
    assert fcfy.iloc[2] == 0.0


def test_fcf_yield_handles_zero_market_cap_as_nan() -> None:
    """Zero/negative market cap is undefined -> NaN, never inf."""
    fcf_ttm = pd.Series([200.0, 200.0])
    market_cap = pd.Series([0.0, -1000.0])
    fcfy = c_fcf_yield(fcf_ttm, market_cap)
    assert fcfy.isna().all()
    assert not np.isinf(fcfy).any()


def test_all_functions_propagate_pit() -> None:
    """A signal at date t must not see fundamentals released after t."""
    idx = pd.to_datetime(["2026-01-31", "2026-02-28", "2026-03-31"])
    close = pd.Series([100.0, 100.0, 100.0], index=idx)
    eps_ttm = pd.Series([5.0, 5.0, 5.0], index=idx)
    # The Jan and Mar fundamentals are public on/before their dates; the Feb one
    # is not released until March, so it must be masked at the Feb index date.
    release_date = pd.Series(
        pd.to_datetime(["2026-01-15", "2026-03-10", "2026-03-20"]),
        index=idx,
    )
    pe = c_pe_ratio(close, eps_ttm, release_date=release_date)
    assert pe.iloc[0] == 20.0  # released 2026-01-15 <= 2026-01-31
    assert np.isnan(pe.iloc[1])  # released 2026-03-10 > 2026-02-28 -> masked
    assert pe.iloc[2] == 20.0  # released 2026-03-20 <= 2026-03-31

    # The guard is shared by every value function via the same release_date arg.
    pb = c_pb_ratio(close, eps_ttm, release_date=release_date)
    ev_eb = c_ev_ebitda(close, eps_ttm, release_date=release_date)
    fcfy = c_fcf_yield(close, eps_ttm, release_date=release_date)
    for series in (pb, ev_eb, fcfy):
        assert np.isnan(series.iloc[1])
        assert not np.isnan(series.iloc[0])
        assert not np.isnan(series.iloc[2])
