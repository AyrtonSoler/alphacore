"""RED tests for common.benchmark — KN600 holdings and returns loaders.

The KN600 benchmark is provided as data: daily constituent weights (the optimizer's
``w_bench`` and the active-weight reference) and a daily return series (for the
backtest/attribution). The two files use different date conventions — holdings in
ISO ``YYYY-MM-DD``, returns in ``DD/MM/YYYY`` — which the loaders normalise.

Acceptance
----------
- Holdings load to a date-indexed wide frame of float weights summing to ~1 per day.
- Returns load to a date-indexed float series, parsed with the DD/MM/YYYY convention
  (a day-of-month > 12 must parse as the day, not the month).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from common.benchmark import load_benchmark_returns, load_benchmark_weights

_REAL_HOLDINGS = Path("docs/stock-picking-official-info/Benchmark/index_daily_holdings_2017.csv")
_REAL_RETURNS = Path("docs/stock-picking-official-info/Benchmark/kn600_returns.csv")


def _write_holdings(path: Path) -> None:
    path.write_text("m_date,AAA,BBB,CCC\n2017-01-03,0.5,0.3,0.2\n2017-01-04,0.4,0.4,0.2\n")


def _write_returns(path: Path) -> None:
    # DD/MM/YYYY: 31/01 is the 31st of January, not an impossible 31st month.
    path.write_text("m_date,kn600_returns\n05/01/2015,-0.0178\n31/01/2015,0.0123\n")


def test_holdings_indexed_by_date_with_float_weights(tmp_path: Path) -> None:
    """Weights load as a date-indexed wide float frame."""
    p = tmp_path / "holdings.csv"
    _write_holdings(p)
    weights = load_benchmark_weights(p)
    assert isinstance(weights.index, pd.DatetimeIndex)
    assert weights.index[0] == pd.Timestamp("2017-01-03")
    assert list(weights.columns) == ["AAA", "BBB", "CCC"]
    assert weights.to_numpy().dtype == np.float64


def test_holdings_weights_sum_to_one_per_day(tmp_path: Path) -> None:
    """Each day's constituent weights sum to ~1."""
    p = tmp_path / "holdings.csv"
    _write_holdings(p)
    weights = load_benchmark_weights(p)
    assert np.allclose(weights.sum(axis=1).to_numpy(), 1.0)


def test_returns_indexed_by_date(tmp_path: Path) -> None:
    """Returns load to a date-indexed float series."""
    p = tmp_path / "returns.csv"
    _write_returns(p)
    returns = load_benchmark_returns(p)
    assert isinstance(returns.index, pd.DatetimeIndex)
    assert np.isclose(returns.iloc[0], -0.0178)


def test_returns_use_day_first_date_format(tmp_path: Path) -> None:
    """31/01/2015 parses as 31 January (DD/MM/YYYY), proving day-first parsing."""
    p = tmp_path / "returns.csv"
    _write_returns(p)
    returns = load_benchmark_returns(p)
    assert returns.index[1] == pd.Timestamp("2015-01-31")


@pytest.mark.slow
@pytest.mark.skipif(not _REAL_HOLDINGS.exists(), reason="benchmark data absent (e.g. CI)")
def test_real_holdings_smoke() -> None:
    """The real KN600 holdings load and each day's weights sum to ~1."""
    weights = load_benchmark_weights(_REAL_HOLDINGS)
    assert len(weights) > 100
    assert np.allclose(weights.sum(axis=1).to_numpy(), 1.0, atol=1e-3)


@pytest.mark.slow
@pytest.mark.skipif(not _REAL_RETURNS.exists(), reason="benchmark data absent (e.g. CI)")
def test_real_returns_smoke() -> None:
    """The real KN600 return series loads as a monotonic date-indexed series."""
    returns = load_benchmark_returns(_REAL_RETURNS)
    assert len(returns) > 1000
    assert returns.index.is_monotonic_increasing
