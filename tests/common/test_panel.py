"""RED tests for common.panel — curator parquet -> clean signal frame.

The Data Curator writes one parquet per ticker with ~200 columns whose numeric
fields are stored as Python ``Decimal`` objects (object dtype) and whose dates are
``datetime.date``. The panel loader maps the curator's column names to the single
semantic taxonomy the v3 signals consume (§5.2), coerces values to float, and
indexes by date — the foundation every downstream stage builds on.

Acceptance
----------
- The frame is indexed by a sorted DatetimeIndex parsed from ``m_date``.
- Decimal/object numeric columns become float64; missing values become NaN.
- Columns carry the semantic names (``close``, ``total_assets``, …), not the raw
  curator names (``m_close``, ``fbs_assets``).
- The fundamental ``release_date`` is parsed to datetime for the PIT guard.
- Unmapped curator columns are dropped.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from common.panel import load_ticker_frame

_REAL_TICKER = Path("Investment_Strategy/Data_Curator/AAPL.parquet")


def _write_curator_parquet(path: Path) -> None:
    dates = [dt.date(2020, 1, 31), dt.date(2020, 2, 28), dt.date(2020, 3, 31)]
    pd.DataFrame(
        {
            "m_date": dates,
            "m_close": [Decimal("100.5"), None, Decimal("102.0")],
            "m_volume": [1000, 2000, 1500],
            "fis_gross_profit": [Decimal("300"), Decimal("300"), Decimal("310")],
            "fis_revenues": [Decimal("1000"), Decimal("1000"), Decimal("1050")],
            "fis_net_income": [Decimal("100"), Decimal("100"), Decimal("110")],
            "fbs_assets": [Decimal("2000"), Decimal("2000"), Decimal("2100")],
            "fbs_longterm_debt": [Decimal("400"), Decimal("400"), Decimal("390")],
            "fbs_current_assets": [Decimal("800"), Decimal("800"), Decimal("820")],
            "fbs_current_liabilities": [Decimal("500"), Decimal("500"), Decimal("510")],
            "fis_weighted_average_diluted_shares_outstanding": [
                Decimal("50"),
                Decimal("50"),
                Decimal("49"),
            ],
            "fcf_net_cash_from_operating_activities": [
                Decimal("150"),
                Decimal("150"),
                Decimal("160"),
            ],
            "f_filing_date": [dt.date(2019, 12, 1)] * 3,
            "fbs_unused_padding_column": [Decimal("1"), Decimal("2"), Decimal("3")],
        }
    ).to_parquet(path)


def test_indexed_by_sorted_datetime(tmp_path: Path) -> None:
    """The frame is indexed by a sorted DatetimeIndex parsed from m_date."""
    p = tmp_path / "AAA.parquet"
    _write_curator_parquet(p)
    frame = load_ticker_frame(p)
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert frame.index.is_monotonic_increasing
    assert frame.index[0] == pd.Timestamp("2020-01-31")


def test_decimals_coerced_to_float_with_nan(tmp_path: Path) -> None:
    """Decimal/object numbers become float64; missing becomes NaN."""
    p = tmp_path / "AAA.parquet"
    _write_curator_parquet(p)
    frame = load_ticker_frame(p)
    assert frame["close"].dtype == np.float64
    assert frame["close"].iloc[0] == 100.5
    assert np.isnan(frame["close"].iloc[1])  # the None
    assert frame["total_assets"].iloc[0] == 2000.0


def test_columns_use_semantic_names(tmp_path: Path) -> None:
    """Raw curator names are renamed to the single signal taxonomy."""
    p = tmp_path / "AAA.parquet"
    _write_curator_parquet(p)
    frame = load_ticker_frame(p)
    assert {"close", "gross_profit", "total_assets", "cash_flow_operations"} <= set(frame.columns)
    assert "m_close" not in frame.columns
    assert "fbs_assets" not in frame.columns


def test_release_date_parsed_to_datetime(tmp_path: Path) -> None:
    """The fundamental release date is parsed for the point-in-time guard."""
    p = tmp_path / "AAA.parquet"
    _write_curator_parquet(p)
    frame = load_ticker_frame(p)
    assert pd.api.types.is_datetime64_any_dtype(frame["release_date"])
    assert frame["release_date"].iloc[0] == pd.Timestamp("2019-12-01")


def test_unmapped_columns_are_dropped(tmp_path: Path) -> None:
    """Columns outside the semantic map do not leak into the frame."""
    p = tmp_path / "AAA.parquet"
    _write_curator_parquet(p)
    frame = load_ticker_frame(p)
    assert "fbs_unused_padding_column" not in frame.columns


@pytest.mark.slow
@pytest.mark.skipif(
    not _REAL_TICKER.exists(), reason="curator data is gitignored / absent (e.g. CI)"
)
def test_loads_a_real_curator_file_smoke() -> None:
    """The FIELD_MAP matches the real curator schema (guards against schema drift)."""
    frame = load_ticker_frame(_REAL_TICKER)
    assert len(frame) > 1000  # ~9 years of daily rows
    assert {"close", "total_assets", "gross_profit", "cash_flow_operations"} <= set(frame.columns)
    assert frame["close"].notna().any()
    assert frame["total_assets"].notna().any()  # fundamentals are carried daily
    assert pd.api.types.is_datetime64_any_dtype(frame["release_date"])
