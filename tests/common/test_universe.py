"""RED tests for common.universe — the KN600 investable universe loader.

The Track-1 universe was narrowed to the historical KN600 components, listed in
``KN_Hack_Investable_Universe_KN_600.xlsx`` (sheet "Stock Picking"): one row per
name with its FMP ticker, name, ISIN, sector and industry.

Acceptance
----------
- The universe loads to a frame indexed by ticker with a sector column.
- The real file holds the expected number of names and well-known tickers.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from common.universe import load_investable_universe

_REAL = Path("docs/stock-picking-official-info/KN_Hack_Investable_Universe_KN_600.xlsx")


def _write_universe(path: Path) -> None:
    pd.DataFrame(
        {
            "fmp_ticker": ["AAPL", "MSFT", "JPM"],
            "name": ["Apple", "Microsoft", "JPMorgan"],
            "isin": ["US1", "US2", "US3"],
            "sector": ["Technology", "Technology", "Financial Services"],
            "industry": ["Consumer Electronics", "Software", "Banks"],
        }
    ).to_excel(path, sheet_name="Stock Picking", index=False)


def test_universe_indexed_by_ticker_with_sector(tmp_path: Path) -> None:
    """The universe loads indexed by ticker, carrying the sector column."""
    p = tmp_path / "universe.xlsx"
    _write_universe(p)
    universe = load_investable_universe(p)
    assert universe.index.name == "ticker"
    assert list(universe.index) == ["AAPL", "MSFT", "JPM"]
    assert universe.loc["JPM", "sector"] == "Financial Services"


@pytest.mark.slow
@pytest.mark.skipif(not _REAL.exists(), reason="universe file absent (e.g. CI)")
def test_real_universe_smoke() -> None:
    """The real KN600 investable universe loads with its ~786 names."""
    universe = load_investable_universe(_REAL)
    assert len(universe) > 700
    assert "AAPL" in universe.index
