"""Curator panel loader — map raw curator parquet to the signal taxonomy.

The Data Curator writes one parquet per ticker with ~200 columns, numeric fields as
Python ``Decimal`` objects and dates as ``datetime.date``. This module is the I/O
boundary that turns one such file into a clean, float-typed, date-indexed frame
whose columns carry the **single semantic taxonomy** the v3 signals consume (§5.2,
which mandates one taxonomy shared by code and config). Every downstream stage —
per-ticker signals, cross-sectional z-score, neutralization — builds on this frame.

The mapping lives here as one authoritative constant so the signal functions never
hard-code curator column names. Values are coerced with ``pd.to_numeric(...,
errors="coerce")`` (Decimal -> float, missing -> NaN); the fundamental
``release_date`` is parsed to datetime so the point-in-time guard
(:func:`signals._common.apply_pit`) can mask not-yet-public figures.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .io import read_parquet

# Curator date column and the fundamental publication date used by the PIT guard.
_DATE_COLUMN = "m_date"
_RELEASE_DATE_COLUMN = "f_filing_date"

# Semantic name -> raw curator column. The single taxonomy the signals consume.
FIELD_MAP: dict[str, str] = {
    "close": "m_close",
    "volume": "m_volume",
    "vwap": "m_vwap",
    "gross_profit": "fis_gross_profit",
    "revenue": "fis_revenues",
    "net_income": "fis_net_income",
    "total_assets": "fbs_assets",
    "long_term_debt": "fbs_longterm_debt",
    "current_assets": "fbs_current_assets",
    "current_liabilities": "fbs_current_liabilities",
    "shares_outstanding": "fis_weighted_average_diluted_shares_outstanding",
    "cash_flow_operations": "fcf_net_cash_from_operating_activities",
}


def load_ticker_frame(path: Path | str) -> pd.DataFrame:
    """Load one ticker's curator parquet into a clean, date-indexed signal frame.

    Parameters
    ----------
    path:
        Path to a per-ticker curator parquet (e.g. ``Data_Curator/AAPL.parquet``).

    Returns
    -------
    pandas.DataFrame
        Indexed by a sorted DatetimeIndex parsed from ``m_date``. Columns are the
        semantic field names present in :data:`FIELD_MAP` (coerced to float64,
        missing values as NaN) plus ``release_date`` (datetime) for the PIT guard.
        Unmapped curator columns are dropped.

    Notes
    -----
    Only mapped fields actually present in the file are included, so the loader is
    robust to schema drift across providers. ``pd.to_numeric(..., errors="coerce")``
    turns the curator's ``Decimal`` objects into floats and any unparseable value
    into NaN, never raising.

    """
    raw = read_parquet(path)
    index = pd.DatetimeIndex(pd.to_datetime(raw[_DATE_COLUMN]))

    # .to_numpy() so the new DatetimeIndex aligns positionally instead of pandas
    # reindexing the source Series' RangeIndex (which would yield all-NaN).
    columns = {
        semantic: pd.to_numeric(raw[curator_column], errors="coerce").to_numpy()
        for semantic, curator_column in FIELD_MAP.items()
        if curator_column in raw.columns
    }
    frame = pd.DataFrame(columns, index=index)
    if _RELEASE_DATE_COLUMN in raw.columns:
        frame["release_date"] = pd.to_datetime(raw[_RELEASE_DATE_COLUMN]).to_numpy()

    return frame.sort_index()
