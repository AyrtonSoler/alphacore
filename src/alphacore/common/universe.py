"""Investable-universe loader — the historical KN600 components (Track 1).

The Track-1 universe was narrowed to the historical KN600 constituents, listed in
``KN_Hack_Investable_Universe_KN_600.xlsx`` (sheet "Stock Picking"). This loader
turns that file into a ticker-indexed frame; the index is the selectable universe
and the ``sector`` / ``industry`` columns are available metadata.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_SHEET = "Stock Picking"
_TICKER_COLUMN = "fmp_ticker"


def load_investable_universe(path: Path | str) -> pd.DataFrame:
    """Load the KN600 investable universe into a ticker-indexed frame.

    Parameters
    ----------
    path:
        Path to ``KN_Hack_Investable_Universe_KN_600.xlsx``.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``ticker`` (the FMP ticker), with ``name``, ``isin``, ``sector``
        and ``industry`` columns. The index is the selectable universe.

    """
    frame = pd.read_excel(path, sheet_name=_SHEET)
    return frame.rename(columns={_TICKER_COLUMN: "ticker"}).set_index("ticker")
