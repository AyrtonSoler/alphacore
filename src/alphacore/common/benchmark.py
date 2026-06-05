"""KN600 benchmark loaders — daily constituent weights and return series.

The KN600 is provided as data (Track 1 rules). Two files feed the strategy:

- ``index_daily_holdings_*.csv`` — a wide ``date x ticker`` matrix of daily
  constituent weights (each row sums to 1). This is the optimizer's ``w_bench`` and
  the reference for active weights and sector bands (§5.8).
- ``kn600_returns.csv`` — the daily benchmark return series, for the backtest's
  relative mandate and the attribution's active return (§5.9).

The two files use different date conventions — holdings in ISO ``YYYY-MM-DD``,
returns in day-first ``DD/MM/YYYY`` — which these loaders normalise to a
DatetimeIndex so downstream code never has to care.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_DATE_COLUMN = "m_date"
_RETURNS_COLUMN = "kn600_returns"
_RETURNS_DATE_FORMAT = "%d/%m/%Y"


def load_benchmark_weights(path: Path | str) -> pd.DataFrame:
    """Load the KN600 daily constituent weights.

    Parameters
    ----------
    path:
        Path to ``index_daily_holdings_*.csv`` (ISO ``YYYY-MM-DD`` dates).

    Returns
    -------
    pandas.DataFrame
        A wide frame indexed by date, one float column per ticker, where each row is
        that day's benchmark weights (summing to ~1).

    """
    raw = pd.read_csv(path)
    index = pd.DatetimeIndex(pd.to_datetime(raw[_DATE_COLUMN]))
    weights = raw.drop(columns=[_DATE_COLUMN]).astype(float)
    weights.index = index
    return weights


def load_benchmark_returns(path: Path | str) -> pd.Series[float]:
    """Load the KN600 daily return series.

    Parameters
    ----------
    path:
        Path to ``kn600_returns.csv`` (day-first ``DD/MM/YYYY`` dates).

    Returns
    -------
    pandas.Series
        Daily benchmark returns indexed by date, named ``kn600_returns``.

    """
    raw = pd.read_csv(path)
    index = pd.DatetimeIndex(pd.to_datetime(raw[_DATE_COLUMN], format=_RETURNS_DATE_FORMAT))
    return pd.Series(
        raw[_RETURNS_COLUMN].astype(float).to_numpy(),
        index=index,
        name=_RETURNS_COLUMN,
    )
