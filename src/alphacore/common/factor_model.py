"""KN factor-model loaders — wide factor CSVs to per-date design matrices.

The official KN model ships one wide ``date x ticker`` CSV per factor (the five
styles beta/size/value/residual-volatility/momentum, the market factor, the 11 GICS
industries, and the idiosyncratic return ``f_idyo_returns``). Each cell is that
factor's return *contribution* for a name on a day.

Two consumers:

- §5.4 neutralization needs, on a rebalancing date, the ``ticker x factor`` design
  matrix (regress each signal on it and keep the residual). Because the regression
  is scale-invariant per column, the contributions serve directly as regressors —
  no need to recover the underlying exposures.
- §5.9 attribution weights those same contributions by the portfolio.

The wide CSVs are large (hundreds of MB each); these loaders read one factor file at
a time so memory stays bounded to the factors actually needed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import pandas as pd


def load_factor_matrix(path: Path | str) -> pd.DataFrame:
    """Load one wide factor CSV into a date-indexed, ticker-columned float frame.

    Parameters
    ----------
    path:
        Path to a ``f_*.csv`` factor file (leading unnamed date column, then one
        column per ticker).

    Returns
    -------
    pandas.DataFrame
        Indexed by date (DatetimeIndex), one float column per ticker, holding that
        factor's per name/day return contribution.

    """
    raw = pd.read_csv(path, index_col=0)
    raw.index = pd.DatetimeIndex(pd.to_datetime(raw.index))
    return raw.astype(float)


def factor_exposures_at(
    matrices: Mapping[str, pd.DataFrame],
    date: pd.Timestamp,
    tickers: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Assemble the per-date ticker-by-factor design matrix.

    Parameters
    ----------
    matrices:
        Mapping from factor name to its loaded matrix (from
        :func:`load_factor_matrix`), each indexed by date with ticker columns.
    date:
        The rebalancing date whose cross-section to extract.
    tickers:
        Optional universe to restrict (and order) the rows to. When omitted, all
        tickers common to the factors are used.

    Returns
    -------
    pandas.DataFrame
        Rows indexed by ticker, one column per factor (in ``matrices`` order),
        holding each name's factor contributions on ``date``. Rows with a missing
        value in any factor are dropped so the design is complete.

    """
    columns = {name: matrix.loc[date] for name, matrix in matrices.items()}
    design = pd.DataFrame(columns)
    if tickers is not None:
        design = design.reindex(index=list(tickers))
    return design.dropna()


def gics_sector_map(industry_matrices: Mapping[str, pd.DataFrame]) -> pd.Series[str]:
    """Derive each ticker's GICS sector from the 11 industry contribution files.

    Parameters
    ----------
    industry_matrices:
        Mapping from GICS sector name to its loaded industry matrix (from
        :func:`load_factor_matrix`). A name has non-zero contributions in exactly
        one industry — its own — across the sample.

    Returns
    -------
    pandas.Series
        Indexed by ticker, the sector name with the largest absolute total
        contribution. Tickers with no non-zero contribution in any industry (never
        a member) are NaN.

    Notes
    -----
    Membership is read from the whole sample, not a single day: an industry factor's
    contribution can be zero on a given date even for its own members, so summing the
    absolute contribution over time identifies the sole industry robustly.

    """
    totals = pd.DataFrame(
        {sector: matrix.abs().sum(axis=0) for sector, matrix in industry_matrices.items()}
    )
    # idxmax over string columns returns the column labels (sector names); the
    # pandas stub mistypes it as Series[int], so cast to the true Series[str].
    sectors = cast("pd.Series[str]", totals.idxmax(axis=1))
    never_member = totals.sum(axis=1) == 0.0
    return sectors.mask(never_member)
