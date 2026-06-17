"""Pure factor-design helpers: assemble per-date design matrices from in-memory factor
matrices. Split out of the original factor-model module (the CSV loader stays in the data
adapter of each consumer)."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import pandas as pd


def factor_exposures_at(
    matrices: Mapping[str, pd.DataFrame],
    date: pd.Timestamp,
    tickers: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Assemble the per-date ticker-by-factor design matrix.

    Parameters
    ----------
    matrices:
        Mapping from factor name to its matrix: a DataFrame indexed by date with
        ticker columns (as produced by the consumer's factor-matrix loader).
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
        Mapping from GICS sector name to its industry matrix: a DataFrame indexed by
        date with ticker columns (as produced by the consumer's factor-matrix loader).
        A name has non-zero contributions in exactly one industry — its own — across
        the sample.

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
