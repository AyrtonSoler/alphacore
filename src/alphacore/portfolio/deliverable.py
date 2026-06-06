"""Deliverable shaping for the judges' backtest engine.

The judges load a "signals dataframe" — one row per rebalance date, one column per
ticker, each cell a target weight — into their backtest engine. Two adjustments make
our pipeline output match the contract:

- every investable-universe name appears as a column (zeros for unheld names), so a
  zero weight is visibly a signal output, not a hard-coded exclusion (the PDF's
  philosophical rule);
- an explicit all-cash row at the official 2017 start documents the pre-strategy
  warm-up window (the KN factor model and 12-1 momentum are unavailable before 2018).

The frame entering these helpers is already in Backtest-Engine layout: a ``date_column``
sentinel header plus one column per held ticker (see
``run_portfolio_construction._to_be_format``).
"""

from __future__ import annotations

import pandas as pd

_DATE_COLUMN = "date_column"


def expand_to_universe(wide: pd.DataFrame, universe: list[str]) -> pd.DataFrame:
    """Reindex weight columns to the full investable universe, zero-filling unheld names.

    Parameters
    ----------
    wide:
        Backtest-engine layout: a ``date_column`` plus one column per held ticker.
    universe:
        The full investable-universe tickers. The output carries exactly these
        (deduplicated and sorted) as weight columns, regardless of which were held.

    Returns
    -------
    pandas.DataFrame
        ``date_column`` first, then the sorted universe columns; names that were never
        held are ``0.0``. Idempotent.

    Raises
    ------
    ValueError
        If a held ticker is absent from ``universe`` — a universe/data mismatch we
        surface rather than silently drop.

    """
    held = [c for c in wide.columns if c != _DATE_COLUMN]
    extra = sorted(set(held) - set(universe))
    if extra:
        raise ValueError(f"held tickers outside the investable universe: {extra}")
    cols = sorted(set(universe))
    out = wide.set_index(_DATE_COLUMN).reindex(columns=cols).fillna(0.0)
    return out.reset_index()


def prepend_cash_row(wide: pd.DataFrame, start_date: str) -> pd.DataFrame:
    """Prepend one all-cash (all-zero) rebalance row dated ``start_date``.

    Parameters
    ----------
    wide:
        A frame already in deliverable layout (``date_column`` + weight columns).
    start_date:
        The official backtest start (ISO ``YYYY-MM-DD``). The injected row holds 100%
        cash — every weight is ``0.0`` — documenting the warm-up window before the
        strategy has a valid factor-neutralized signal.

    Returns
    -------
    pandas.DataFrame
        ``wide`` with a single leading ``start_date`` row, sorted by date. A no-op when
        a row for ``start_date`` is already present.

    """
    if (wide[_DATE_COLUMN] == start_date).any():
        return wide.sort_values(_DATE_COLUMN).reset_index(drop=True)
    weight_cols = [c for c in wide.columns if c != _DATE_COLUMN]
    cash = pd.DataFrame([{_DATE_COLUMN: start_date, **{c: 0.0 for c in weight_cols}}])
    return (
        pd.concat([cash, wide], ignore_index=True).sort_values(_DATE_COLUMN).reset_index(drop=True)
    )


def to_deliverable_format(wide: pd.DataFrame, universe: list[str], start_date: str) -> pd.DataFrame:
    """Shape a Backtest-Engine wide frame into the judges' deliverable.

    Expands the columns to the full investable universe, then prepends the explicit
    2017 cash warm-up row.

    Parameters
    ----------
    wide:
        Backtest-engine layout (``date_column`` + held-ticker columns).
    universe:
        Full investable-universe tickers (becomes the column set, sorted).
    start_date:
        Official backtest start date for the leading cash row (ISO ``YYYY-MM-DD``).

    Returns
    -------
    pandas.DataFrame
        Deliverable layout: ``date_column``, then every universe name; one leading
        all-cash row then the strategy's rebalance rows.

    """
    return prepend_cash_row(expand_to_universe(wide, universe), start_date)
