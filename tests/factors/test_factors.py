"""Tests for alphacore.factors — KN factor matrices -> per-date exposures.

The official KN model ships one wide ``date x ticker`` CSV per factor (the five
styles, market, the 11 GICS industries, and the idiosyncratic return). Each cell is
that factor's return contribution for a name on a day. For the §5.4 neutralization
the strategy needs, on a rebalancing date, the ``ticker x factor`` design matrix; for
the §5.9 attribution it needs the same contributions weighted by the portfolio.

Acceptance
----------
- ``factor_exposures_at`` assembles the per-date ticker-by-factor design matrix.
- Tickers missing in any factor on that date are dropped (a complete design only).
"""

from __future__ import annotations

import pandas as pd

from alphacore.factors import factor_exposures_at, gics_sector_map


def _factor_frame(rows: dict[str, list[float]]) -> pd.DataFrame:
    """Build a date-indexed, ticker-columned factor matrix from in-memory rows."""
    tickers = ["AAA", "BBB", "CCC"]
    frame = pd.DataFrame.from_dict(rows, orient="index", columns=tickers)
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    return frame.astype(float)


def test_factor_exposures_at_assembles_ticker_by_factor() -> None:
    """The per-date design matrix has tickers as rows and factors as columns."""
    matrices = {
        "beta": _factor_frame({"2020-01-31": [0.1, 0.2, 0.3]}),
        "size": _factor_frame({"2020-01-31": [1.0, 2.0, 3.0]}),
    }
    x = factor_exposures_at(matrices, pd.Timestamp("2020-01-31"))
    assert list(x.index) == ["AAA", "BBB", "CCC"]
    assert list(x.columns) == ["beta", "size"]
    assert x.loc["BBB", "size"] == 2.0
    assert x.loc["CCC", "beta"] == 0.3


def test_factor_exposures_drops_tickers_missing_in_any_factor() -> None:
    """A ticker absent (NaN) in any factor on that date is dropped from the design."""
    matrices = {
        "beta": _factor_frame({"2020-01-31": [0.1, 0.2, float("nan")]}),
        "size": _factor_frame({"2020-01-31": [1.0, 2.0, 3.0]}),
    }
    x = factor_exposures_at(matrices, pd.Timestamp("2020-01-31"))
    assert list(x.index) == ["AAA", "BBB"]  # CCC dropped (NaN beta)


def test_factor_exposures_restricts_to_requested_tickers() -> None:
    """When a universe is supplied, only those tickers appear, in order."""
    matrices = {"beta": _factor_frame({"2020-01-31": [0.1, 0.2, 0.3]})}
    x = factor_exposures_at(matrices, pd.Timestamp("2020-01-31"), tickers=["CCC", "AAA"])
    assert list(x.index) == ["CCC", "AAA"]


def test_gics_sector_map_assigns_each_ticker_its_industry() -> None:
    """A ticker maps to the one industry whose file carries its contributions."""
    # AAA & BBB are financials (non-zero in fin only), CCC is tech.
    industries = {
        "Financials": _factor_frame(
            {"2020-01-31": [0.01, 0.02, 0.0], "2020-02-28": [0.0, 0.01, 0.0]}
        ),
        "Information Technology": _factor_frame(
            {"2020-01-31": [0.0, 0.0, 0.03], "2020-02-28": [0.0, 0.0, 0.01]}
        ),
    }
    sectors = gics_sector_map(industries)
    assert sectors["AAA"] == "Financials"
    assert sectors["BBB"] == "Financials"
    assert sectors["CCC"] == "Information Technology"


def test_gics_sector_map_marks_unassigned_ticker_nan() -> None:
    """A ticker with no non-zero industry contribution is unassigned (NaN)."""
    industries = {"Financials": _factor_frame({"2020-01-31": [0.01, 0.0, 0.0]})}
    sectors = gics_sector_map(industries)
    assert sectors["AAA"] == "Financials"
    assert pd.isna(sectors["BBB"])  # all-zero -> no sector
