"""RED tests for common.factor_model — KN factor matrices -> per-date exposures.

The official KN model ships one wide ``date x ticker`` CSV per factor (the five
styles, market, the 11 GICS industries, and the idiosyncratic return). Each cell is
that factor's return contribution for a name on a day. For the §5.4 neutralization
the strategy needs, on a rebalancing date, the ``ticker x factor`` design matrix; for
the §5.9 attribution it needs the same contributions weighted by the portfolio.

Acceptance
----------
- A factor CSV loads to a date-indexed, ticker-columned float matrix.
- ``factor_exposures_at`` assembles the per-date ticker-by-factor design matrix.
- Tickers missing in any factor on that date are dropped (a complete design only).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common.factor_model import factor_exposures_at, load_factor_matrix


def _write_factor_csv(path: Path, rows: dict[str, list[float]]) -> None:
    # Leading empty header cell + ticker columns, matching the KN file layout.
    tickers = ["AAA", "BBB", "CCC"]
    lines = [",".join(["", *tickers])]
    for date, values in rows.items():
        lines.append(",".join([date, *(str(v) for v in values)]))
    path.write_text("\n".join(lines) + "\n")


def test_load_factor_matrix_is_date_indexed_floats(tmp_path: Path) -> None:
    """A factor CSV loads to a date-indexed, ticker-columned float frame."""
    p = tmp_path / "f_beta.csv"
    _write_factor_csv(p, {"2018-03-28": [1.0, 2.0, 3.0], "2018-03-29": [4.0, 5.0, 6.0]})
    matrix = load_factor_matrix(p)
    assert isinstance(matrix.index, pd.DatetimeIndex)
    assert list(matrix.columns) == ["AAA", "BBB", "CCC"]
    assert matrix.to_numpy().dtype == np.float64
    assert matrix.loc[pd.Timestamp("2018-03-28"), "BBB"] == 2.0


def test_factor_exposures_at_assembles_ticker_by_factor(tmp_path: Path) -> None:
    """The per-date design matrix has tickers as rows and factors as columns."""
    beta = tmp_path / "f_beta.csv"
    size = tmp_path / "f_size.csv"
    _write_factor_csv(beta, {"2020-01-31": [0.1, 0.2, 0.3]})
    _write_factor_csv(size, {"2020-01-31": [1.0, 2.0, 3.0]})
    matrices = {"beta": load_factor_matrix(beta), "size": load_factor_matrix(size)}
    x = factor_exposures_at(matrices, pd.Timestamp("2020-01-31"))
    assert list(x.index) == ["AAA", "BBB", "CCC"]
    assert list(x.columns) == ["beta", "size"]
    assert x.loc["BBB", "size"] == 2.0
    assert x.loc["CCC", "beta"] == 0.3


def test_factor_exposures_drops_tickers_missing_in_any_factor(tmp_path: Path) -> None:
    """A ticker absent (NaN) in any factor on that date is dropped from the design."""
    beta = tmp_path / "f_beta.csv"
    size = tmp_path / "f_size.csv"
    _write_factor_csv(beta, {"2020-01-31": [0.1, 0.2, float("nan")]})
    _write_factor_csv(size, {"2020-01-31": [1.0, 2.0, 3.0]})
    matrices = {"beta": load_factor_matrix(beta), "size": load_factor_matrix(size)}
    x = factor_exposures_at(matrices, pd.Timestamp("2020-01-31"))
    assert list(x.index) == ["AAA", "BBB"]  # CCC dropped (NaN beta)


def test_factor_exposures_restricts_to_requested_tickers(tmp_path: Path) -> None:
    """When a universe is supplied, only those tickers appear, in order."""
    beta = tmp_path / "f_beta.csv"
    _write_factor_csv(beta, {"2020-01-31": [0.1, 0.2, 0.3]})
    matrices = {"beta": load_factor_matrix(beta)}
    x = factor_exposures_at(matrices, pd.Timestamp("2020-01-31"), tickers=["CCC", "AAA"])
    assert list(x.index) == ["CCC", "AAA"]
