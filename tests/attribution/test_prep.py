"""Tests for attribution.prep — shaping KN artifacts for the attribution loader.

Acceptance
----------
- assets_from_weights returns the ticker columns of a BE-format weights file.
- slim_factor_frame keeps only intersecting assets and names the date index ``date``;
  idempotent.
- slim_factor_csv round-trips to a file whose first column header is ``date``.
- prepare_factor_returns writes one ``<factor>.csv`` per requested factor.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from attribution.prep import (
    assets_from_weights,
    prepare_factor_returns,
    slim_factor_csv,
    slim_factor_frame,
)


def _factor_frame() -> pd.DataFrame:
    # Date-indexed factor matrix with an unnamed index (as the KN files have).
    return pd.DataFrame(
        {"AAA": [0.01, 0.02], "BBB": [-0.01, 0.0], "ZZZ": [0.03, 0.03]},
        index=["2018-03-28", "2018-03-29"],
    )


def test_assets_from_weights_returns_ticker_columns(tmp_path: Path) -> None:
    wide = pd.DataFrame({"date_column": ["2017-01-03"], "AAA": [0.0], "BBB": [0.99]})
    path = tmp_path / "w.csv"
    wide.to_csv(path, index=False)
    assert assets_from_weights(path) == ["AAA", "BBB"]


def test_slim_factor_frame_keeps_intersection_and_names_date() -> None:
    out = slim_factor_frame(_factor_frame(), ["BBB", "AAA"])
    assert list(out.columns) == ["AAA", "BBB"]  # original order preserved
    assert "ZZZ" not in out.columns
    assert out.index.name == "date"


def test_slim_factor_frame_is_idempotent() -> None:
    once = slim_factor_frame(_factor_frame(), ["AAA", "BBB"])
    twice = slim_factor_frame(once, ["AAA", "BBB"])
    pd.testing.assert_frame_equal(once, twice)


def test_slim_factor_csv_writes_named_date_header(tmp_path: Path) -> None:
    src = tmp_path / "f_beta.csv"
    _factor_frame().to_csv(src)  # writes an unnamed first header cell
    out = slim_factor_csv(src, ["AAA"], tmp_path / "out" / "f_beta.csv")
    back = pd.read_csv(out)
    assert back.columns[0] == "date"
    assert list(back.columns) == ["date", "AAA"]


def test_prepare_factor_returns_writes_one_file_per_factor(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    for name in ("f_beta", "f_momentum"):
        _factor_frame().to_csv(src / f"{name}.csv")
    out_dir = tmp_path / "factors"
    written = prepare_factor_returns(
        {"f_beta": src / "f_beta.csv", "f_momentum": src / "f_momentum.csv"},
        ["AAA", "BBB"],
        out_dir,
    )
    assert {p.name for p in written} == {"f_beta.csv", "f_momentum.csv"}
    assert (out_dir / "f_beta.csv").is_file()
    assert (out_dir / "f_momentum.csv").is_file()
