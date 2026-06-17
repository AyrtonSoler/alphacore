"""RED tests for portfolio.deliverable — judges' weights-file shaping.

Acceptance
----------
- expand_to_universe carries exactly the sorted universe as weight columns, zero-filling
  names that were never held, and is idempotent.
- A held ticker outside the universe raises (surfaces a universe/data mismatch).
- prepend_cash_row injects exactly one leading all-zero row at the official start date,
  and is a no-op when that date is already present.
- to_deliverable_format composes both and round-trips through .xlsx unchanged.
"""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

from alphacore.portfolio.deliverable import (
    expand_to_universe,
    prepend_cash_row,
    to_deliverable_format,
)

_DATE = "date_column"


def _wide() -> pd.DataFrame:
    # Two rebalance dates, two held names (AAA, BBB) out of a 4-name universe.
    return pd.DataFrame(
        {
            _DATE: ["2018-04-30", "2018-07-31"],
            "AAA": [0.5, 0.0],
            "BBB": [0.49, 0.99],
        }
    )


_UNIVERSE = ["BBB", "AAA", "CCC", "DDD"]


def test_expand_carries_sorted_universe_and_zero_fills() -> None:
    out = expand_to_universe(_wide(), _UNIVERSE)
    assert list(out.columns) == [_DATE, "AAA", "BBB", "CCC", "DDD"]
    assert (out["CCC"] == 0.0).all()
    assert (out["DDD"] == 0.0).all()
    assert out.loc[out[_DATE] == "2018-04-30", "AAA"].iloc[0] == pytest.approx(0.5)


def test_expand_is_idempotent() -> None:
    once = expand_to_universe(_wide(), _UNIVERSE)
    twice = expand_to_universe(once, _UNIVERSE)
    pd.testing.assert_frame_equal(once, twice)


def test_expand_raises_on_held_ticker_outside_universe() -> None:
    with pytest.raises(ValueError, match="outside the investable universe"):
        expand_to_universe(_wide(), ["AAA", "CCC"])  # BBB held but not in universe


def test_prepend_injects_single_all_zero_row_sorted_first() -> None:
    out = prepend_cash_row(expand_to_universe(_wide(), _UNIVERSE), "2017-01-03")
    assert out[_DATE].tolist() == ["2017-01-03", "2018-04-30", "2018-07-31"]
    cash = out.loc[out[_DATE] == "2017-01-03"]
    weight_cols = [c for c in out.columns if c != _DATE]
    assert (cash[weight_cols].to_numpy() == 0.0).all()
    assert (out[_DATE] == "2017-01-03").sum() == 1


def test_prepend_is_noop_when_start_date_already_present() -> None:
    base = prepend_cash_row(expand_to_universe(_wide(), _UNIVERSE), "2017-01-03")
    again = prepend_cash_row(base, "2017-01-03")
    pd.testing.assert_frame_equal(base, again)


def test_to_deliverable_format_composes_expand_and_cash() -> None:
    out = to_deliverable_format(_wide(), _UNIVERSE, "2017-01-03")
    assert list(out.columns) == [_DATE, "AAA", "BBB", "CCC", "DDD"]
    assert out[_DATE].tolist() == ["2017-01-03", "2018-04-30", "2018-07-31"]
    assert out.loc[out[_DATE] == "2018-07-31", ["AAA", "BBB", "CCC", "DDD"]].sum(axis=1).iloc[
        0
    ] == pytest.approx(0.99)
    assert out.loc[out[_DATE] == "2017-01-03", ["AAA", "BBB", "CCC", "DDD"]].sum(axis=1).iloc[
        0
    ] == pytest.approx(0.0)


def test_to_deliverable_round_trips_through_xlsx(tmp_path: pathlib.Path) -> None:
    out = to_deliverable_format(_wide(), _UNIVERSE, "2017-01-03")
    path = tmp_path / "portfolio_weights.xlsx"
    out.to_excel(path, index=False)
    back = pd.read_excel(path, dtype={_DATE: str})
    pd.testing.assert_frame_equal(
        out.reset_index(drop=True), back.reset_index(drop=True), check_dtype=False
    )
