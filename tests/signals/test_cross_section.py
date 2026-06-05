"""RED tests for signals.cross_section — robust sector-neutral z-score (v3 §5.3).

Stage 3 turns each raw signal into a robust, sector-neutral z-score so that a
name is ranked against its GICS peers (banks vs banks, REITs vs REITs) and not
against the whole market — killing the accidental sector bet and giving leverage
ratios meaning inside financials.

Acceptance
----------
- Winsorize clips to within-sector order-statistic bounds and is **idempotent**:
  winsorizing twice equals winsorizing once (the CLAUDE.md invariant).
- Winsorize is sector-local: an outlier in one sector never moves another's bounds.
- NaN passes through every transform untouched (a missing fundamental stays missing).
- The robust z-score centres each sector at its median (median -> 0) and scales by
  1.4826 * MAD, so a symmetric sector is antisymmetric around 0.
- A sector with no dispersion (all equal, or a lone name) yields z = 0, not inf/NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from signals.cross_section import c_winsorize, c_zscore_sector_neutral


def test_winsorize_clips_within_sector_order_statistics() -> None:
    """The bottom/top decile of a sector is pulled to the order-statistic bound."""
    values = pd.Series([float(i) for i in range(10)])  # 0..9, one sector
    sector = pd.Series(["A"] * 10)
    w = c_winsorize(values, sector, lower_q=0.1, upper_q=0.9)
    # k = floor(0.1*10) = 1 -> lower bound is the 2nd smallest (1.0), upper the
    # 2nd largest (8.0); the extreme 0 and 9 are pulled in, the middle is untouched.
    assert w.iloc[0] == 1.0
    assert w.iloc[9] == 8.0
    assert list(w.iloc[1:9]) == [float(i) for i in range(1, 9)]


def test_winsorize_is_sector_local() -> None:
    """An outlier in sector A leaves sector B's values completely unchanged."""
    values = pd.Series([0.0, 1.0, 2.0, 3.0, 1_000.0, 10.0, 11.0, 12.0, 13.0, 14.0])
    sector = pd.Series(["A"] * 5 + ["B"] * 5)
    w = c_winsorize(values, sector, lower_q=0.1, upper_q=0.9)
    # Sector B has no outlier and floor(0.1*5)=0 -> no clipping at all.
    assert list(w.iloc[5:]) == [10.0, 11.0, 12.0, 13.0, 14.0]


def test_winsorize_preserves_nan() -> None:
    """A NaN input position is still NaN after winsorizing (no imputation)."""
    values = pd.Series([0.0, 1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    sector = pd.Series(["A"] * 10)
    w = c_winsorize(values, sector, lower_q=0.1, upper_q=0.9)
    assert np.isnan(w.iloc[2])


@settings(max_examples=75, deadline=None)
@given(
    raw=st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=10,
        max_size=200,
    )
)
def test_winsorize_is_idempotent(raw: list[float]) -> None:
    """Winsorizing an already-winsorized series changes nothing (idempotence)."""
    values = pd.Series(raw)
    sector = pd.Series(["A"] * len(raw))
    once = c_winsorize(values, sector, lower_q=0.1, upper_q=0.9)
    twice = c_winsorize(once, sector, lower_q=0.1, upper_q=0.9)
    pd.testing.assert_series_equal(once, twice, check_names=False)


def test_zscore_centres_sector_at_median() -> None:
    """The robust z is 0 at the sector median and antisymmetric around it."""
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    sector = pd.Series(["A"] * 5)
    z = c_zscore_sector_neutral(values, sector, lower_q=0.0, upper_q=1.0)
    assert np.isclose(z.iloc[2], 0.0)  # median value -> 0
    assert np.isclose(z.iloc[0], -z.iloc[4])  # symmetric tails
    assert np.isclose(z.iloc[1], -z.iloc[3])


def test_zscore_zero_dispersion_sector_is_zero() -> None:
    """A sector whose names are all equal carries no signal -> z = 0, never inf."""
    values = pd.Series([7.0, 7.0, 7.0, 7.0])
    sector = pd.Series(["A"] * 4)
    z = c_zscore_sector_neutral(values, sector)
    assert (z == 0.0).all()
    assert not np.isinf(z).any()


def test_zscore_single_name_sector_is_zero() -> None:
    """A lone name in its sector cannot be ranked against peers -> z = 0."""
    values = pd.Series([3.0, 1.0, 2.0])
    sector = pd.Series(["solo", "B", "B"])
    z = c_zscore_sector_neutral(values, sector)
    assert z.iloc[0] == 0.0


def test_zscore_is_computed_independently_per_sector() -> None:
    """Two identical sub-series in different sectors get identical z-scores."""
    values = pd.Series([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    sector = pd.Series(["A", "A", "A", "B", "B", "B"])
    z = c_zscore_sector_neutral(values, sector, lower_q=0.0, upper_q=1.0)
    assert np.isclose(z.iloc[0], z.iloc[3])
    assert np.isclose(z.iloc[1], z.iloc[4])
    assert np.isclose(z.iloc[2], z.iloc[5])


def test_zscore_preserves_nan() -> None:
    """A NaN input position stays NaN through the z-score."""
    values = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
    sector = pd.Series(["A"] * 5)
    z = c_zscore_sector_neutral(values, sector, lower_q=0.0, upper_q=1.0)
    assert np.isnan(z.iloc[2])
