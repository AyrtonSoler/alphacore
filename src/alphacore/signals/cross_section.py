"""Cross-sectional signal transforms — robust sector-neutral z-score (v3 §5.3).

Stage 3 of the v3 pipeline. Where :mod:`signals.value`, :mod:`signals.quality`
and :mod:`signals.momentum` produce *per-name time series*, these functions act
**across names on a single rebalancing cross-section**, grouped by GICS sector.
They turn each raw signal into a robust, sector-neutral z-score so that:

- a name is ranked against its sector peers, not the whole market — restoring
  meaning to leverage/margin ratios inside financials and REITs;
- the accidental sector bet of a flat cross-sectional rank (which overweights
  structurally cheap sectors) is removed at the signal level; and
- outliers move the ranking through the median/MAD, not the mean/standard
  deviation, so a single broken multiple cannot dominate.

The caller applies these per rebalancing date (e.g. ``panel.groupby(date)``); each
function here handles only the within-date, within-sector grouping. ``NaN`` inputs
pass through untouched — a missing fundamental stays missing, never imputed.

Notes
-----
Winsorization clips to within-sector **order statistics** (empirical quantiles
that land on actual data points) rather than interpolated quantiles. This makes
the operation **idempotent**: after clipping, the tail values sit exactly on the
boundary order statistic, so recomputing the bound returns the same value and a
second pass is a no-op — the invariant CLAUDE.md requires of winsorize.

References
----------
.. [Tukey1962] Tukey, J. W. (1962). "The Future of Data Analysis." Annals of
   Mathematical Statistics 33(1), 1-67. (Winsorization / robust summaries.)
.. [RC1993] Rousseeuw, P. J. and Croux, C. (1993). "Alternatives to the Median
   Absolute Deviation." Journal of the American Statistical Association 88(424),
   1273-1283. (The 1.4826 MAD-to-sigma consistency constant.)

"""

from __future__ import annotations

import numpy as np
import pandas as pd

# MAD-to-standard-deviation consistency constant under normality (1 / Phi^-1(0.75)).
_MAD_TO_SIGMA = 1.4826


def _winsorize_group(group: pd.Series[float], lower_q: float, upper_q: float) -> pd.Series[float]:
    """Clip one sector's values to its order-statistic bounds (idempotent)."""
    present = group.dropna()
    n = len(present)
    if n == 0:
        return group
    ordered = np.sort(present.to_numpy())
    # Number of points to pull in from each tail; floor keeps the bounds on real
    # data points, which is what makes a second pass a no-op. The 1e-9 nudge
    # absorbs float-representation dust (e.g. (1-0.9)*10 == 0.9999...8) so a tail
    # fraction that should be a whole count does not silently drop one.
    k_low = int(np.floor(lower_q * n + 1e-9))
    k_high = int(np.floor((1.0 - upper_q) * n + 1e-9))
    lower = ordered[min(k_low, n - 1)]
    upper = ordered[max(0, n - 1 - k_high)]
    return group.clip(lower=lower, upper=upper)


def _zscore_group(group: pd.Series[float]) -> pd.Series[float]:
    """Robust z-score of one sector via median and MAD; zero-dispersion -> 0."""
    present = group.dropna()
    if len(present) == 0:
        return group
    median = float(np.median(present.to_numpy()))
    mad = float(np.median(np.abs(present.to_numpy() - median)))
    if mad == 0.0:
        # No dispersion (all equal, or a lone name): no relative signal -> neutral 0,
        # while genuine NaN inputs stay NaN.
        return pd.Series(np.where(group.isna(), np.nan, 0.0), index=group.index)
    return (group - median) / (_MAD_TO_SIGMA * mad)


def c_winsorize(
    values: pd.Series[float],
    sector: pd.Series[str],
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> pd.Series[float]:
    """Winsorize a cross-section to within-sector order-statistic bounds.

    Parameters
    ----------
    values:
        The raw signal for one rebalancing cross-section, indexed by name.
    sector:
        GICS sector label per name, aligned to ``values``.
    lower_q, upper_q:
        Lower/upper quantiles defining the clip. Defaults ``0.01``/``0.99`` clip
        the most extreme 1% of each tail within each sector.

    Returns
    -------
    pandas.Series
        ``values`` with each sector's tail values pulled to that sector's lower /
        upper order statistic. ``NaN`` positions are preserved. Idempotent:
        ``c_winsorize(c_winsorize(x)) == c_winsorize(x)``.

    Notes
    -----
    The bounds are real data points (order statistics), so after one pass the
    clipped tails sit exactly on the boundary; recomputing the bound returns the
    same value and a second pass changes nothing.

    References
    ----------
    Winsorization from [Tukey1962]_.

    """
    return values.groupby(sector, sort=False).transform(
        _winsorize_group, lower_q=lower_q, upper_q=upper_q
    )


def c_zscore_sector_neutral(
    values: pd.Series[float],
    sector: pd.Series[str],
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> pd.Series[float]:
    """Robust, sector-neutral z-score of a cross-section (winsorize then z).

    Parameters
    ----------
    values:
        The raw signal for one rebalancing cross-section, indexed by name.
    sector:
        GICS sector label per name, aligned to ``values``.
    lower_q, upper_q:
        Winsorization quantiles applied within each sector before standardising.

    Returns
    -------
    pandas.Series
        Per name, ``(winsorized - sector_median) / (1.4826 * sector_MAD)``. Each
        sector is centred at its median (median -> 0) and scaled by its MAD, so
        the score is comparable across sectors. A sector with no dispersion (all
        equal, or a single name) yields 0 rather than inf/NaN. ``NaN`` inputs are
        preserved.

    Notes
    -----
    Median and MAD make the standardisation robust: a single broken multiple
    shifts neither the centre nor the scale the way a mean/standard-deviation
    z-score would. The 1.4826 factor rescales the MAD to a standard-deviation
    equivalent under normality so the scores read like ordinary z-scores.

    References
    ----------
    Robust scale via MAD and the 1.4826 constant from [RC1993]_.

    """
    winsorized = c_winsorize(values, sector, lower_q=lower_q, upper_q=upper_q)
    return winsorized.groupby(sector, sort=False).transform(_zscore_group)
