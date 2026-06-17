"""Shared signal primitives — safe division and the point-in-time guard.

Every fundamental-driven signal in :mod:`alphacore.signals` obeys the same two conventions,
so they live here once and are imported by the individual signal modules rather
than re-defined per file:

1. **Undefined, not infinite.** A ratio with a zero (or non-positive) denominator
   is economically meaningless and must drop out of cross-sectional ranking
   instead of dominating it, so :func:`safe_divide` returns ``NaN`` — never
   ``+/-inf`` — for such positions and raises no divide warning.
2. **Point-in-time.** A fundamental is knowable only from its ``release_date``
   onward; :func:`apply_pit` masks any observation whose underlying figure had not
   yet been published as of the index date, eliminating look-ahead at the signal
   level.

These functions are pure and operate element-wise on aligned ``pandas`` series.

"""

from __future__ import annotations

import numpy as np
import pandas as pd


def safe_divide(
    numerator: pd.Series[float],
    denominator: pd.Series[float],
    *,
    require_positive_denominator: bool,
) -> pd.Series[float]:
    """Divide two series, returning NaN where the result is undefined.

    Parameters
    ----------
    numerator, denominator:
        Aligned float series.
    require_positive_denominator:
        When True, a non-positive denominator (``<= 0``) is treated as undefined.
        When False, only an exactly-zero denominator is undefined (the sign of
        the ratio is allowed to carry economic meaning).

    Returns
    -------
    pandas.Series
        ``numerator / denominator`` with undefined positions set to NaN. No
        ``inf`` is ever produced, so the operation raises no divide warnings.

    """
    invalid = denominator <= 0 if require_positive_denominator else denominator == 0
    safe_denominator = denominator.where(~invalid, other=np.nan)
    return numerator / safe_denominator


def apply_pit(
    values: pd.Series[float],
    release_date: pd.Series[pd.Timestamp] | None,
) -> pd.Series[float]:
    """Mask values not yet public as of their index date (point-in-time guard).

    Parameters
    ----------
    values:
        The computed signal, indexed by observation date.
    release_date:
        Per-observation publication date of the underlying fundamental, aligned
        to ``values.index``. ``None`` disables the guard.

    Returns
    -------
    pandas.Series
        ``values`` with every position whose ``release_date`` is strictly after
        its index date set to NaN.

    """
    if release_date is None:
        return values
    index_dates = pd.Series(pd.to_datetime(values.index), index=values.index)
    not_yet_public = pd.to_datetime(release_date) > index_dates
    return values.where(~not_yet_public, other=np.nan)
