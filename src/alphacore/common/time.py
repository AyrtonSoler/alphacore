"""NYSE trading-day calendar — a thin, typed wrapper over QuantLib.

The research lab indexes time by *trading days*, not calendar days. This module
exposes the New York Stock Exchange session calendar (weekends + exchange
holidays removed) so signal, portfolio, and backtest code share one definition of
"a trading day" and never accidentally diverge.

Notes
-----
QuantLib's ``UnitedStates(NYSE)`` calendar encodes the official NYSE holiday
schedule (New Year's, MLK Day, Washington's Birthday, Good Friday, Memorial Day,
Juneteenth, Independence Day, Labor Day, Thanksgiving, Christmas).

"""

from __future__ import annotations

from datetime import date

import QuantLib as ql  # noqa: N813  (`ql` is the universal QuantLib alias)

_NYSE = ql.UnitedStates(ql.UnitedStates.NYSE)


def _to_ql(day: date) -> ql.Date:
    """Convert a :class:`datetime.date` to a :class:`QuantLib.Date`."""
    return ql.Date(day.day, day.month, day.year)


def _to_py(day: ql.Date) -> date:
    """Convert a :class:`QuantLib.Date` to a :class:`datetime.date`."""
    return date(day.year(), day.month(), day.dayOfMonth())


def trading_days(start: date, end: date) -> list[date]:
    """List NYSE trading days in ``[start, end]`` (both endpoints inclusive).

    Parameters
    ----------
    start:
        First calendar date to consider (included if it is a trading day).
    end:
        Last calendar date to consider (included if it is a trading day).

    Returns
    -------
    list[datetime.date]
        Trading days in ascending order. Weekends and NYSE holidays are removed.

    Notes
    -----
    There are ~252 NYSE sessions per calendar year. 2023 had 250 sessions.

    """
    days: list[date] = []
    current = _to_ql(start)
    last = _to_ql(end)
    while current <= last:
        if _NYSE.isBusinessDay(current):
            days.append(_to_py(current))
        current = current + 1
    return days
