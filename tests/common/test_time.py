"""RED tests for common.time — NYSE trading calendar wrapping QuantLib.

Acceptance: US trading calendar has 252 ± 2 days in 2023 and excludes 2023-07-04.
"""

from __future__ import annotations

from datetime import date

from common.time import trading_days


def test_2023_has_about_252_trading_days() -> None:
    """The NYSE traded ~252 sessions in 2023 (250-254 tolerance band)."""
    days = trading_days(date(2023, 1, 1), date(2023, 12, 31))
    assert 250 <= len(days) <= 254


def test_excludes_independence_day_2023() -> None:
    """2023-07-04 (Independence Day) is a market holiday — never a trading day."""
    days = trading_days(date(2023, 1, 1), date(2023, 12, 31))
    assert date(2023, 7, 4) not in days


def test_includes_a_regular_weekday() -> None:
    """An ordinary mid-week, non-holiday session is present and sorted ascending."""
    days = trading_days(date(2023, 1, 1), date(2023, 12, 31))
    assert date(2023, 6, 1) in days  # Thursday, no holiday
    assert days == sorted(days)


def test_range_is_inclusive_of_endpoints() -> None:
    """A start/end that are themselves trading days are both included."""
    days = trading_days(date(2023, 6, 1), date(2023, 6, 30))
    assert days[0] == date(2023, 6, 1)
    assert days[-1] == date(2023, 6, 30)
