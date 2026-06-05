"""Value signals — price and enterprise multiples, plus a point-in-time guard.

Value is the second pillar of the cross-sectional factor zoo: cheap stocks (low
price relative to fundamentals) earn a premium over expensive ones. This module
exposes the four multiples the research lab consumes as pure functions of price
and fundamental series, all registrable with the Data Curator by their ``c_``
prefix.

Two conventions are enforced throughout:

1. **Undefined, not infinite.** A multiple with a zero or negative denominator is
   economically meaningless (a firm with zero earnings has no P/E; negative book
   equity has no P/B). Such cases return ``NaN`` rather than ``+/-inf`` so they
   drop cleanly out of cross-sectional ranking instead of dominating it.
2. **Point-in-time.** Fundamentals are known only from their ``release_date``
   onward. When a ``release_date`` series is supplied, any observation whose
   fundamental had not yet been published as of the index date is masked to
   ``NaN`` — eliminating look-ahead bias at the signal level.

References
----------
.. [FF1992] Fama, E. F. and French, K. R. (1992). "The Cross-Section of Expected
   Stock Returns." Journal of Finance 47(2), 427-465. (Book-to-market / value.)
.. [FF1993] Fama, E. F. and French, K. R. (1993). "Common Risk Factors in the
   Returns on Stocks and Bonds." Journal of Financial Economics 33(1), 3-56.
   (The HML value factor.)
.. [Damodaran1994] Damodaran, A. (1994). Damodaran on Valuation. Wiley.
   (Enterprise-value multiples, incl. EV/EBITDA, and relative valuation.)

"""

from __future__ import annotations

import pandas as pd

from ._common import apply_pit, safe_divide


def c_pe_ratio(
    close: pd.Series[float],
    eps_ttm: pd.Series[float],
    release_date: pd.Series[pd.Timestamp] | None = None,
) -> pd.Series[float]:
    """Price-to-earnings ratio (trailing twelve months).

    Parameters
    ----------
    close:
        Closing prices indexed by trading day.
    eps_ttm:
        Trailing-twelve-month earnings per share, aligned to ``close``.
    release_date:
        Optional publication date of the EPS figure for the point-in-time guard.

    Returns
    -------
    pandas.Series
        ``close / eps_ttm``. NaN where EPS is zero or negative (the multiple is
        undefined), and where the figure is not yet public.

    Notes
    -----
    Low P/E is the classic value tilt. Negative or zero earnings make the ratio
    economically meaningless, so they resolve to NaN rather than a misleading
    large or negative number.

    References
    ----------
    Value premium from [FF1992]_; relative valuation from [Damodaran1994]_.

    """
    pe = safe_divide(close, eps_ttm, require_positive_denominator=True)
    return apply_pit(pe, release_date)


def c_pb_ratio(
    close: pd.Series[float],
    book_value_per_share: pd.Series[float],
    release_date: pd.Series[pd.Timestamp] | None = None,
) -> pd.Series[float]:
    """Price-to-book ratio.

    Parameters
    ----------
    close:
        Closing prices indexed by trading day.
    book_value_per_share:
        Book value of equity per share, aligned to ``close``.
    release_date:
        Optional publication date of the book value for the point-in-time guard.

    Returns
    -------
    pandas.Series
        ``close / book_value_per_share``. NaN where book value is zero or
        negative (negative equity has no meaningful multiple), and where the
        figure is not yet public.

    Notes
    -----
    Book-to-market (the reciprocal) is the canonical Fama-French value variable;
    we expose its more commonly quoted inverse, P/B.

    References
    ----------
    Book-to-market value factor from [FF1992]_ and [FF1993]_.

    """
    pb = safe_divide(close, book_value_per_share, require_positive_denominator=True)
    return apply_pit(pb, release_date)


def c_ev_ebitda(
    ev: pd.Series[float],
    ebitda_ttm: pd.Series[float],
    release_date: pd.Series[pd.Timestamp] | None = None,
) -> pd.Series[float]:
    """Enterprise-value-to-EBITDA multiple (trailing twelve months).

    Parameters
    ----------
    ev:
        Enterprise value (market cap + net debt), indexed by date.
    ebitda_ttm:
        Trailing-twelve-month EBITDA, aligned to ``ev``.
    release_date:
        Optional publication date of EBITDA for the point-in-time guard.

    Returns
    -------
    pandas.Series
        ``ev / ebitda_ttm``. NaN where EBITDA is zero or negative (undefined),
        and where the figure is not yet public.

    Notes
    -----
    EV/EBITDA is capital-structure neutral, making it the preferred cross-sector
    enterprise multiple in relative valuation.

    References
    ----------
    Enterprise multiples from [Damodaran1994]_.

    """
    ev_ebitda = safe_divide(ev, ebitda_ttm, require_positive_denominator=True)
    return apply_pit(ev_ebitda, release_date)


def c_fcf_yield(
    fcf_ttm: pd.Series[float],
    market_cap: pd.Series[float],
    release_date: pd.Series[pd.Timestamp] | None = None,
) -> pd.Series[float]:
    """Free-cash-flow yield (trailing twelve months).

    Parameters
    ----------
    fcf_ttm:
        Trailing-twelve-month free cash flow, indexed by date.
    market_cap:
        Equity market capitalisation, aligned to ``fcf_ttm``.
    release_date:
        Optional publication date of the FCF figure for the point-in-time guard.

    Returns
    -------
    pandas.Series
        ``fcf_ttm / market_cap``. The numerator's sign is preserved (negative FCF
        gives a negative yield), but a zero or negative market cap is undefined
        and resolves to NaN.

    Notes
    -----
    FCF yield is a cash-based value signal less sensitive to accrual accounting
    than earnings multiples; higher is cheaper.

    References
    ----------
    Cash-flow-based value from [Damodaran1994]_; value premium from [FF1992]_.

    """
    fcfy = safe_divide(fcf_ttm, market_cap, require_positive_denominator=True)
    return apply_pit(fcfy, release_date)
