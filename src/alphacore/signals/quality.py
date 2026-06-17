"""Quality signals — the orthogonal motor of the v3 strategy (§5.2).

Quality is the one axis the KN Factor Model does *not* span (its five styles are
beta, size, value, residual-volatility and momentum), so the residual alpha the
strategy targets has to be carried mostly by quality. This module implements the
quality engine as pure, point-in-time functions of fundamental series, each
oriented so that **higher is better** for cross-sectional ranking:

- **Gross profitability** (gross profit over assets) — the cleanest profitability
  signal, largely orthogonal to the price-based value multiples.
- **Accruals** — the cash backing of earnings; cash-backed earnings score higher.
- **Asset growth** — conservative balance-sheet expansion scores higher.
- **Net issuance** — net buybacks score higher than dilution.

These deliberately avoid ``ROE`` / margins / leverage (used in earlier drafts),
which share ``net_income`` / ``equity`` with the value multiples; the signals here
share much less DNA with value, keeping Quality the orthogonal axis.

All signals reuse the shared :func:`alphacore.signals._common.safe_divide` (NaN, never inf,
on a non-positive denominator) and :func:`alphacore.signals._common.apply_pit` (mask any
figure not yet public at the index date).

References
----------
.. [NovyMarx2013] Novy-Marx, R. (2013). "The Other Side of Value: The Gross
   Profitability Premium." Journal of Financial Economics 108(1), 1-28.
.. [Sloan1996] Sloan, R. G. (1996). "Do Stock Prices Fully Reflect Information in
   Accruals and Cash Flows about Future Earnings?" The Accounting Review 71(3),
   289-315.
.. [CGS2008] Cooper, M. J., Gulen, H. and Schill, M. J. (2008). "Asset Growth and
   the Cross-Section of Stock Returns." Journal of Finance 63(4), 1609-1651.
.. [PW2008] Pontiff, J. and Woodgate, A. (2008). "Share Issuance and Cross-
   Sectional Returns." Journal of Finance 63(2), 921-945.
.. [Piotroski2000] Piotroski, J. D. (2000). "Value Investing: The Use of
   Historical Financial Statement Information to Separate Winners from Losers."
   Journal of Accounting Research 38(Supplement), 1-41.

"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import apply_pit, safe_divide


def c_gross_profitability(
    gross_profit: pd.Series[float],
    total_assets: pd.Series[float],
    release_date: pd.Series[pd.Timestamp] | None = None,
) -> pd.Series[float]:
    """Gross profitability: gross profit scaled by total assets.

    Parameters
    ----------
    gross_profit:
        Gross profit (revenue minus cost of goods sold), indexed by date.
    total_assets:
        Total assets, aligned to ``gross_profit``.
    release_date:
        Optional publication date of the figures for the point-in-time guard.

    Returns
    -------
    pandas.Series
        ``gross_profit / total_assets``. NaN where the asset base is zero or
        negative (undefined), and where the figures are not yet public. Higher
        is better.

    Notes
    -----
    Novy-Marx shows gross profitability is "the other side of value": as
    powerful as book-to-market in the cross-section yet largely orthogonal to it,
    because the numerator sits at the top of the income statement (least polluted
    by accounting choices) rather than at the bottom like earnings.

    References
    ----------
    Gross profitability premium from [NovyMarx2013]_.

    """
    gp = safe_divide(gross_profit, total_assets, require_positive_denominator=True)
    return apply_pit(gp, release_date)


def c_accruals(
    net_income: pd.Series[float],
    cash_flow_operations: pd.Series[float],
    total_assets: pd.Series[float],
    release_date: pd.Series[pd.Timestamp] | None = None,
) -> pd.Series[float]:
    """Accruals score: the negative of total accruals scaled by assets.

    Parameters
    ----------
    net_income:
        Net income for the period, indexed by date.
    cash_flow_operations:
        Cash flow from operations, aligned to ``net_income``.
    total_assets:
        Total assets, aligned to ``net_income``.
    release_date:
        Optional publication date of the figures for the point-in-time guard.

    Returns
    -------
    pandas.Series
        ``-(net_income - cash_flow_operations) / total_assets``. NaN where the
        asset base is zero or negative, and where the figures are not yet public.
        Higher is better: earnings backed by cash (``cfo > net_income``) score
        positive, accrual-heavy earnings score negative.

    Notes
    -----
    Sloan documents that the accrual component of earnings is far less persistent
    than the cash component, yet investors anchor on headline earnings — so firms
    with high accruals subsequently underperform. Negating total accruals orients
    the signal so that cash-backed quality ranks high.

    References
    ----------
    Accruals anomaly from [Sloan1996]_.

    """
    total_accruals = net_income - cash_flow_operations
    score = safe_divide(-total_accruals, total_assets, require_positive_denominator=True)
    return apply_pit(score, release_date)


def c_asset_growth(
    total_assets: pd.Series[float],
    total_assets_prev: pd.Series[float],
    release_date: pd.Series[pd.Timestamp] | None = None,
) -> pd.Series[float]:
    """Asset-growth score: the negative of year-over-year asset growth.

    Parameters
    ----------
    total_assets:
        Current-period total assets, indexed by date.
    total_assets_prev:
        Prior-period (typically year-ago) total assets, aligned to
        ``total_assets``.
    release_date:
        Optional publication date of the current figure for the point-in-time
        guard.

    Returns
    -------
    pandas.Series
        ``-(total_assets - total_assets_prev) / total_assets_prev``. NaN where the
        prior asset base is zero or negative, and where the figure is not yet
        public. Higher is better: conservative (shrinking) balance sheets score
        positive, aggressive expansion scores negative.

    Notes
    -----
    Cooper, Gulen and Schill show that firms aggressively growing their asset base
    (often via empire-building investment or overpriced acquisitions) earn lower
    subsequent returns. Negating asset growth rewards capital discipline.

    References
    ----------
    Asset-growth anomaly from [CGS2008]_.

    """
    growth = safe_divide(
        total_assets - total_assets_prev,
        total_assets_prev,
        require_positive_denominator=True,
    )
    return apply_pit(-growth, release_date)


def c_net_issuance(
    shares_outstanding: pd.Series[float],
    shares_outstanding_prev: pd.Series[float],
    release_date: pd.Series[pd.Timestamp] | None = None,
) -> pd.Series[float]:
    """Net-issuance score: the negative of net share issuance.

    Parameters
    ----------
    shares_outstanding:
        Current-period shares outstanding, indexed by date.
    shares_outstanding_prev:
        Prior-period (typically year-ago) shares outstanding, aligned to
        ``shares_outstanding``.
    release_date:
        Optional publication date of the current figure for the point-in-time
        guard.

    Returns
    -------
    pandas.Series
        ``-(shares - shares_prev) / shares_prev``. NaN where the prior share count
        is zero or negative, and where the figure is not yet public. Higher is
        better: net buybacks (shares fall) score positive, dilution scores
        negative.

    Notes
    -----
    Pontiff and Woodgate find that net share issuance predicts the cross-section
    of returns beyond size, book-to-market and momentum: issuers underperform,
    repurchasers outperform. Negating net issuance rewards shareholder-friendly
    capital return.

    References
    ----------
    Share-issuance anomaly from [PW2008]_.

    """
    issuance = safe_divide(
        shares_outstanding - shares_outstanding_prev,
        shares_outstanding_prev,
        require_positive_denominator=True,
    )
    return apply_pit(-issuance, release_date)


def c_piotroski_f(
    current: pd.DataFrame,
    prior: pd.DataFrame,
    release_date: pd.Series[pd.Timestamp] | None = None,
) -> pd.Series[float]:
    """Piotroski F-Score: a 0-9 tally of fundamental-health criteria.

    Parameters
    ----------
    current:
        Current-period fundamentals, one row per index date, with columns
        ``net_income``, ``cash_flow_operations``, ``total_assets``,
        ``long_term_debt``, ``current_assets``, ``current_liabilities``,
        ``shares_outstanding``, ``gross_profit`` and ``revenue``.
    prior:
        Prior-period (year-ago) fundamentals with the same columns and index,
        used for the six year-over-year criteria. ``cash_flow_operations`` is
        unused in the prior frame.
    release_date:
        Optional publication date of the current filing for the point-in-time
        guard.

    Returns
    -------
    pandas.Series
        Integer F-Score in ``[0, 9]`` per date (float once the PIT guard masks a
        not-yet-public filing to NaN). Higher is better. Each criterion whose
        inputs are missing or undefined contributes 0 — a firm that cannot
        demonstrate a positive cannot earn its point.

    Notes
    -----
    The nine binary criteria (one point each):

    Profitability — (1) ``ROA > 0``; (2) ``CFO > 0``; (3) ``ΔROA > 0``;
    (4) accrual quality ``CFO/assets > ROA``.

    Leverage, liquidity and funding — (5) ``Δleverage < 0`` (long-term-debt
    ratio falls); (6) ``Δcurrent-ratio > 0``; (7) no net share issuance.

    Operating efficiency — (8) ``Δgross-margin > 0``; (9) ``Δasset-turnover > 0``.

    All ratios go through :func:`safe_divide`, so an undefined denominator yields
    NaN, and ``NaN`` comparisons evaluate False — the criterion is simply not met.

    References
    ----------
    The nine-signal fundamental score from [Piotroski2000]_.

    """
    pos = {"require_positive_denominator": True}

    roa = safe_divide(current["net_income"], current["total_assets"], **pos)
    roa_prev = safe_divide(prior["net_income"], prior["total_assets"], **pos)
    cfo_to_assets = safe_divide(current["cash_flow_operations"], current["total_assets"], **pos)
    leverage = safe_divide(current["long_term_debt"], current["total_assets"], **pos)
    leverage_prev = safe_divide(prior["long_term_debt"], prior["total_assets"], **pos)
    current_ratio = safe_divide(current["current_assets"], current["current_liabilities"], **pos)
    current_ratio_prev = safe_divide(prior["current_assets"], prior["current_liabilities"], **pos)
    gross_margin = safe_divide(current["gross_profit"], current["revenue"], **pos)
    gross_margin_prev = safe_divide(prior["gross_profit"], prior["revenue"], **pos)
    asset_turnover = safe_divide(current["revenue"], current["total_assets"], **pos)
    asset_turnover_prev = safe_divide(prior["revenue"], prior["total_assets"], **pos)

    criteria = [
        roa > 0,  # 1. positive return on assets
        current["cash_flow_operations"] > 0,  # 2. positive operating cash flow
        roa > roa_prev,  # 3. rising profitability
        cfo_to_assets > roa,  # 4. earnings backed by cash (low accruals)
        leverage < leverage_prev,  # 5. falling leverage
        current_ratio > current_ratio_prev,  # 6. improving liquidity
        current["shares_outstanding"] <= prior["shares_outstanding"],  # 7. no dilution
        gross_margin > gross_margin_prev,  # 8. rising gross margin
        asset_turnover > asset_turnover_prev,  # 9. rising asset turnover
    ]

    # Each criterion is a boolean Series (NaN comparisons evaluate False, so a
    # missing input simply fails its criterion). Sum them as integers, then carry
    # the 0-9 tally as float so the PIT guard can mask not-yet-public rows to NaN.
    tally = np.zeros(len(current), dtype=np.int64)
    for criterion in criteria:
        tally += criterion.to_numpy(dtype=bool).astype(np.int64)
    score: pd.Series[float] = pd.Series(tally.astype(np.float64), index=current.index)
    return apply_pit(score, release_date)
