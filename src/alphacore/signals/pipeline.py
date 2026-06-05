"""Per-ticker signal computation — wiring loaded data to the signal functions.

The bridge from one loaded curator frame (:func:`common.panel.load_ticker_frame`)
to a tidy per-ticker frame of named signals, ready for the cross-sectional stages
(§5.3 z-score, §5.4 neutralization). It:

- builds the year-over-year "previous" fundamentals a few signals need. Annual
  fundamentals are carried daily in the curator output, so a one-year session shift
  recovers the prior filing's value (an approximation appropriate to the annual,
  slow-moving Quality engine, §3);
- threads ``release_date`` through every fundamental signal so the point-in-time
  guard masks not-yet-public figures; and
- emits family-prefixed columns (``quality_*``, ``momentum_*``, ``risk_*``).

Risk signals are computed here but, per §5.2, never enter the alpha — they feed the
regime overlay and the optimizer's risk model only.
"""

from __future__ import annotations

import pandas as pd

from .momentum import c_roc_6m, c_roc_12m, c_sma_50_200_signal
from .quality import (
    c_accruals,
    c_asset_growth,
    c_gross_profitability,
    c_net_issuance,
    c_piotroski_f,
)
from .risk import c_drawdown, c_idiosyncratic_vol, c_realized_vol

_SESSIONS_PER_YEAR = 252

# Fields the Piotroski F-Score consumes, in the order its current/prior frames expect.
_PIOTROSKI_FIELDS = [
    "net_income",
    "cash_flow_operations",
    "total_assets",
    "long_term_debt",
    "current_assets",
    "current_liabilities",
    "shares_outstanding",
    "gross_profit",
    "revenue",
]


def compute_ticker_signals(
    frame: pd.DataFrame,
    market_returns: pd.Series[float] | None = None,
) -> pd.DataFrame:
    """Compute the per-ticker quality, momentum and risk signals.

    Parameters
    ----------
    frame:
        A loaded curator frame from :func:`common.panel.load_ticker_frame`, indexed
        by date with the semantic field columns and ``release_date``.
    market_returns:
        Optional market return series aligned to ``frame.index``. When given, the
        CAPM idiosyncratic volatility (``risk_idiosyncratic_vol``) is added.

    Returns
    -------
    pandas.DataFrame
        Indexed like ``frame``, with ``quality_*``, ``momentum_*`` and ``risk_*``
        signal columns. Fundamental signals are NaN before their ``release_date``;
        windowed signals are NaN during warm-up.

    Notes
    -----
    The "previous" fundamentals use a one-year session shift on the daily-carried
    series, so at filing boundaries they lag by at most a few sessions — immaterial
    for the slow, annual Quality signals and for monthly rebalancing.

    """
    release_date = frame["release_date"] if "release_date" in frame.columns else None
    close = frame["close"]
    total_assets_prev = frame["total_assets"].shift(_SESSIONS_PER_YEAR)
    shares_prev = frame["shares_outstanding"].shift(_SESSIONS_PER_YEAR)
    current_fundamentals = frame[_PIOTROSKI_FIELDS]
    prior_fundamentals = current_fundamentals.shift(_SESSIONS_PER_YEAR)

    signals = pd.DataFrame(index=frame.index)
    signals["quality_gross_profitability"] = c_gross_profitability(
        frame["gross_profit"], frame["total_assets"], release_date
    )
    signals["quality_accruals"] = c_accruals(
        frame["net_income"], frame["cash_flow_operations"], frame["total_assets"], release_date
    )
    signals["quality_asset_growth"] = c_asset_growth(
        frame["total_assets"], total_assets_prev, release_date
    )
    signals["quality_net_issuance"] = c_net_issuance(
        frame["shares_outstanding"], shares_prev, release_date
    )
    signals["quality_piotroski"] = c_piotroski_f(
        current_fundamentals, prior_fundamentals, release_date
    )

    signals["momentum_roc12"] = c_roc_12m(close)
    signals["momentum_roc6"] = c_roc_6m(close)
    signals["momentum_trend"] = c_sma_50_200_signal(close)

    returns = close.pct_change(fill_method=None)
    signals["risk_realized_vol"] = c_realized_vol(returns)
    signals["risk_drawdown"] = c_drawdown(close)
    if market_returns is not None:
        signals["risk_idiosyncratic_vol"] = c_idiosyncratic_vol(returns, market_returns)

    return signals
