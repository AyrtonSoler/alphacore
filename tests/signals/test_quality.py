"""RED tests for signals.quality — the orthogonal Quality motor (v3 §5.2).

These signals are the engine of the v3 strategy precisely because they share
little DNA with the five KN style factors or with the value multiples: gross
profitability, accruals, asset growth and net issuance are fundamental-structure
signals, not price multiples.

Acceptance
----------
- Gross profitability = gross_profit / total_assets; higher is better; a
  non-positive asset base is NaN (never inf).
- Accruals score = -(net_income - cfo) / total_assets, so a firm whose earnings
  are *backed by cash* (cfo > net_income) scores positive (better), and a firm
  accruing earnings without cash scores negative (worse).
- Asset-growth score = -(assets - assets_prev) / assets_prev, so conservative
  (shrinking) asset bases score positive and aggressive growth scores negative.
- Net-issuance score = -(shares - shares_prev) / shares_prev, so net buybacks
  score positive and dilution scores negative.
- Every signal honours point-in-time: a value at t is NaN when its fundamental's
  release_date is after t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from alphacore.signals.quality import (
    c_accruals,
    c_asset_growth,
    c_gross_profitability,
    c_net_issuance,
    c_piotroski_f,
)

# Field order shared by the current/prior frames the Piotroski score consumes.
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


def _firm_frame(values: dict[str, float], index: pd.Index) -> pd.DataFrame:
    """Broadcast a single firm-year's fields across ``index`` into a frame."""
    return pd.DataFrame(
        {field: [values[field]] * len(index) for field in _PIOTROSKI_FIELDS},
        index=index,
    )


def test_gross_profitability_is_gross_profit_over_assets() -> None:
    """GP/A reproduces the textbook ratio; higher is better (Novy-Marx 2013)."""
    gross_profit = pd.Series([300.0, 600.0])
    total_assets = pd.Series([1000.0, 1000.0])
    gp = c_gross_profitability(gross_profit, total_assets)
    assert gp.iloc[0] == 0.30
    assert gp.iloc[1] == 0.60  # more gross profit on the same assets ranks higher


def test_gross_profitability_non_positive_assets_is_nan_not_inf() -> None:
    """A zero or negative asset base is undefined -> NaN, never inf."""
    gross_profit = pd.Series([300.0, 300.0])
    total_assets = pd.Series([0.0, -500.0])
    gp = c_gross_profitability(gross_profit, total_assets)
    assert gp.isna().all()
    assert not np.isinf(gp).any()


def test_accruals_rewards_cash_backed_earnings() -> None:
    """cfo > net_income (cash-backed) scores positive; the reverse scores negative."""
    net_income = pd.Series([100.0, 200.0])
    cfo = pd.Series([180.0, 150.0])
    total_assets = pd.Series([1000.0, 1000.0])
    accr = c_accruals(net_income, cfo, total_assets)
    assert accr.iloc[0] == 0.08  # -(100 - 180)/1000 = +0.08 -> better
    assert accr.iloc[1] == -0.05  # -(200 - 150)/1000 = -0.05 -> worse


def test_accruals_non_positive_assets_is_nan() -> None:
    """Non-positive asset base is undefined -> NaN, never inf."""
    accr = c_accruals(pd.Series([100.0]), pd.Series([50.0]), pd.Series([0.0]))
    assert accr.isna().all()
    assert not np.isinf(accr).any()


def test_asset_growth_rewards_conservative_balance_sheet() -> None:
    """Shrinking assets score positive; aggressive expansion scores negative."""
    total_assets = pd.Series([950.0, 1100.0])
    total_assets_prev = pd.Series([1000.0, 1000.0])
    ag = c_asset_growth(total_assets, total_assets_prev)
    assert ag.iloc[0] == 0.05  # -(950 - 1000)/1000 = +0.05 -> conservative, better
    assert np.isclose(ag.iloc[1], -0.10)  # -(1100 - 1000)/1000 = -0.10 -> worse


def test_net_issuance_rewards_buybacks_penalises_dilution() -> None:
    """Net buyback (shares fall) scores positive; dilution scores negative."""
    shares = pd.Series([90.0, 110.0])
    shares_prev = pd.Series([100.0, 100.0])
    ni = c_net_issuance(shares, shares_prev)
    assert ni.iloc[0] == 0.10  # -(90 - 100)/100 = +0.10 -> buyback, better
    assert np.isclose(ni.iloc[1], -0.10)  # -(110 - 100)/100 = -0.10 -> dilution


def test_quality_signals_propagate_pit() -> None:
    """Each signal is NaN when its fundamental is not yet public at the index date."""
    idx = pd.to_datetime(["2026-01-31", "2026-02-28", "2026-03-31"])
    release_date = pd.Series(
        pd.to_datetime(["2026-01-15", "2026-03-10", "2026-03-20"]),
        index=idx,
    )
    gp = c_gross_profitability(
        pd.Series([300.0, 300.0, 300.0], index=idx),
        pd.Series([1000.0, 1000.0, 1000.0], index=idx),
        release_date=release_date,
    )
    assert gp.iloc[0] == 0.30  # released 2026-01-15 <= 2026-01-31
    assert np.isnan(gp.iloc[1])  # released 2026-03-10 > 2026-02-28 -> masked
    assert gp.iloc[2] == 0.30  # released 2026-03-20 <= 2026-03-31


@settings(max_examples=50, deadline=None)
@given(
    gross_profit=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
    total_assets=st.floats(min_value=1.0, max_value=1e9, allow_nan=False),
)
def test_gross_profitability_never_infinite(gross_profit: float, total_assets: float) -> None:
    """For any positive asset base the ratio is finite (no inf can leak through)."""
    gp = c_gross_profitability(pd.Series([gross_profit]), pd.Series([total_assets]))
    assert np.isfinite(gp.iloc[0])


# A firm that satisfies all nine Piotroski criteria simultaneously.
_PERFECT_CURRENT = {
    "net_income": 100.0,  # ROA > 0
    "cash_flow_operations": 150.0,  # CFO > 0 and CFO/TA (0.15) > ROA (0.10)
    "total_assets": 1000.0,
    "long_term_debt": 100.0,  # leverage 0.10 < prior 0.18
    "current_assets": 500.0,  # current ratio 2.0 > prior 1.33
    "current_liabilities": 250.0,
    "shares_outstanding": 90.0,  # no issuance (90 <= 100)
    "gross_profit": 300.0,  # gross margin 0.30 > prior 0.222
    "revenue": 1000.0,  # asset turnover 1.0 > prior 0.818
}
_PERFECT_PRIOR = {
    "net_income": 50.0,  # ROA prior 0.0455 < current 0.10
    "cash_flow_operations": 0.0,  # unused in prior
    "total_assets": 1100.0,
    "long_term_debt": 200.0,
    "current_assets": 400.0,
    "current_liabilities": 300.0,
    "shares_outstanding": 100.0,
    "gross_profit": 200.0,
    "revenue": 900.0,
}

# A firm that fails every criterion.
_WORST_CURRENT = {
    "net_income": -100.0,
    "cash_flow_operations": -200.0,
    "total_assets": 1000.0,
    "long_term_debt": 300.0,
    "current_assets": 200.0,
    "current_liabilities": 400.0,
    "shares_outstanding": 110.0,
    "gross_profit": 100.0,
    "revenue": 1000.0,
}
_WORST_PRIOR = {
    "net_income": 100.0,
    "cash_flow_operations": 0.0,
    "total_assets": 1000.0,
    "long_term_debt": 100.0,
    "current_assets": 400.0,
    "current_liabilities": 200.0,
    "shares_outstanding": 100.0,
    "gross_profit": 400.0,
    "revenue": 1200.0,
}


def test_piotroski_perfect_firm_scores_nine() -> None:
    """A firm meeting all nine fundamental-health criteria scores 9."""
    idx = pd.RangeIndex(2)
    score = c_piotroski_f(_firm_frame(_PERFECT_CURRENT, idx), _firm_frame(_PERFECT_PRIOR, idx))
    assert (score == 9).all()


def test_piotroski_worst_firm_scores_zero() -> None:
    """A firm failing every criterion scores 0."""
    idx = pd.RangeIndex(2)
    score = c_piotroski_f(_firm_frame(_WORST_CURRENT, idx), _firm_frame(_WORST_PRIOR, idx))
    assert (score == 0).all()


def test_piotroski_score_is_bounded_zero_to_nine() -> None:
    """Every score lies in the integer range [0, 9] by construction."""
    idx = pd.RangeIndex(2)
    for current, prior in (
        (_PERFECT_CURRENT, _PERFECT_PRIOR),
        (_WORST_CURRENT, _WORST_PRIOR),
    ):
        score = c_piotroski_f(_firm_frame(current, idx), _firm_frame(prior, idx))
        assert ((score >= 0) & (score <= 9)).all()


def test_piotroski_propagates_pit() -> None:
    """The score is NaN where the current filing is not yet public at the date."""
    idx = pd.to_datetime(["2026-01-31", "2026-02-28"])
    release_date = pd.Series(pd.to_datetime(["2026-01-15", "2026-03-10"]), index=idx)
    score = c_piotroski_f(
        _firm_frame(_PERFECT_CURRENT, idx),
        _firm_frame(_PERFECT_PRIOR, idx),
        release_date=release_date,
    )
    assert score.iloc[0] == 9  # public on 2026-01-15 <= 2026-01-31
    assert np.isnan(score.iloc[1])  # public 2026-03-10 > 2026-02-28 -> masked
