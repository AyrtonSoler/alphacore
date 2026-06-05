"""RED tests for assemble_cross_section — the residual-alpha panel (§5.3+§5.4).

The keystone glue of the data wiring: take every ticker's per-ticker signal frame,
extract the point-in-time cross-section on a rebalancing date (as-of the last public
row), turn each raw signal into a sector-neutral robust z-score (§5.3), average the
z-scores into sleeve composites, and residualize each composite against the factor
design (§5.4). The output is the per-name residual-alpha score the optimizer eats.

The headline guarantee: the score is orthogonal to every factor *by construction* —
the entire v3 thesis, demonstrated end-to-end.

Acceptance
----------
- The score and each sleeve residual are orthogonal to every factor column.
- Only names common to the signals, sectors and factor design appear.
- A name whose data is not yet public on the date is excluded (PIT, as-of).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from signals.pipeline import DEFAULT_SLEEVES, assemble_cross_section

_QUALITY = ["quality_gross_profitability", "quality_accruals", "quality_piotroski"]
_MOMENTUM = ["momentum_roc12", "momentum_roc6"]
_SLEEVES = {"quality": _QUALITY, "momentum": _MOMENTUM}
# Every signal across the canonical sleeves, so the default-sleeves path is covered.
_ALL_SIGNALS = [col for cols in DEFAULT_SLEEVES.values() for col in cols]
_DATES = pd.to_datetime(["2020-12-31", "2021-01-29"])


def _frames(values: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """One constant-over-time per-ticker frame per row of ``values``."""
    return {
        ticker: pd.DataFrame(
            {col: [values.loc[ticker, col]] * len(_DATES) for col in values.columns},
            index=_DATES,
        )
        for ticker in values.index
    }


def _setup(n: int = 24, seed: int = 0):
    rng = np.random.default_rng(seed)
    tickers = [f"T{i:02d}" for i in range(n)]
    values = pd.DataFrame(
        rng.normal(size=(n, len(_ALL_SIGNALS))), index=tickers, columns=_ALL_SIGNALS
    )
    sectors = pd.Series(["A", "B", "C"] * (n // 3), index=tickers)
    factor_design = pd.DataFrame(
        rng.normal(size=(n, 3)), index=tickers, columns=["beta", "size", "value"]
    )
    return _frames(values), sectors, factor_design, tickers


def _max_abs_factor_corr(score: pd.Series, factors: pd.DataFrame) -> float:
    s = score.to_numpy()
    return max(
        abs(np.corrcoef(s, factors.loc[score.index, col].to_numpy())[0, 1])
        for col in factors.columns
    )


def test_score_is_orthogonal_to_every_factor() -> None:
    """The residual-alpha score carries ~zero exposure to each factor."""
    frames, sectors, design, _ = _setup()
    panel = assemble_cross_section(frames, _DATES[-1], sectors, design, sleeves=_SLEEVES)
    assert _max_abs_factor_corr(panel["score"], design) < 1e-8


def test_each_sleeve_residual_is_orthogonal_to_factors() -> None:
    """Every sleeve residual is factor-neutral, not just the combined score."""
    frames, sectors, design, _ = _setup(seed=1)
    panel = assemble_cross_section(frames, _DATES[-1], sectors, design, sleeves=_SLEEVES)
    for sleeve in _SLEEVES:
        assert _max_abs_factor_corr(panel[f"{sleeve}_residual"], design) < 1e-8


def test_only_common_names_survive() -> None:
    """A name missing from the factor design or sector map is dropped."""
    frames, sectors, design, tickers = _setup(seed=2)
    design = design.drop(index=tickers[0])  # drop one name's factor exposures
    sectors = sectors.drop(index=tickers[1])  # and another's sector
    panel = assemble_cross_section(frames, _DATES[-1], sectors, design, sleeves=_SLEEVES)
    assert tickers[0] not in panel.index
    assert tickers[1] not in panel.index
    assert tickers[2] in panel.index


def test_defaults_to_canonical_sleeves() -> None:
    """Without an explicit sleeves arg, the canonical quality/value/momentum run."""
    frames, sectors, design, _ = _setup(seed=4)
    panel = assemble_cross_section(frames, _DATES[-1], sectors, design)
    for sleeve in DEFAULT_SLEEVES:
        assert f"{sleeve}_residual" in panel.columns
    assert "score" in panel.columns


def test_not_yet_public_names_are_excluded() -> None:
    """A name whose frame starts after the date contributes no row (as-of/PIT)."""
    frames, sectors, design, tickers = _setup(seed=3)
    # Shift one ticker's frame to start after the rebalancing date.
    late = tickers[0]
    frames[late] = frames[late].set_axis(pd.to_datetime(["2099-01-31", "2099-02-28"]), axis=0)
    panel = assemble_cross_section(frames, _DATES[-1], sectors, design, sleeves=_SLEEVES)
    assert late not in panel.index
