"""RED tests for ml.trust_governor — the ML distrust, made mechanism (v3 §5.6).

The ML never drives blindly. The governor encodes four locks; this module covers
the two that are pure score logic:

- **Earned weight (lock 2):** the ML score gets weight ``1 - lambda`` only if it beat
  the linear baseline out-of-sample; otherwise ``lambda = 1`` and the strategy runs
  100% on rules, degrading gracefully.
- **Bounded influence (lock 3):** the ML can move a name's rank by at most ``k_max``
  percentiles, so there are no extreme, inexplicable bets.

(Monotonicity — lock 1 — is enforced at EBM training; the optimizer — lock 5 —
sets the weights. Those live elsewhere.)

Acceptance
----------
- If the ML does not beat the baseline OOS, lambda = 1 (pure rules).
- If it wins, lambda falls toward lambda_min as the edge grows, staying in
  [lambda_min, 1].
- Blending with lambda=1 returns the rules score unchanged.
- Bounding influence keeps every name's governed rank within k_max percentiles of
  its rules rank.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphacore.ml.trust_governor import blend_scores, bound_rank_influence, trust_weight


def test_lambda_is_one_when_ml_does_not_beat_baseline() -> None:
    """No OOS edge -> the ML earns zero weight (pure rules)."""
    assert trust_weight(ic_ml_oos=0.03, ic_baseline_oos=0.05) == 1.0
    assert trust_weight(ic_ml_oos=0.05, ic_baseline_oos=0.05) == 1.0


def test_lambda_decreases_as_edge_grows_but_stays_bounded() -> None:
    """A winning ML pulls lambda down toward lambda_min, never below it."""
    small = trust_weight(ic_ml_oos=0.051, ic_baseline_oos=0.05)
    large = trust_weight(ic_ml_oos=0.20, ic_baseline_oos=0.05)
    assert 0.5 <= large < small <= 1.0
    assert large == 0.5  # a big, stable edge saturates at lambda_min


def test_min_edge_benches_a_phantom_edge() -> None:
    """A tiny edge below min_edge is treated as noise -> pure rules (lambda = 1)."""
    # Edge of +0.01 looks like a win, but below a 0.02 margin it earns nothing.
    assert trust_weight(ic_ml_oos=0.01, ic_baseline_oos=0.0, min_edge=0.02) == 1.0
    # The same edge above the margin does earn weight.
    assert trust_weight(ic_ml_oos=0.05, ic_baseline_oos=0.0, min_edge=0.02) < 1.0


def test_lambda_min_is_configurable() -> None:
    """The floor on rules weight is a pre-registered parameter."""
    lam = trust_weight(ic_ml_oos=0.5, ic_baseline_oos=0.0, lambda_min=0.7)
    assert lam == 0.7


def test_blend_with_lambda_one_returns_rules() -> None:
    """lambda = 1 reproduces the rules score exactly (ML has no say)."""
    rules = pd.Series([0.1, -0.2, 0.3], index=["A", "B", "C"])
    ml = pd.Series([0.9, 0.9, -0.9], index=["A", "B", "C"])
    blended = blend_scores(rules, ml, lambda_weight=1.0)
    pd.testing.assert_series_equal(blended, rules, check_names=False)


def test_blend_is_convex_combination() -> None:
    """The blended score is lambda*rules + (1-lambda)*ml."""
    rules = pd.Series([1.0, 0.0])
    ml = pd.Series([0.0, 1.0])
    blended = blend_scores(rules, ml, lambda_weight=0.75)
    assert np.allclose(blended.to_numpy(), [0.75, 0.25])


def test_bound_rank_influence_caps_deviation() -> None:
    """No governed rank strays more than k_max percentiles from the rules rank."""
    rng = np.random.default_rng(0)
    names = [f"N{i}" for i in range(100)]
    rules = pd.Series(rng.normal(size=100), index=names)
    # An ML score that reshuffles the ranking aggressively.
    ml = pd.Series(rng.normal(size=100), index=names)
    blended = blend_scores(rules, ml, lambda_weight=0.5)
    k_max = 20.0
    # The governed score is the bounded percentile, so compare it directly.
    governed = bound_rank_influence(blended, rules, k_max_pctile=k_max)
    rules_pct = rules.rank(pct=True) * 100.0
    assert (np.abs(governed - rules_pct) <= k_max + 1e-9).all()


def test_bound_rank_influence_is_noop_when_already_within_band() -> None:
    """If the ML barely moves ranks, bounding leaves the ordering intact."""
    names = [f"N{i}" for i in range(50)]
    rules = pd.Series(np.arange(50.0), index=names)
    blended = rules + 0.01  # same ordering
    governed = bound_rank_influence(blended, rules, k_max_pctile=20.0)
    assert list(governed.rank()) == list(blended.rank())
