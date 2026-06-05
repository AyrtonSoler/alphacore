"""RED tests for ml.experiment — the governed linear-vs-EBM decision (§5.5/§8).

Ties the pieces together: run a scorer across the purged walk-forward folds, score
each held-out fold by IC, and average. The Trust Governor (§5.6) then compares the
EBM's mean OOS IC to the linear baseline's to decide the ML's blend weight — so the
ML earns weight only when it genuinely beats the baseline out-of-sample.

Acceptance
----------
- One OOS IC per fold.
- A linear target yields a positive mean OOS IC for the linear scorer.
- On a pure interaction the EBM's mean OOS IC beats the linear baseline's, and the
  resulting governor lambda is < 1 (the ML earns weight).
- On a noise target both score ~0 and lambda = 1 (the ML is benched — graceful
  degradation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.experiment import mean_oos_ic, walk_forward_ic
from ml.scorers import EbmScorer, LinearScorer
from ml.trust_governor import trust_weight
from robustness.cross_validation import purged_walk_forward_splits

_COLS = ["x0", "x1", "x2"]


def _features(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.uniform(-1.0, 1.0, size=(n, 3)), columns=_COLS)


def _splits(n: int):
    return purged_walk_forward_splits(n, min_train_size=300, test_size=150, label_horizon=1)


def test_walk_forward_ic_returns_one_value_per_fold() -> None:
    """The harness produces exactly one out-of-sample IC per fold."""
    n = 600
    x = _features(n, seed=0)
    y = pd.Series(x["x0"].to_numpy())
    splits = _splits(n)
    ics = walk_forward_ic(LinearScorer, x, y, splits)
    assert len(ics) == len(splits)
    assert len(ics) >= 1


def test_linear_target_gives_positive_mean_ic() -> None:
    """A linear signal is learned out-of-sample by the linear scorer."""
    n = 600
    x = _features(n, seed=1)
    rng = np.random.default_rng(20)
    y = pd.Series(x.to_numpy() @ [2.0, -1.0, 0.5] + 0.05 * rng.normal(size=n))
    assert mean_oos_ic(LinearScorer, x, y, _splits(n)) > 0.5


@pytest.mark.slow
def test_interaction_lets_ebm_earn_governor_weight() -> None:
    """EBM beats the linear baseline OOS on an interaction, so lambda < 1."""
    n = 600
    x = _features(n, seed=2)
    rng = np.random.default_rng(21)
    y = pd.Series(x["x0"].to_numpy() * x["x1"].to_numpy() + 0.02 * rng.normal(size=n))
    splits = _splits(n)
    ic_linear = mean_oos_ic(LinearScorer, x, y, splits)
    ic_ebm = mean_oos_ic(lambda: EbmScorer(interactions=3, outer_bags=2), x, y, splits)
    assert ic_ebm > ic_linear
    assert trust_weight(ic_ml_oos=ic_ebm, ic_baseline_oos=ic_linear) < 1.0


@pytest.mark.slow
def test_noise_target_benches_the_ml() -> None:
    """With no learnable signal both score ~0 and the governor runs pure rules."""
    n = 600
    x = _features(n, seed=3)
    rng = np.random.default_rng(22)
    y = pd.Series(rng.normal(size=n))  # independent of the features
    splits = _splits(n)
    ic_linear = mean_oos_ic(LinearScorer, x, y, splits)
    ic_ebm = mean_oos_ic(lambda: EbmScorer(interactions=3, outer_bags=2), x, y, splits)
    # Both ICs are ~0; requiring a margin benches the ML rather than trusting noise.
    assert trust_weight(ic_ml_oos=ic_ebm, ic_baseline_oos=ic_linear, min_edge=0.02) == 1.0


def test_mean_oos_ic_of_empty_splits_is_zero() -> None:
    """No folds -> no evidence -> a neutral zero IC."""
    x = _features(50, seed=4)
    y = pd.Series(x["x0"].to_numpy())
    assert mean_oos_ic(LinearScorer, x, y, []) == 0.0
