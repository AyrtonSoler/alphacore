"""RED tests for ml.governed — the governed alpha panel (§5.5/§5.6 wired).

Walk-forward: an ElasticNet ML model is trained (point-in-time) on the sleeve-
residual features to predict the residual forward return, and the Trust Governor
blends it with the rules score only to the extent it beats the rules out-of-sample
(expanding-window IC). The output is a date x name governed-alpha panel (in
cross-sectional rank units, so its scale is consistent across dates).

Acceptance
----------
- When the target is a learnable signal the rules under-weight, the ML earns weight
  and the governed panel differs from the rules ranking on the out-of-sample dates.
- When the target is pure noise, the governor benches the ML and the governed panel
  matches the rules ranking.
- Early dates (insufficient history) fall back to the rules ranking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.governed import build_governed_alpha

_DATES = pd.date_range("2018-01-31", periods=40, freq="ME")
_NAMES = [f"N{i:02d}" for i in range(30)]


def _panel(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(40, 30)), index=_DATES, columns=_NAMES)


def _rank(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rank(axis=1, pct=True)


def test_ml_earns_weight_on_a_learnable_target() -> None:
    """If the target tracks a feature the rules ignore, governed != rules OOS."""
    f_a = _panel(1)
    f_b = _panel(2)
    features = {"a": f_a, "b": f_b}
    rules = f_a  # the rules only use feature 'a'
    target = f_b + 0.05 * _panel(3)  # but the future tracks feature 'b'
    governed = build_governed_alpha(rules, features, target, min_train_periods=12, refit_every=6)
    oos = _DATES[20:]  # well past the training warm-up
    rules_rank = _rank(rules)
    diff = (governed.loc[oos] - rules_rank.loc[oos]).abs().to_numpy()
    assert np.nanmax(diff) > 1e-6  # the ML moved the scores


def test_ml_is_benched_on_noise_target() -> None:
    """A pure-noise target earns the ML no weight -> governed == rules ranking."""
    f_a, f_b = _panel(1), _panel(2)
    rules = f_a
    target = _panel(99)  # independent of the features
    governed = build_governed_alpha(
        rules, {"a": f_a, "b": f_b}, target, min_train_periods=12, refit_every=6, min_edge=0.02
    )
    rules_rank = _rank(rules)
    pd.testing.assert_frame_equal(governed, rules_rank, atol=1e-9)


def test_output_is_rank_units_and_dated_like_rules() -> None:
    """The governed panel shares the rules index/columns and is in [0, 1] rank units."""
    f_a, f_b = _panel(1), _panel(2)
    governed = build_governed_alpha(f_a, {"a": f_a, "b": f_b}, _panel(5), min_train_periods=12)
    assert list(governed.index) == list(_DATES)
    assert list(governed.columns) == _NAMES
    valid = governed.to_numpy()[~np.isnan(governed.to_numpy())]
    assert valid.min() >= 0.0
    assert valid.max() <= 1.0
