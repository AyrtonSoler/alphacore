"""TDD for ml.allocation — the offline ML allocation overlay.

The overlay never changes which names are held (Q+M selection owns that). It only
distributes the invested budget across the already-selected names, using a frozen,
offline-trained prediction of each name's forward return. ``tilt_weights`` is the pure
run-time core: deterministic arithmetic on the frozen predictions, no model inference.

Invariants under test:
- lambda = 0 (no earned trust) reproduces the current equal-weight strategy exactly.
- higher predicted return -> more money (monotone), bounded by the per-name cap.
- weights sum to the invested budget (1 - cash) and every selected name is kept (> 0).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from alphacore.ml.allocation import (
    AllocatorParams,
    earned_lambda,
    predict_forward_returns,
    tilt_weights,
)
from alphacore.ml.metrics import information_coefficient

_PARAMS = AllocatorParams()


def _synth_panel(
    n_dates: int = 36, n_names: int = 20, seed: int = 42
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DatetimeIndex]:
    """A panel where forward return depends linearly on a feature (a learnable signal)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-31", periods=n_dates, freq="ME")
    names = [f"N{i}" for i in range(n_names)]
    feature = pd.DataFrame(rng.standard_normal((n_dates, n_names)), index=dates, columns=names)
    noise = pd.DataFrame(rng.standard_normal((n_dates, n_names)), index=dates, columns=names)
    target = 0.5 * feature + 0.5 * noise
    return {"f": feature}, target, dates


def test_allocator_params_defaults_keep_tilt_bounded_positive() -> None:
    """lambda_max * z_clip < 1 so the multiplier (1 + lam*z) can never zero a name."""
    assert _PARAMS.lambda_max * _PARAMS.z_clip < 1.0
    assert _PARAMS.cap_per_name == 0.125
    assert _PARAMS.cash == 0.01


def test_tilt_weights_lambda_zero_is_equal_weight() -> None:
    """No earned trust (lambda=0) => exact equal weight of the invested budget."""
    preds = pd.Series([0.1, -0.2, 0.05], index=["A", "B", "C"])
    weights = tilt_weights(preds, lam=0.0, params=_PARAMS)
    invested = 1.0 - _PARAMS.cash
    assert weights.to_numpy() == pytest.approx(np.full(3, invested / 3))
    assert weights.sum() == pytest.approx(invested)


def test_tilt_weights_equal_predictions_is_equal_weight() -> None:
    """Zero cross-sectional spread => no tilt, even with lambda > 0."""
    preds = pd.Series([0.2, 0.2, 0.2], index=["A", "B", "C"])
    weights = tilt_weights(preds, lam=_PARAMS.lambda_max, params=_PARAMS)
    assert weights.to_numpy() == pytest.approx(np.full(3, (1.0 - _PARAMS.cash) / 3))


def test_tilt_weights_monotone_in_prediction() -> None:
    """Higher predicted return -> strictly higher weight; all names kept; sums to budget."""
    # Cap raised so it does not bind (4 names cannot share 99% under a 12.5% cap); the
    # cap behaviour has its own test below.
    params = dataclasses.replace(_PARAMS, cap_per_name=0.50)
    preds = pd.Series([0.30, 0.10, -0.05, -0.20], index=["A", "B", "C", "D"])
    weights = tilt_weights(preds, lam=params.lambda_max, params=params)
    assert weights["A"] > weights["B"] > weights["C"] > weights["D"]
    assert (weights > 0).all()
    assert weights.sum() == pytest.approx(1.0 - _PARAMS.cash)


def test_tilt_weights_respects_cap() -> None:
    """An outlier prediction is clipped to the per-name cap; budget still allocated."""
    params = dataclasses.replace(_PARAMS, cap_per_name=0.30)
    preds = pd.Series([10.0, 0.0, 0.0, 0.0], index=["A", "B", "C", "D"])
    weights = tilt_weights(preds, lam=params.lambda_max, params=params)
    assert (weights <= params.cap_per_name + 1e-9).all()
    assert weights.max() == pytest.approx(params.cap_per_name)
    assert weights.sum() == pytest.approx(1.0 - params.cash)


def test_tilt_weights_is_deterministic() -> None:
    """Pure arithmetic: identical inputs => byte-identical outputs."""
    preds = pd.Series([0.3, 0.1, -0.2], index=["A", "B", "C"])
    first = tilt_weights(preds, lam=0.25, params=_PARAMS)
    second = tilt_weights(preds, lam=0.25, params=_PARAMS)
    pd.testing.assert_series_equal(first, second)


def test_predict_forward_returns_warmup_then_skillful() -> None:
    """No predictions before warm-up; afterwards they have positive OOS skill (IC>0)."""
    features, target, dates = _synth_panel()
    params = dataclasses.replace(_PARAMS, min_train_periods=12, refit_every=6)
    preds = predict_forward_returns(features, target, params)

    assert preds.loc[dates[:12]].isna().all().all()
    ics = [
        information_coefficient(preds.loc[d].dropna(), target.loc[d])
        for d in dates[12:]
        if preds.loc[d].notna().any()
    ]
    assert len(ics) > 0
    assert float(np.mean(ics)) > 0.1


def test_predict_forward_returns_is_causal() -> None:
    """A prediction never depends on a target realized on or after its own date."""
    features, target, dates = _synth_panel()
    params = dataclasses.replace(_PARAMS, min_train_periods=12, refit_every=6)
    preds = predict_forward_returns(features, target, params)

    perturbed = target.copy()
    perturbed.iloc[20] = perturbed.iloc[20] + 100.0
    preds_perturbed = predict_forward_returns(features, perturbed, params)

    # A prediction before the perturbed date is unchanged (trained only on earlier data)...
    pd.testing.assert_series_equal(preds.loc[dates[17]], preds_perturbed.loc[dates[17]])
    # ...while a later prediction (whose model trained on the perturbed date) does change.
    assert not preds.loc[dates[28]].equals(preds_perturbed.loc[dates[28]])


def test_allocator_params_purge_default_is_one() -> None:
    """The default purge is one period (the quarterly grid's minimal gap)."""
    assert AllocatorParams().purge_periods == 1


def test_predict_forward_returns_respects_purge_periods() -> None:
    """A purge wider than the history leaves no training data, so nothing is predicted.

    This guards the long-horizon target case: a forward-1-year economic-gain label must
    not leak, so the purge is widened to cover the label horizon.
    """
    features, target, _ = _synth_panel()
    params = dataclasses.replace(_PARAMS, min_train_periods=12, purge_periods=999)
    preds = predict_forward_returns(features, target, params)
    assert preds.isna().all().all()


def test_predict_forward_returns_is_seeded() -> None:
    """Same inputs and seed => identical frozen prediction panel."""
    features, target, _ = _synth_panel()
    params = dataclasses.replace(_PARAMS, min_train_periods=12)
    pd.testing.assert_frame_equal(
        predict_forward_returns(features, target, params),
        predict_forward_returns(features, target, params),
    )


def test_earned_lambda_bounded_causal_and_rises_with_skill() -> None:
    """lambda in [0, lambda_max], zero with no IC history, positive once skill accrues."""
    features, target, _ = _synth_panel()
    params = dataclasses.replace(_PARAMS, min_train_periods=12, refit_every=6)
    preds = predict_forward_returns(features, target, params)
    lam = earned_lambda(preds, target, params)

    assert (lam >= 0.0).all()
    assert (lam <= params.lambda_max + 1e-12).all()
    assert lam.iloc[0] == 0.0  # no out-of-sample IC history yet
    assert lam.iloc[-1] > 0.0  # a genuinely skillful model earns a tilt by the end
