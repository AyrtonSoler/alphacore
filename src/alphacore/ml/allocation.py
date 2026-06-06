"""ML allocation overlay — offline-trained sizing within the Q+M selection (§5.7).

The overlay never changes *which* names are held — the Quality + Momentum residual alpha
owns selection. It only decides *how the money is split* across the already-selected
names, by tilting away from equal weight toward names a model expects to earn a higher
forward return.

Two-phase by design (the model runs *before* the strategy, its output is frozen):

- **Phase A (offline, this module's :func:`predict_forward_returns`)** trains a glass-box
  walk-forward model on each name's *features* (never its ticker identity, so the policy
  transfers to any future name / universe state) to predict its forward return, and emits
  a frozen date x name prediction panel. Done once, ahead of any run.
- **Phase B (run time, :func:`tilt_weights`)** is pure deterministic arithmetic on those
  frozen predictions — no model inference. The strategy run only *reads* the artifact.

The tilt is bounded and trust-gated: ``lambda`` (the tilt strength) is earned out of
sample, and ``lambda = 0`` reproduces the current equal-weight strategy exactly, so the
overlay can never do worse than equal weight when the model has no demonstrated edge.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .metrics import information_coefficient
from .scorers import LinearScorer

# Below this cross-sectional spread the predictions are effectively identical; tilting
# would divide ~0 deviations by a ~0 std and amplify floating-point noise into a spurious
# tilt, so we fall back to equal weight.
_SPREAD_EPS = 1e-12

# Minimum cross-section size for a usable prediction / IC, and minimum OOS-IC history
# before the trust gate will award any tilt.
_MIN_NAMES = 2
_MIN_IC_HISTORY = 3


@dataclasses.dataclass(frozen=True)
class AllocatorParams:
    """Tunable configuration of the ML allocation overlay.

    Attributes
    ----------
    lambda_max:
        Cap on the tilt strength. With ``lambda_max * z_clip < 1`` the multiplier
        ``1 + lambda*z`` stays strictly positive, so every selected name is kept.
    gain:
        Sensitivity of the earned tilt to the out-of-sample information-coefficient
        edge (see the Trust Governor).
    min_edge:
        Minimum OOS IC edge the model must beat before it earns any tilt.
    min_train_periods:
        Periods of history before the model starts predicting (walk-forward warm-up).
    refit_every:
        Refit cadence, in periods (cost control).
    purge_periods:
        Periods dropped between the train window and the prediction date so a label
        cannot leak into training. Must be >= the target's realization horizon (e.g. 4
        for a forward-1-year economic-gain label on a quarterly grid).
    cap_per_name:
        Per-name weight cap (the official 12.5%).
    cash:
        Cash held out of the invested budget; weights sum to ``1 - cash``.
    z_clip:
        Clip on the cross-sectional prediction z-score, bounding the tilt per name.
    random_state:
        Seed for the model (reproducibility).

    """

    lambda_max: float = 0.30
    gain: float = 5.0
    min_edge: float = 0.0
    min_train_periods: int = 24
    refit_every: int = 6
    purge_periods: int = 1
    cap_per_name: float = 0.125
    cash: float = 0.01
    z_clip: float = 3.0
    random_state: int = 42


def predict_forward_returns(
    features: Mapping[str, pd.DataFrame],
    target: pd.DataFrame,
    params: AllocatorParams,
) -> pd.DataFrame:
    """Frozen Phase-A artifact: walk-forward out-of-sample forward-return predictions.

    For each date past the warm-up, an ElasticNet is (re)fit on the stacked feature/target
    history *strictly before* that date (with a one-period purge) and used to predict every
    name's forward return on that date. The model learns from *features*, never ticker
    identity, so the policy transfers to names unseen in training. Trained offline; the
    result is a constant the strategy run reads, never recomputes.

    Parameters
    ----------
    features:
        Feature name -> (date x name) panel.
    target:
        Forward return (date x name): the value realized *after* each date, the label.
    params:
        Allocator configuration (``min_train_periods``, ``refit_every``, ``random_state``).

    Returns
    -------
    pandas.DataFrame
        Date x name out-of-sample predictions; NaN before the warm-up or where a name has
        no complete feature row.

    """
    dates = list(target.index)
    predictions = pd.DataFrame(index=target.index, columns=target.columns, dtype=float)
    scorer: LinearScorer | None = None
    for k, date in enumerate(dates):
        if k < params.min_train_periods:
            continue
        if scorer is None or (k - params.min_train_periods) % params.refit_every == 0:
            x_train, y_train = _training_matrix(features, target, dates[: k - params.purge_periods])
            if len(x_train) >= _MIN_NAMES:
                scorer = LinearScorer(random_state=params.random_state).fit(x_train, y_train)
        if scorer is None:
            continue
        x_today = _features_at(features, date)
        if not x_today.empty:
            predictions.loc[date, x_today.index] = scorer.predict(x_today).to_numpy()
    return predictions


def earned_lambda(
    predictions: pd.DataFrame,
    target: pd.DataFrame,
    params: AllocatorParams,
) -> pd.Series[float]:
    """Per-date tilt strength in ``[0, lambda_max]``, earned out of sample and causally.

    The baseline is equal weight, which has no per-name preference (information coefficient
    0). At each date the gate looks only at the realized OOS information coefficients of
    *earlier* predictions; once enough history accrues, the tilt grows with the average
    edge over ``min_edge`` (scaled by ``gain``) up to ``lambda_max``. Until the model has
    demonstrably beaten equal weight, the tilt is 0 and the strategy stays equal-weight.

    Parameters
    ----------
    predictions:
        The frozen Phase-A prediction panel (date x name).
    target:
        Realized forward returns (date x name).
    params:
        Allocator configuration (``gain``, ``min_edge``, ``lambda_max``).

    Returns
    -------
    pandas.Series
        Tilt strength indexed by date, in ``[0, lambda_max]``.

    """
    lam = pd.Series(0.0, index=predictions.index, dtype=float)
    history: list[float] = []
    for date in predictions.index:
        if len(history) >= _MIN_IC_HISTORY:
            edge = float(np.mean(history))
            trust = min(max(params.gain * (edge - params.min_edge), 0.0), 1.0)
            lam.loc[date] = params.lambda_max * trust
        predicted = predictions.loc[date].dropna()
        if len(predicted) >= _MIN_NAMES:
            history.append(information_coefficient(predicted, target.loc[date]))
    return lam


def _training_matrix(
    features: Mapping[str, pd.DataFrame],
    target: pd.DataFrame,
    train_dates: Sequence[pd.Timestamp],
) -> tuple[pd.DataFrame, pd.Series[float]]:
    """Stack the feature/target panels over ``train_dates`` into an (X, y) design."""
    columns = {name: panel.loc[list(train_dates)].stack() for name, panel in features.items()}
    design = pd.DataFrame(columns)
    design["__target__"] = target.loc[list(train_dates)].stack()
    design = design.dropna()
    return design[list(features)], design["__target__"]


def _features_at(features: Mapping[str, pd.DataFrame], date: pd.Timestamp) -> pd.DataFrame:
    """Per-name feature matrix on ``date`` (complete rows only)."""
    return pd.DataFrame({name: panel.loc[date] for name, panel in features.items()}).dropna()


def tilt_weights(
    predictions: pd.Series[float],
    lam: float,
    params: AllocatorParams,
) -> pd.Series[float]:
    """Distribute the invested budget across selected names by tilting on predictions.

    Pure, deterministic arithmetic on the frozen predictions — no model inference. The
    invested budget ``1 - cash`` is split as ``base * (1 + lam*z)`` per name, where
    ``base`` is the equal weight and ``z`` is the clipped cross-sectional z-score of the
    predicted forward returns, then clipped to the per-name cap and renormalized.

    Parameters
    ----------
    predictions:
        Predicted forward return per selected name (the frozen Phase-A output).
    lam:
        Earned tilt strength in ``[0, lambda_max]``. ``0`` => exact equal weight.
    params:
        Allocator configuration (cap, cash, z-clip).

    Returns
    -------
    pandas.Series
        Weights indexed by name, each in ``(0, cap]``, summing to ``1 - cash``. With
        ``lam == 0`` or zero cross-sectional spread, every weight is the equal weight.

    """
    names = predictions.index
    n = len(names)
    invested = 1.0 - params.cash
    if n == 0:
        return pd.Series(dtype=float)
    base = invested / n
    spread = float(predictions.std(ddof=0))
    if lam <= 0.0 or spread <= _SPREAD_EPS:
        return pd.Series(base, index=names, dtype=float)
    z = ((predictions - predictions.mean()) / spread).clip(-params.z_clip, params.z_clip)
    raw = (base * (1.0 + lam * z)).clip(lower=0.0)
    return _normalize_with_cap(raw, invested, params.cap_per_name)


def _normalize_with_cap(weights: pd.Series[float], budget: float, cap: float) -> pd.Series[float]:
    """Scale non-negative weights to ``budget`` with no name exceeding ``cap``.

    Water-filling: repeatedly rescale the free (un-capped) names to absorb the budget;
    any that would exceed ``cap`` are pinned at ``cap`` and the rest re-share the
    remainder, until no free name exceeds the cap.
    """
    result = weights.clip(lower=0.0).astype(float)
    free = pd.Series(data=True, index=result.index)
    fixed_total = 0.0
    for _ in range(len(result) + 1):
        free_sum = float(result[free].sum())
        remaining = budget - fixed_total
        if free_sum <= 0.0:
            break
        scaled = result[free] * (remaining / free_sum)
        over = scaled.index[scaled > cap]
        if len(over) == 0:
            result[free] = scaled
            break
        result[over] = cap
        free[over] = False
        fixed_total += cap * len(over)
    return result
