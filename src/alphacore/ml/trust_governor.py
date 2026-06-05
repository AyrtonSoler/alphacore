"""Trust Governor — the ML distrust, encoded as a mechanism (v3 §5.6).

The ML layer never drives the portfolio blindly. The governor boxes it in from
both sides with four locks; this module implements the two that are pure score
arithmetic:

- **Lock 2 — weight earned out-of-sample.** The non-linear ML score blends into the
  final score with weight ``1 - lambda`` only if it beat the linear baseline's
  information coefficient OOS. If it did not, ``lambda = 1`` and the strategy runs
  entirely on the rules, degrading gracefully — this is what lets v3 ship *without*
  ML if the non-linearity turns out not to exist.
- **Lock 3 — bounded influence.** The ML may shift a name's rank by at most
  ``k_max`` percentiles from its rules rank, so there are no extreme, inexplicable
  bets.

Lock 1 (monotonic constraints) is enforced when the EBM is trained; lock 5 (the ML
adjusts the *score*, never the *weights* — those come from the factor-neutral
optimizer, §5.8) is structural. Together they keep the ML caged on two sides.

References
----------
.. [Grinold1989] Grinold, R. C. (1989). "The Fundamental Law of Active Management."
   Journal of Portfolio Management 15(3), 30-37. (The information coefficient.)

"""

from __future__ import annotations

import pandas as pd

_DEFAULT_GAIN = 5.0
_DEFAULT_LAMBDA_MIN = 0.5


def trust_weight(
    ic_ml_oos: float,
    ic_baseline_oos: float,
    *,
    gain: float = _DEFAULT_GAIN,
    lambda_min: float = _DEFAULT_LAMBDA_MIN,
) -> float:
    """Rules weight ``lambda`` in the rules/ML blend, earned out-of-sample.

    Parameters
    ----------
    ic_ml_oos:
        Out-of-sample information coefficient of the non-linear ML score.
    ic_baseline_oos:
        Out-of-sample IC of the linear baseline (the null hypothesis).
    gain:
        Sensitivity of ``lambda`` to the IC edge ``ic_ml - ic_baseline``. Larger
        ``gain`` lets a small but real edge earn weight faster.
    lambda_min:
        Floor on the rules weight: even a large, stable edge keeps at least this much
        of the rules score (the ML never gets more than ``1 - lambda_min``).

    Returns
    -------
    float
        ``lambda`` in ``[lambda_min, 1]``. Exactly 1.0 when the ML did not beat the
        baseline OOS (``ic_ml <= ic_baseline``); otherwise ``1 - min(gain * edge,
        1 - lambda_min)``, decreasing toward ``lambda_min`` as the edge grows.

    References
    ----------
    Information coefficient from [Grinold1989]_.

    """
    edge = ic_ml_oos - ic_baseline_oos
    if edge <= 0.0:
        return 1.0
    ml_share = min(gain * edge, 1.0 - lambda_min)
    return 1.0 - ml_share


def blend_scores(
    score_rules: pd.Series[float],
    score_ml: pd.Series[float],
    lambda_weight: float,
) -> pd.Series[float]:
    """Convex blend of the rules and ML scores: ``lambda*rules + (1-lambda)*ml``.

    Parameters
    ----------
    score_rules, score_ml:
        Aligned per-name scores from the rules and the ML model.
    lambda_weight:
        The rules weight from :func:`trust_weight`. ``1.0`` reproduces the rules
        score exactly.

    Returns
    -------
    pandas.Series
        ``lambda_weight * score_rules + (1 - lambda_weight) * score_ml``.

    """
    return lambda_weight * score_rules + (1.0 - lambda_weight) * score_ml


def bound_rank_influence(
    blended: pd.Series[float],
    score_rules: pd.Series[float],
    k_max_pctile: float,
) -> pd.Series[float]:
    """Cap how far the ML may move a name's rank from its rules rank (lock 3).

    Parameters
    ----------
    blended:
        The blended score from :func:`blend_scores`.
    score_rules:
        The pure-rules score, defining the reference ranking.
    k_max_pctile:
        Maximum allowed deviation, in percentile points (0-100), of a name's
        governed rank from its rules rank.

    Returns
    -------
    pandas.Series
        The governed score as a percentile (0-100): the blended percentile clipped
        into ``[rules_pctile - k_max, rules_pctile + k_max]`` per name. By
        construction no name's governed percentile strays more than ``k_max`` from
        its rules percentile, so the ML cannot make an extreme, inexplicable bet.

    """
    rules_pct = score_rules.rank(pct=True) * 100.0
    blended_pct = blended.rank(pct=True) * 100.0
    return blended_pct.clip(lower=rules_pct - k_max_pctile, upper=rules_pct + k_max_pctile)
