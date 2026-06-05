"""ML evaluation metrics — the information coefficient (v3 §5.5/§8).

The information coefficient (IC) is the cross-sectional Spearman rank correlation
between a model's predicted score and the realized residual return. It is the single
number the falsifiable linear-vs-non-linear experiment is judged on out-of-sample
(§8), and the quantity the Trust Governor (§5.6) compares to set the ML's blend
weight. Being rank-based, it is robust to the score's scale and to outliers.

References
----------
.. [Grinold1989] Grinold, R. C. (1989). "The Fundamental Law of Active Management."
   Journal of Portfolio Management 15(3), 30-37. (IC as skill measure.)

"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_MIN_PAIRS = 2


def information_coefficient(
    predicted: pd.Series[float],
    realized: pd.Series[float],
) -> float:
    """Cross-sectional Spearman rank correlation of predictions vs outcomes.

    Parameters
    ----------
    predicted:
        Predicted scores, indexed by name.
    realized:
        Realized residual returns (or any outcome), indexed by name.

    Returns
    -------
    float
        Spearman rank correlation over the names present in both series. Returns
        0.0 when fewer than two paired, finite observations are available or the
        correlation is undefined (no dispersion).

    References
    ----------
    The information coefficient from [Grinold1989]_.

    """
    paired = pd.concat([predicted.rename("p"), realized.rename("r")], axis=1).dropna()
    if len(paired) < _MIN_PAIRS:
        return 0.0
    rho = spearmanr(paired["p"].to_numpy(), paired["r"].to_numpy()).statistic
    return 0.0 if np.isnan(rho) else float(rho)
