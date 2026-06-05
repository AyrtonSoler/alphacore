"""Purged, embargoed walk-forward validation (v3 §8).

Replaces v2's naive walk-forward with the leakage-aware expanding scheme of Lopez
de Prado: the test block always follows the training history chronologically, and
the training observations whose forward label (over a horizon of ``label_horizon``
rebalances) would overlap the test window — plus an additional ``embargo`` gap — are
purged. Without the purge the overlapping label silently leaks future information
into the fit and inflates out-of-sample performance.

The splitter is a pure function over integer period positions (rebalancing dates),
so it is deterministic and independent of the panel's data.

References
----------
.. [LdP2018] Lopez de Prado, M. (2018). Advances in Financial Machine Learning.
   Wiley. Chapter 7 (purging and embargoing in cross-validation).

"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_IndexArray = NDArray[np.intp]


def purged_walk_forward_splits(
    n_samples: int,
    *,
    min_train_size: int,
    test_size: int,
    label_horizon: int = 1,
    embargo: int = 0,
) -> list[tuple[_IndexArray, _IndexArray]]:
    """Generate purged, embargoed expanding walk-forward train/test folds.

    Parameters
    ----------
    n_samples:
        Number of ordered periods (e.g. monthly rebalancing dates), positions
        ``0 .. n_samples - 1``.
    min_train_size:
        Position at which the first test block begins; the first (pre-purge) train
        spans ``[0, min_train_size)``.
    test_size:
        Number of consecutive periods in each test block.
    label_horizon:
        Forward horizon of the prediction label, in periods. The last
        ``label_horizon`` train periods before a test block overlap its labels and
        are purged. Default 1.
    embargo:
        Extra periods purged immediately before each test block, beyond the horizon,
        to buffer residual serial correlation. Default 0.

    Returns
    -------
    list of (train_index, test_index)
        Each a pair of integer numpy arrays. The test block is
        ``[train_end, train_end + test_size)`` and the train is
        ``[0, train_end - label_horizon - embargo)``; ``train_end`` advances by
        ``test_size`` each fold. Folds whose test block would exceed ``n_samples``
        are dropped, so an insufficiently long history yields an empty list.

    References
    ----------
    Purging and embargoing from [LdP2018]_.

    """
    gap = label_horizon + embargo
    folds: list[tuple[_IndexArray, _IndexArray]] = []
    train_end = min_train_size
    while train_end + test_size <= n_samples:
        train_stop = train_end - gap
        if train_stop > 0:
            train_index = np.arange(0, train_stop)
            test_index = np.arange(train_end, train_end + test_size)
            folds.append((train_index, test_index))
        train_end += test_size
    return folds
