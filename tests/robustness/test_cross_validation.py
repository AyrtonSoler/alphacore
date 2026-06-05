"""RED tests for robustness.cross_validation — purged walk-forward (v3 §8).

Replaces the simple walk-forward of v2 with the purged + embargoed expanding scheme
of Lopez de Prado: each test block follows the train chronologically, and the train
observations whose forward label (horizon h) would overlap the test — plus an
embargo gap — are removed, so the overlapping label cannot leak the future.

Acceptance
----------
- Folds are chronological: every train index precedes every test index.
- Train and test never overlap.
- The `label_horizon + embargo` periods immediately before each test block are
  purged out of the train (the leakage guard).
- The train window expands fold over fold.
- The fold count and exact indices match a worked example.
"""

from __future__ import annotations

from robustness.cross_validation import purged_walk_forward_splits


def test_worked_example_default_horizon() -> None:
    """n=20, min_train=8, test=4, horizon=1, embargo=0 -> three expanding folds."""
    folds = purged_walk_forward_splits(
        20, min_train_size=8, test_size=4, label_horizon=1, embargo=0
    )
    assert len(folds) == 3
    train0, test0 = folds[0]
    assert list(train0) == list(range(0, 7))  # index 7 purged (overlaps test start)
    assert list(test0) == [8, 9, 10, 11]
    train1, test1 = folds[1]
    assert list(train1) == list(range(0, 11))
    assert list(test1) == [12, 13, 14, 15]


def test_folds_are_chronological_and_non_overlapping() -> None:
    """Every train index is strictly before every test index, with no overlap."""
    folds = purged_walk_forward_splits(
        50, min_train_size=12, test_size=6, label_horizon=3, embargo=2
    )
    for train, test in folds:
        assert set(train).isdisjoint(set(test))
        assert train.max() < test.min()


def test_purge_and_embargo_gap_is_respected() -> None:
    """The horizon+embargo periods just before each test block are removed."""
    horizon, embargo = 3, 2
    folds = purged_walk_forward_splits(
        50, min_train_size=12, test_size=6, label_horizon=horizon, embargo=embargo
    )
    for train, test in folds:
        # The gap between the last kept train index and the first test index must
        # cover both the label horizon and the embargo.
        gap = int(test.min()) - int(train.max()) - 1
        assert gap == horizon + embargo


def test_train_window_expands() -> None:
    """Each successive fold trains on at least as much history as the previous."""
    folds = purged_walk_forward_splits(
        40, min_train_size=10, test_size=5, label_horizon=1, embargo=0
    )
    sizes = [len(train) for train, _ in folds]
    assert all(b > a for a, b in zip(sizes, sizes[1:], strict=False))


def test_no_fold_when_history_too_short() -> None:
    """If the first test block does not fit, no folds are produced."""
    folds = purged_walk_forward_splits(
        10, min_train_size=8, test_size=4, label_horizon=1, embargo=0
    )
    assert folds == []
