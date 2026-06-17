"""RED tests for common.io — parquet I/O and versioned cache keys.

Acceptance: cache_key(name, params, version) is stable and changes when params change.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from common.io import cache_key, read_parquet, write_parquet


def test_cache_key_is_stable_across_param_ordering() -> None:
    """Same logical inputs produce the same key regardless of dict insertion order."""
    k1 = cache_key("signals", {"window": 14, "alpha": 0.3}, version="v1")
    k2 = cache_key("signals", {"alpha": 0.3, "window": 14}, version="v1")
    assert k1 == k2


def test_cache_key_changes_when_params_change() -> None:
    """Any change to name, params, or version yields a different key."""
    base = cache_key("signals", {"window": 14}, version="v1")
    assert cache_key("signals", {"window": 21}, version="v1") != base
    assert cache_key("signals", {"window": 14}, version="v2") != base
    assert cache_key("momentum", {"window": 14}, version="v1") != base


def test_parquet_roundtrip_preserves_frame(tmp_path: Path) -> None:
    """write_parquet then read_parquet returns an identical DataFrame."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    path = tmp_path / "frame.parquet"

    write_parquet(df, path)
    out = read_parquet(path)

    pd.testing.assert_frame_equal(df, out)


def test_write_parquet_creates_parent_dirs(tmp_path: Path) -> None:
    """Writing into a not-yet-existing subdirectory creates the path."""
    df = pd.DataFrame({"a": [1]})
    path = tmp_path / "nested" / "deeper" / "frame.parquet"

    write_parquet(df, path)

    assert path.is_file()
