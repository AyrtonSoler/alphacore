"""Parquet I/O and deterministic, versioned cache keys.

Parquet is the canonical on-disk format for every intermediate artifact (curated
data, signals, portfolios). :func:`cache_key` derives a stable identifier from the
*inputs* that produced an artifact so reproductions are byte-addressable: change a
parameter or bump the data version and the key changes, invalidating stale caches.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Mapping

_DIGEST_LEN = 16


def cache_key(name: str, params: Mapping[str, object], version: str) -> str:
    """Derive a stable cache key from an artifact's name, params, and version.

    Parameters
    ----------
    name:
        Logical artifact name (e.g. ``"signals"``).
    params:
        Parameters that produced the artifact. Serialised order-independently, so
        ``{"a": 1, "b": 2}`` and ``{"b": 2, "a": 1}`` yield the same key.
    version:
        Data/schema version tag. Bumping it invalidates all prior keys.

    Returns
    -------
    str
        ``"{name}-{version}-{16-hex-digest}"`` — stable across runs and machines.

    Notes
    -----
    Hashing is over a ``sort_keys=True`` JSON payload, so the key is deterministic
    and independent of dict insertion order. Non-JSON-native values fall back to
    their ``str`` representation.

    """
    payload = json.dumps(
        {"name": name, "params": params, "version": version},
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{name}-{version}-{digest[:_DIGEST_LEN]}"


def read_parquet(path: Path | str) -> pd.DataFrame:
    """Read a parquet file into a DataFrame.

    Parameters
    ----------
    path:
        Path to a parquet file.

    Returns
    -------
    pandas.DataFrame
        The deserialised frame.

    """
    return pd.read_parquet(path)


def write_parquet(df: pd.DataFrame, path: Path | str) -> None:
    """Write a DataFrame to parquet, creating parent directories as needed.

    Parameters
    ----------
    df:
        Frame to persist.
    path:
        Destination path. Missing parent directories are created.

    Returns
    -------
    None

    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(destination)
