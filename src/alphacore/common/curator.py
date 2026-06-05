"""Helpers for the parallel Data Curator download: thread sizing and resume-skip.

The KN Hack investable universe is ~3,660 tickers pulled from FMP (Starter ≈ 300
calls/min on a single key). Each ticker triggers several endpoint calls, so an
unbounded thread pool bursts past the rate limit and earns 429s. These pure helpers
keep the entry script thin and testable: :func:`resolve_max_threads` chooses a
conservative worker count and :func:`pending_identifiers` filters out tickers already
on disk, so a crashed or throttled run resumes without re-fetching completed names.

Notes
-----
Output convention is one file per identifier, ``<output_base_dir>/<identifier>.parquet``
(plus a sibling ``.csv``); the parquet's presence is treated as the completion marker.

"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

# Conservative default kept well under the FMP Starter limit (~300 calls/min). Override
# with the KNDC_MAX_THREADS environment variable once observed 429 rates allow it.
DEFAULT_MAX_THREADS = 8


def resolve_max_threads(num_identifiers: int, env_value: str | None) -> int:
    """Resolve the worker-thread count for the parallel download.

    Parameters
    ----------
    num_identifiers:
        Number of identifiers to download. The result never exceeds this.
    env_value:
        Raw ``KNDC_MAX_THREADS`` value, or ``None`` when unset. When unset, falls back
        to :data:`DEFAULT_MAX_THREADS`.

    Returns
    -------
    int
        At least 1, and at most ``num_identifiers`` when there is work to do.

    Raises
    ------
    ValueError
        If ``env_value`` is set but is not a positive integer.

    """
    if num_identifiers <= 0:
        return 1

    if env_value is None:
        chosen = DEFAULT_MAX_THREADS
    else:
        chosen = int(env_value)  # ValueError on non-integer strings ("abc", "2.5")
        if chosen < 1:
            message = f"KNDC_MAX_THREADS must be a positive integer, got {env_value!r}"
            raise ValueError(message)

    return max(1, min(chosen, num_identifiers))


def pending_identifiers(
    identifiers: Iterable[str],
    output_base_dir: str | Path,
) -> tuple[str, ...]:
    """Return identifiers whose output parquet does not yet exist, order preserved.

    Parameters
    ----------
    identifiers:
        Full identifier list from the Data Curator configuration.
    output_base_dir:
        Directory the Data Curator writes ``<identifier>.parquet`` files into.

    Returns
    -------
    tuple of str
        The subset still needing a download — a resumed run skips completed tickers.

    """
    base = Path(output_base_dir)
    return tuple(
        identifier for identifier in identifiers if not (base / f"{identifier}.parquet").is_file()
    )
