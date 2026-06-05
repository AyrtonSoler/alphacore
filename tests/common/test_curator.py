"""RED tests for common.curator — parallel-download thread sizing and resume-skip.

Acceptance:
- resolve_max_threads caps the default conservatively for the FMP rate limit, honours
  the KNDC_MAX_THREADS override, never exceeds the identifier count, and rejects junk.
- pending_identifiers drops tickers whose ``<id>.parquet`` already exists so a crashed
  or rate-limited run resumes without re-fetching completed tickers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from common.curator import DEFAULT_MAX_THREADS, pending_identifiers, resolve_max_threads


def test_resolve_max_threads_uses_capped_default_when_env_unset() -> None:
    """With many identifiers and no override, fall back to the conservative cap."""
    assert resolve_max_threads(3660, None) == DEFAULT_MAX_THREADS


def test_resolve_max_threads_never_exceeds_identifier_count() -> None:
    """Never spin up more workers than there are identifiers to download."""
    assert resolve_max_threads(3, None) == 3


def test_resolve_max_threads_honours_env_override() -> None:
    """An explicit KNDC_MAX_THREADS value wins over the default."""
    assert resolve_max_threads(3660, "16") == 16


def test_resolve_max_threads_with_zero_identifiers_is_one() -> None:
    """A degenerate empty universe still yields a valid single worker."""
    assert resolve_max_threads(0, "16") == 1


@pytest.mark.parametrize("bad", ["0", "-4", "abc", "2.5"])
def test_resolve_max_threads_rejects_non_positive_int(bad: str) -> None:
    """A set-but-invalid override is a hard error, not a silent fallback."""
    with pytest.raises(ValueError):  # noqa: PT011
        resolve_max_threads(10, bad)


def test_pending_identifiers_skips_already_downloaded(tmp_path: Path) -> None:
    """Identifiers with an existing ``<id>.parquet`` are dropped; order is preserved."""
    (tmp_path / "AAPL.parquet").touch()
    (tmp_path / "GOOGL.parquet").touch()

    pending = pending_identifiers(("AAPL", "MSFT", "GOOGL", "AMZN"), tmp_path)

    assert pending == ("MSFT", "AMZN")


def test_pending_identifiers_returns_all_when_none_downloaded(tmp_path: Path) -> None:
    """A fresh run (empty output dir) leaves every identifier pending."""
    identifiers = ("AAPL", "MSFT", "GOOGL")
    assert pending_identifiers(identifiers, tmp_path) == identifiers
