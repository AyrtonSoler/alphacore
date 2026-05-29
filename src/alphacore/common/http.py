"""HTTP client with exponential backoff and token-bucket rate limiting.

External data providers throttle aggressively (HTTP 429) and fail transiently
(5xx). :class:`Client` wraps :mod:`httpx` with deterministic, jitter-free
exponential backoff and an optional :class:`TokenBucket` so the research lab is a
well-behaved API citizen and reproductions are byte-identical.

Notes
-----
Backoff is intentionally jitter-free (no RNG) to honour the project's determinism
rules. Time and sleeping are injected (``sleep``/``now`` parameters) so tests run
instantly and without real clocks.

"""

from __future__ import annotations

import time as _time
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

# Status codes worth retrying: rate limiting + transient upstream failures.
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


class TokenBucket:
    """A token-bucket rate limiter with an injectable clock.

    Tokens refill continuously at ``rate`` per second up to ``capacity``. Each
    :meth:`acquire` consumes tokens, sleeping only when the bucket is empty.

    Parameters
    ----------
    rate:
        Tokens replenished per second.
    capacity:
        Maximum tokens the bucket holds (burst size).
    now:
        Monotonic clock returning seconds. Injectable for tests.
    sleep:
        Blocking sleep taking seconds. Injectable for tests.

    """

    def __init__(
        self,
        rate: float,
        capacity: float,
        *,
        now: Callable[[], float] = _time.monotonic,
        sleep: Callable[[float], None] = _time.sleep,
    ) -> None:
        """Initialise a full bucket; see the class docstring for parameters."""
        self._rate = rate
        self._capacity = capacity
        self._tokens = capacity
        self._now = now
        self._sleep = sleep
        self._last = now()

    def _refill(self) -> None:
        """Add tokens accrued since the last check, capped at capacity."""
        current = self._now()
        elapsed = current - self._last
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last = current

    def acquire(self, tokens: float = 1.0) -> float:
        """Consume ``tokens``, sleeping if the bucket cannot yet satisfy them.

        Parameters
        ----------
        tokens:
            Number of tokens to consume (default 1).

        Returns
        -------
        float
            Seconds slept (``0.0`` when tokens were immediately available).

        """
        self._refill()
        waited = 0.0
        if self._tokens < tokens:
            deficit = tokens - self._tokens
            waited = deficit / self._rate
            self._sleep(waited)
            self._refill()
        self._tokens -= tokens
        return waited


class Client:
    """An HTTP client that retries throttled/transient responses with backoff.

    Parameters
    ----------
    max_retries:
        Maximum retry attempts after the initial request (default 5).
    base_backoff:
        Base delay in seconds; attempt ``n`` sleeps ``base_backoff * 2**n``.
    timeout:
        Per-request timeout in seconds.
    sleep:
        Blocking sleep taking seconds. Injectable for tests.
    rate_limit:
        Optional :class:`TokenBucket` consulted before each request.

    """

    def __init__(
        self,
        *,
        max_retries: int = 5,
        base_backoff: float = 0.5,
        timeout: float = 10.0,
        sleep: Callable[[float], None] = _time.sleep,
        rate_limit: TokenBucket | None = None,
    ) -> None:
        """Construct the client; see the class docstring for parameters."""
        self._max_retries = max_retries
        self._base_backoff = base_backoff
        self._sleep = sleep
        self._rate_limit = rate_limit
        self._client = httpx.Client(timeout=timeout)

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        """Issue a GET request, retrying throttled/transient responses.

        Parameters
        ----------
        url:
            Absolute request URL.
        params:
            Optional query parameters.
        headers:
            Optional request headers.

        Returns
        -------
        httpx.Response
            The first successful (or final non-retryable) response.

        Raises
        ------
        httpx.HTTPStatusError
            If the retry budget is exhausted on a retryable status, or any
            non-retryable error status is returned.

        """
        attempt = 0
        while True:
            if self._rate_limit is not None:
                self._rate_limit.acquire()
            response = self._client.get(url, params=params, headers=headers)
            retryable = response.status_code in _RETRY_STATUS
            if not retryable or attempt >= self._max_retries:
                response.raise_for_status()
                return response
            self._sleep(self._base_backoff * (2**attempt))
            attempt += 1
