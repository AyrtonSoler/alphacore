"""RED tests for common.http — HTTP client with backoff + rate limiting.

Acceptance: a 429 response triggers exponential backoff (mocked via respx).
"""

from __future__ import annotations

import httpx
import pytest
import respx

from common.http import Client, TokenBucket


@respx.mock
def test_429_triggers_exponential_backoff_then_succeeds() -> None:
    """Two 429s then a 200: the client retries, sleeping with exponential delays."""
    route = respx.get("https://api.test/data").mock(
        side_effect=[
            httpx.Response(429),
            httpx.Response(429),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    delays: list[float] = []
    client = Client(max_retries=5, base_backoff=0.5, sleep=delays.append)

    resp = client.get("https://api.test/data")

    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    assert route.call_count == 3
    # base * 2**0, base * 2**1 — deterministic, no jitter.
    assert delays == [0.5, 1.0]


@respx.mock
def test_gives_up_after_max_retries() -> None:
    """Persistent 429s exhaust the retry budget and raise on the final response."""
    respx.get("https://api.test/down").mock(return_value=httpx.Response(429))
    delays: list[float] = []
    client = Client(max_retries=3, base_backoff=0.1, sleep=delays.append)

    with pytest.raises(httpx.HTTPStatusError):
        client.get("https://api.test/down")

    # 1 initial attempt + 3 retries = 4 calls; backoff slept before each retry.
    assert len(delays) == 3


@respx.mock
def test_success_does_not_sleep() -> None:
    """A first-try 200 returns immediately without any backoff."""
    respx.get("https://api.test/ok").mock(return_value=httpx.Response(200))
    delays: list[float] = []
    client = Client(max_retries=5, base_backoff=0.5, sleep=delays.append)

    resp = client.get("https://api.test/ok")

    assert resp.status_code == 200
    assert delays == []


def test_token_bucket_serves_burst_then_throttles() -> None:
    """A full bucket serves its burst without sleeping, then waits to refill."""
    clock = {"t": 0.0}
    waits: list[float] = []

    def now() -> float:
        return clock["t"]

    def sleep(seconds: float) -> None:
        clock["t"] += seconds
        waits.append(seconds)

    bucket = TokenBucket(rate=2.0, capacity=2.0, now=now, sleep=sleep)

    assert bucket.acquire() == 0.0  # 2 -> 1, immediate
    assert bucket.acquire() == 0.0  # 1 -> 0, immediate
    assert bucket.acquire() == 0.5  # empty: wait deficit (1 token) / rate (2/s)
    assert waits == [0.5]
