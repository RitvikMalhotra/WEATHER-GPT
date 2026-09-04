"""Shared HTTP access for upstream meteorological sources.

One place decides how we talk to the outside world: deadlines, retry policy,
which failures are worth retrying, and how transport errors become application
errors. Providers get a thin ``get_json`` and stay focused on their payloads.

The retry policy is intentionally small — bounded attempts with exponential
backoff and jitter, only on failures that are plausibly transient. No circuit
breaker, no bulkheads, no retry library. A weather API that is down stays down;
hammering it does not help, and the pipeline already has provider fallback.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any, Mapping

import httpx

from app.config.logging import get_logger
from app.core.exceptions import WeatherProviderError, WeatherProviderTimeoutError

logger = get_logger(__name__)

#: Status codes worth a second attempt: rate limiting and transient upstream faults.
RETRYABLE_STATUS_CODES = frozenset({429, 502, 503, 504})


def _extract_reason(response: httpx.Response) -> str | None:
    """Pull an explanation out of an error body, if the upstream offers one.

    Most weather APIs describe a rejected request in the body — Open-Meteo uses
    ``{"error": true, "reason": "..."}``. Surfacing it turns an opaque 400 into
    an actionable log line.
    """
    try:
        body = response.json()
    except ValueError:
        return None
    if not isinstance(body, dict):
        return None
    for key in ("reason", "message", "error_description", "detail"):
        value = body.get(key)
        if isinstance(value, str) and value:
            return value[:500]
    return None


def build_http_client(
    *, timeout_seconds: float, user_agent: str, max_connections: int = 20
) -> httpx.AsyncClient:
    """Create the shared client.

    Owned by the application lifespan so connections are pooled across requests
    and closed exactly once at shutdown.
    """
    return httpx.AsyncClient(
        timeout=httpx.Timeout(timeout_seconds),
        headers={"User-Agent": user_agent, "Accept": "application/json"},
        limits=httpx.Limits(
            max_connections=max_connections, max_keepalive_connections=max_connections
        ),
        follow_redirects=True,
    )


class UpstreamHttpClient:
    """Retrying JSON client scoped to a single provider.

    Args:
        client: shared, lifespan-managed connection pool.
        provider_id: used for error attribution and log correlation.
        max_retries: additional attempts after the first, so ``2`` means up to
            three calls in total.
        backoff_seconds: base delay, doubled per attempt and jittered.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        provider_id: str,
        max_retries: int = 2,
        backoff_seconds: float = 0.25,
    ) -> None:
        self._client = client
        self._provider_id = provider_id
        self._max_retries = max(0, max_retries)
        self._backoff_seconds = backoff_seconds

    async def get_json(
        self, url: str, params: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """GET ``url`` and return the decoded JSON object.

        Raises:
            WeatherProviderTimeoutError: every attempt exceeded the deadline.
            WeatherProviderError: the upstream returned an error status, an
                unreadable body, or a non-object payload.
        """
        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            if attempt:
                await self._sleep_before_retry(attempt)

            try:
                response = await self._client.get(url, params=params)
            except httpx.TimeoutException as exc:
                last_error = exc
                self._log_retry(attempt, url, reason="timeout")
                continue
            except httpx.HTTPError as exc:
                last_error = exc
                self._log_retry(attempt, url, reason=type(exc).__name__)
                continue

            if response.status_code in RETRYABLE_STATUS_CODES:
                last_error = httpx.HTTPStatusError(
                    f"HTTP {response.status_code}", request=response.request, response=response
                )
                self._log_retry(attempt, url, reason=f"http_{response.status_code}")
                continue

            return self._decode(response, url)

        # Every attempt failed.
        if isinstance(last_error, httpx.TimeoutException):
            raise WeatherProviderTimeoutError(
                f"{self._provider_id} did not respond within the deadline.",
                provider_id=self._provider_id,
                details={"url": url, "attempts": self._max_retries + 1},
            ) from last_error

        raise WeatherProviderError(
            f"{self._provider_id} is unreachable.",
            provider_id=self._provider_id,
            details={
                "url": url,
                "attempts": self._max_retries + 1,
                "reason": type(last_error).__name__ if last_error else "unknown",
            },
        ) from last_error

    def _decode(self, response: httpx.Response, url: str) -> dict[str, Any]:
        if response.is_error:
            raise WeatherProviderError(
                f"{self._provider_id} returned HTTP {response.status_code}.",
                provider_id=self._provider_id,
                details={
                    "url": url,
                    "status_code": response.status_code,
                    "upstream_reason": _extract_reason(response),
                },
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise WeatherProviderError(
                f"{self._provider_id} returned a body that is not valid JSON.",
                provider_id=self._provider_id,
                details={"url": url},
            ) from exc

        if not isinstance(payload, dict):
            raise WeatherProviderError(
                f"{self._provider_id} returned {type(payload).__name__}, expected an object.",
                provider_id=self._provider_id,
                details={"url": url},
            )
        return payload

    async def _sleep_before_retry(self, attempt: int) -> None:
        delay = self._backoff_seconds * (2 ** (attempt - 1))
        await asyncio.sleep(delay * (0.5 + random.random()))

    def _log_retry(self, attempt: int, url: str, *, reason: str) -> None:
        logger.warning(
            "provider.request_failed",
            extra={
                "provider": self._provider_id,
                "url": url,
                "attempt": attempt + 1,
                "max_attempts": self._max_retries + 1,
                "reason": reason,
            },
        )
