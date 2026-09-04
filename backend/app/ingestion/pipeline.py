"""The ingestion pipeline.

Implements the spine of the architecture for a single request:

    sources -> fetch -> normalise -> validate -> canonical model

Fetching and normalisation are delegated to the provider (they are inherently
source-specific). What happens here is the part that must be identical for every
source: choosing which sources to try, applying the validation gate, falling
back when one fails, and recording why.

The rule the pipeline exists to enforce: **data that fails validation is never
served.** A source that returns implausible values is treated exactly like a
source that returned an error — we move to the next one. If every source is
exhausted, the request fails loudly rather than returning a number nobody
checked.

Provider fallback is the only resilience strategy here; retries within a single
source belong to the HTTP layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Sequence, TypeVar

from app.config.logging import get_logger
from app.core.exceptions import (
    ForecastUnavailableError,
    WeatherDataUnavailableError,
    WeatherGPTError,
    WeatherProviderError,
)
from app.domain.forecast import Forecast
from app.domain.location import Coordinates
from app.domain.weather import WeatherReport
from app.ingestion.validation import ValidationResult, WeatherValidator
from app.providers.base import ProviderCapability, WeatherProvider
from app.providers.registry import ProviderRegistry

logger = get_logger(__name__)

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class _Attempt:
    """Why one source did not produce usable data."""

    provider_id: str
    reason: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {"provider": self.provider_id, "reason": self.reason, "detail": self.detail}


class IngestionPipeline:
    """Fetches, validates and returns canonical meteorological records."""

    def __init__(self, registry: ProviderRegistry, validator: WeatherValidator) -> None:
        self._registry = registry
        self._validator = validator

    async def current(
        self, coordinates: Coordinates, *, provider_id: str | None = None
    ) -> WeatherReport:
        """Current conditions for a point, from the first source that passes.

        Raises:
            ProviderNotFoundError: an explicitly requested provider is unknown.
            WeatherDataUnavailableError: no source produced valid data.
        """
        return await self._run(
            capability=ProviderCapability.CURRENT,
            provider_id=provider_id,
            operation="current",
            fetch=lambda provider: provider.fetch_current(coordinates),
            validate=self._validator.validate_current,
            failure=WeatherDataUnavailableError,
            coordinates=coordinates,
        )

    async def forecast(
        self,
        coordinates: Coordinates,
        *,
        days: int,
        include_hourly: bool = False,
        include_daily: bool = True,
        provider_id: str | None = None,
    ) -> Forecast:
        """A forecast for a point, from the first source that passes.

        Raises:
            ProviderNotFoundError: an explicitly requested provider is unknown.
            ForecastUnavailableError: no source produced a valid forecast.
        """
        capability = (
            ProviderCapability.HOURLY_FORECAST
            if include_hourly
            else ProviderCapability.DAILY_FORECAST
        )
        return await self._run(
            capability=capability,
            provider_id=provider_id,
            operation="forecast",
            fetch=lambda provider: provider.fetch_forecast(
                coordinates,
                days=days,
                include_hourly=include_hourly,
                include_daily=include_daily,
            ),
            validate=self._validator.validate_forecast,
            failure=ForecastUnavailableError,
            coordinates=coordinates,
        )

    # --- Internals ----------------------------------------------------------

    async def _run(
        self,
        *,
        capability: ProviderCapability,
        provider_id: str | None,
        operation: str,
        fetch: Callable[[WeatherProvider], Awaitable[T]],
        validate: Callable[[T], ValidationResult],
        failure: type[WeatherGPTError],
        coordinates: Coordinates,
    ) -> T:
        candidates = self._candidates(capability, provider_id, coordinates)
        if not candidates:
            raise failure(
                f"No registered provider can serve {capability.value}.",
                details={"capability": capability.value},
            )

        attempts: list[_Attempt] = []

        for provider in candidates:
            source = provider.metadata.provider_id
            try:
                record = await fetch(provider)
            except WeatherProviderError as exc:
                attempts.append(_Attempt(source, exc.code, exc.message))
                logger.warning(
                    "ingestion.provider_failed",
                    extra={
                        "operation": operation,
                        "provider": source,
                        "error_code": exc.code,
                        "location": coordinates.cache_key,
                    },
                )
                continue

            result = validate(record)
            if not result.is_valid:
                attempts.append(
                    _Attempt(
                        source,
                        "WEATHER_DATA_VALIDATION_FAILED",
                        "; ".join(issue.message for issue in result.errors[:3]),
                    )
                )
                logger.error(
                    "ingestion.validation_rejected",
                    extra={
                        "operation": operation,
                        "provider": source,
                        "location": coordinates.cache_key,
                        "issues": [issue.as_dict() for issue in result.errors],
                    },
                )
                continue

            if result.warnings:
                logger.warning(
                    "ingestion.validation_warnings",
                    extra={
                        "operation": operation,
                        "provider": source,
                        "location": coordinates.cache_key,
                        "issues": [issue.as_dict() for issue in result.warnings],
                    },
                )

            logger.info(
                "ingestion.completed",
                extra={
                    "operation": operation,
                    "provider": source,
                    "location": coordinates.cache_key,
                    "fallbacks_used": len(attempts),
                },
            )
            return record

        # Every candidate failed to fetch or failed validation.
        logger.error(
            "ingestion.exhausted",
            extra={
                "operation": operation,
                "location": coordinates.cache_key,
                "attempts": [attempt.as_dict() for attempt in attempts],
            },
        )
        raise failure(
            "No weather provider returned usable data for this location.",
            details={"attempts": [attempt.as_dict() for attempt in attempts]},
        )

    def _candidates(
        self,
        capability: ProviderCapability,
        provider_id: str | None,
        coordinates: Coordinates | None = None,
    ) -> Sequence[WeatherProvider]:
        """Sources to try, in order.

        An explicit provider request is honoured exactly: one candidate, no
        fallback. Silently substituting a different source would make the
        response's provenance a lie.
        """
        if provider_id is not None:
            return [self._registry.get(provider_id)]
        if coordinates is None:
            return self._registry.for_capability(capability)
        return self._registry.for_capability(
            capability,
            latitude=coordinates.latitude,
            longitude=coordinates.longitude,
        )
