"""The weather provider contract.

A provider is the only part of the system allowed to know how a specific
upstream source speaks. It performs two steps and nothing else:

    fetch   talk to the upstream API (vendor-specific I/O)
    normalise  map the response onto the canonical domain model

It does *not* validate — that is the ingestion pipeline's job, and it must be
applied identically to every source. It does not cache, geocode or decide which
provider to use. Keeping providers this narrow is what makes adding IMD, GFS or
a WRF run a self-contained change.

Adding a provider:

    1. Implement :class:`WeatherProvider` for the new source.
    2. Declare its :class:`ProviderMetadata`, including capabilities and priority.
    3. Register it at startup.

No route, service or model changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from app.domain.forecast import Forecast
from app.domain.location import Coordinates
from app.domain.weather import WeatherReport


class ProviderCapability(str, Enum):
    """What a source can answer.

    The pipeline selects providers by capability, so a radar-only or
    historical-only source can be registered without pretending to serve
    current conditions.
    """

    CURRENT = "current"
    HOURLY_FORECAST = "hourly_forecast"
    DAILY_FORECAST = "daily_forecast"
    HISTORICAL = "historical"
    ALERTS = "alerts"
    RADAR = "radar"


@dataclass(frozen=True, slots=True)
class ProviderMetadata:
    """Static description of a source, used for routing and attribution."""

    provider_id: str
    name: str
    capabilities: frozenset[ProviderCapability]
    source_url: str | None = None
    license: str | None = None
    attribution: str | None = None
    model: str | None = None
    #: Lower sorts first in the fallback chain.
    priority: int = 100
    #: Hard cap on forecast horizon, in days.
    max_forecast_days: int = 16
    #: Extra, provider-specific notes surfaced in the providers endpoint.
    notes: tuple[str, ...] = field(default_factory=tuple)
    #: Area this source covers, as (south, west, north, east) in degrees.
    #: ``None`` means global. A source that covers a region is preferred inside
    #: it and excluded outside it, which is how a national service becomes the
    #: source for its own country without anyone choosing it by hand.
    coverage: tuple[float, float, float, float] | None = None

    def supports(self, capability: ProviderCapability) -> bool:
        return capability in self.capabilities

    def covers(self, latitude: float, longitude: float) -> bool:
        """Whether this source serves the point. Global sources serve every point."""
        if self.coverage is None:
            return True
        south, west, north, east = self.coverage
        return south <= latitude <= north and west <= longitude <= east


class WeatherProvider(ABC):
    """Abstract meteorological data source."""

    @property
    @abstractmethod
    def metadata(self) -> ProviderMetadata:
        """Static description of this source."""

    @abstractmethod
    async def fetch_current(self, coordinates: Coordinates) -> WeatherReport:
        """Return current conditions, normalised to the canonical model.

        Raises:
            WeatherProviderError: the upstream call failed or returned data
                that could not be normalised.
        """

    @abstractmethod
    async def fetch_forecast(
        self,
        coordinates: Coordinates,
        *,
        days: int,
        include_hourly: bool,
        include_daily: bool,
    ) -> Forecast:
        """Return a forecast, normalised to the canonical model.

        Raises:
            WeatherProviderError: the upstream call failed or returned data
                that could not be normalised.
        """

    async def fetch_archive(self, coordinates: Coordinates, *, past_days: int) -> Forecast:
        """Return recent past observations, normalised to the canonical model.

        Optional: a source that keeps no history declares no ``HISTORICAL``
        capability and never receives this call. It is deliberately not
        abstract, so adding archive support to one provider does not break
        every other implementation of this interface.

        The return type is :class:`Forecast` because the shape is identical —
        a location, a provenance and an hourly series. Only the direction in
        time differs, and the caller already knows which way it asked.

        Raises:
            NotImplementedError: this source keeps no archive.
            WeatherProviderError: the upstream call failed or returned data
                that could not be normalised.
        """
        raise NotImplementedError(
            f"{self.metadata.provider_id} does not serve historical data."
        )

    def __repr__(self) -> str:  # pragma: no cover - diagnostic helper
        return f"<{type(self).__name__} id={self.metadata.provider_id!r}>"
