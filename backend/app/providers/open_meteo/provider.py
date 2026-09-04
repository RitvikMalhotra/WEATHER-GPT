"""The Open-Meteo weather provider.

Open-Meteo is the first source wired in because it is keyless, openly licensed
and aggregates several national models (including GFS and ICON) behind one
schema — which makes it a good reference implementation for the interface that
IMD, a direct GFS feed or a local WRF run will later implement.

The class itself is thin by design: fetch through the client, stamp provenance,
delegate the mapping to the normaliser. All the interesting Open-Meteo-specific
knowledge lives in those two modules.
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.domain.forecast import Forecast
from app.domain.location import Coordinates
from app.domain.weather import WeatherReport
from app.providers.base import ProviderCapability, ProviderMetadata, WeatherProvider
from app.providers.open_meteo.client import PROVIDER_ID, OpenMeteoClient
from app.providers.open_meteo.normalizer import (
    build_provenance,
    normalize_current,
    normalize_forecast,
)

METADATA = ProviderMetadata(
    provider_id=PROVIDER_ID,
    name="Open-Meteo",
    capabilities=frozenset(
        {
            ProviderCapability.CURRENT,
            ProviderCapability.HOURLY_FORECAST,
            ProviderCapability.DAILY_FORECAST,
        }
    ),
    source_url="https://open-meteo.com",
    license="CC-BY-4.0",
    attribution="Weather data by Open-Meteo.com",
    model="best_match",
    priority=10,
    max_forecast_days=16,
    notes=(
        "Aggregates national models (ICON, GFS, ARPEGE and others) and selects "
        "the best available for each location.",
    ),
)


class OpenMeteoProvider(WeatherProvider):
    """Current conditions and forecasts from Open-Meteo."""

    def __init__(self, client: OpenMeteoClient) -> None:
        self._client = client

    @property
    def metadata(self) -> ProviderMetadata:
        return METADATA

    async def fetch_current(self, coordinates: Coordinates) -> WeatherReport:
        payload = await self._client.fetch(
            latitude=coordinates.latitude,
            longitude=coordinates.longitude,
            current=True,
        )
        return normalize_current(payload, provenance=self._provenance())

    async def fetch_forecast(
        self,
        coordinates: Coordinates,
        *,
        days: int,
        include_hourly: bool,
        include_daily: bool,
    ) -> Forecast:
        payload = await self._client.fetch(
            latitude=coordinates.latitude,
            longitude=coordinates.longitude,
            hourly=include_hourly,
            daily=include_daily,
            forecast_days=min(days, METADATA.max_forecast_days),
        )
        return normalize_forecast(payload, provenance=self._provenance())

    def _provenance(self):
        return build_provenance(
            fetched_at=datetime.now(timezone.utc),
            source_url=self._client.forecast_url,
            metadata=METADATA,
        )
