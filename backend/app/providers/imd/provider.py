"""The India Meteorological Department provider.

IMD is station-based: it publishes observations and city forecasts for named
stations, not for arbitrary points. Users ask about localities ("Miyapur"), so
the provider maps a resolved coordinate to the nearest IMD station from IMD's
own published station catalogue — and refuses when the nearest station is too
far to describe the requested place.

Refusing is the important half. Returning Nagpur's observation for a village
600 km away, labelled IMD, is worse than falling back to a global model that
actually covers the point: the pipeline's next provider answers instead, with
its own honest provenance.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from math import asin, cos, radians, sin, sqrt

from app.config.logging import get_logger
from app.core.exceptions import WeatherProviderError
from app.domain.forecast import Forecast
from app.domain.location import Coordinates, Location
from app.domain.provenance import DataProvenance
from app.domain.weather import WeatherReport
from app.providers.base import ProviderCapability, ProviderMetadata, WeatherProvider
from app.providers.imd.client import PROVIDER_ID, ImdClient
from app.providers.imd.mapper import (
    ImdMappingError,
    map_city_forecast,
    map_current,
    station_location,
)

logger = get_logger(__name__)

IMD_PROVIDER_ID = PROVIDER_ID

#: India's bounding box, including island territories. Used to keep IMD out of
#: the running for points it does not cover, before any network call.
INDIA_BOUNDS = (6.0, 68.0, 37.5, 97.5)  # south, west, north, east

METADATA = ProviderMetadata(
    provider_id=IMD_PROVIDER_ID,
    name="India Meteorological Department",
    capabilities=frozenset(
        {ProviderCapability.CURRENT, ProviderCapability.DAILY_FORECAST}
    ),
    source_url="https://mausam.imd.gov.in",
    license="© India Meteorological Department",
    attribution="Data © India Meteorological Department (IMD)",
    model="imd_station",
    # Priority is not what promotes IMD in India; coverage is. This orders it
    # among other regional sources should more be added.
    priority=5,
    max_forecast_days=7,
    coverage=INDIA_BOUNDS,
    notes=(
        "India's national meteorological service and the official source for "
        "Indian weather.",
        "Station-based: values come from the nearest IMD station, and the "
        "provider declines when no station is close enough.",
        "Requires an API key issued by IMD, with IP whitelisting.",
    ),
)


def haversine_km(first: Coordinates, second: Coordinates) -> float:
    """Great-circle distance in kilometres."""
    lat1, lon1 = radians(first.latitude), radians(first.longitude)
    lat2, lon2 = radians(second.latitude), radians(second.longitude)
    a = (
        sin((lat2 - lat1) / 2) ** 2
        + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * 6371.0088 * asin(sqrt(a))


class ImdProvider(WeatherProvider):
    """Observations and city forecasts from IMD stations."""

    def __init__(
        self,
        client: ImdClient,
        *,
        max_station_distance_km: float = 50.0,
        catalogue_ttl: timedelta = timedelta(hours=12),
    ) -> None:
        self._client = client
        self._max_station_distance_km = max_station_distance_km
        self._catalogue_ttl = catalogue_ttl
        self._stations: list[tuple[str, Location]] = []
        self._loaded_at: datetime | None = None

    @property
    def metadata(self) -> ProviderMetadata:
        return METADATA

    async def fetch_current(self, coordinates: Coordinates) -> WeatherReport:
        station_id, location, distance = await self._nearest_station(coordinates)
        record = await self._client.current_weather(station_id)
        try:
            return map_current(
                record,
                location=location,
                provenance=self._provenance(station_id, distance),
            )
        except ImdMappingError as exc:
            raise WeatherProviderError(
                str(exc), provider_id=IMD_PROVIDER_ID, details={"station": station_id}
            ) from exc

    async def fetch_forecast(
        self,
        coordinates: Coordinates,
        *,
        days: int,
        include_hourly: bool,
        include_daily: bool,
    ) -> Forecast:
        if include_hourly:
            # The city-forecast product is daily. Saying so lets the pipeline
            # move on rather than returning a daily series to an hourly request.
            raise WeatherProviderError(
                "IMD's city forecast product does not provide hourly values.",
                provider_id=IMD_PROVIDER_ID,
            )
        station_id, location, distance = await self._nearest_station(coordinates)
        record = await self._client.city_forecast(station_id)
        try:
            return map_city_forecast(
                record,
                location=location,
                provenance=self._provenance(station_id, distance),
                days=min(days, METADATA.max_forecast_days),
            )
        except ImdMappingError as exc:
            raise WeatherProviderError(
                str(exc), provider_id=IMD_PROVIDER_ID, details={"station": station_id}
            ) from exc

    # --- Station mapping ----------------------------------------------------

    async def _nearest_station(
        self, coordinates: Coordinates
    ) -> tuple[str, Location, float]:
        """The closest IMD station, or a refusal if none is close enough."""
        stations = await self._catalogue()
        if not stations:
            raise WeatherProviderError(
                "IMD published no usable station catalogue.",
                provider_id=IMD_PROVIDER_ID,
            )

        best_id, best_location, best_distance = None, None, float("inf")
        for station_id, location in stations:
            distance = haversine_km(coordinates, location.coordinates)
            if distance < best_distance:
                best_id, best_location, best_distance = station_id, location, distance

        if best_id is None or best_distance > self._max_station_distance_km:
            # No station describes this place. The pipeline falls through to a
            # provider that covers the point, under its own name.
            raise WeatherProviderError(
                "No IMD station is close enough to describe this location.",
                provider_id=IMD_PROVIDER_ID,
                details={
                    "nearest_km": round(best_distance, 1)
                    if best_distance != float("inf")
                    else None,
                    "limit_km": self._max_station_distance_km,
                },
            )
        assert best_location is not None
        return best_id, best_location, best_distance

    async def _catalogue(self) -> list[tuple[str, Location]]:
        now = datetime.now(timezone.utc)
        if self._stations and self._loaded_at and now - self._loaded_at < self._catalogue_ttl:
            return self._stations

        records = await self._client.station_catalogue()
        stations: list[tuple[str, Location]] = []
        for record in records:
            identifier = record.get("Station_Code") or record.get("Station_Id") or record.get("id")
            location = station_location(record)
            if identifier and location is not None:
                stations.append((str(identifier), location))
        self._stations = stations
        self._loaded_at = now
        logger.info("imd.catalogue_loaded", extra={"stations": len(stations)})
        return stations

    def _provenance(self, station_id: str, distance_km: float) -> DataProvenance:
        return DataProvenance(
            provider_id=METADATA.provider_id,
            provider_name=METADATA.name,
            # The station is the product: which one answered is provenance.
            model=f"{METADATA.model}:{station_id}",
            fetched_at=datetime.now(timezone.utc),
            source_url=METADATA.source_url,
            license=METADATA.license,
            attribution=(
                f"{METADATA.attribution} — station {station_id}, "
                f"{distance_km:.1f} km from the requested point"
            ),
        )
