"""Application service for weather queries.

Sits between the API layer and the ingestion pipeline and owns the application
logic that is not HTTP and not meteorology:

* resolving a place name to coordinates when the caller gave one
* caching responses per location, horizon and source
* recording validated results to durable storage
* handing validated results to the alert engine for evaluation
* merging geocoded place labels onto the location the provider served
* honouring an explicitly requested provider

Routes call this and do nothing else, which keeps request handling free of
business logic and makes every behaviour here testable without an HTTP client.

The cache and the database answer different questions and are kept apart:

    cache     avoid repeating an upstream request  (per-instance, seconds)
    database  durable history and spatial queries  (shared, indefinite)

Both sit behind this interface. The cache is reached only through
``TTLCache.get_or_load`` and the database only through ``PersistenceService``,
so replacing the per-instance cache with a shared one later changes neither this
service's signature nor its callers.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config.logging import get_logger
from app.domain.forecast import Forecast
from app.domain.location import Coordinates, Location
from app.domain.weather import WeatherReport
from app.ingestion.pipeline import IngestionPipeline
from app.services.cache import TTLCache
from app.services.geocoding import GeocodingService
from app.services.alerts import AlertService
from app.services.persistence import PersistenceService

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class LocationQuery:
    """Either coordinates or a place name — exactly one is resolved to a point.

    Coordinates win when both are supplied: they are unambiguous, and a caller
    that has them has already done the resolution.
    """

    coordinates: Coordinates | None = None
    place: str | None = None


class WeatherService:
    """The application's entry point for meteorological queries."""

    def __init__(
        self,
        *,
        pipeline: IngestionPipeline,
        geocoding: GeocodingService,
        current_cache: TTLCache[WeatherReport],
        forecast_cache: TTLCache[Forecast],
        persistence: PersistenceService | None = None,
        alerts: AlertService | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._geocoding = geocoding
        self._current_cache = current_cache
        self._forecast_cache = forecast_cache
        self._persistence = persistence
        self._alerts = alerts

    async def get_current(
        self, query: LocationQuery, *, provider_id: str | None = None
    ) -> WeatherReport:
        """Current conditions for a resolved location.

        Raises:
            LocationNotFoundError: a place name could not be resolved.
            WeatherDataUnavailableError: no source produced valid data.
        """
        location = await self._resolve(query)
        key = f"current|{location.coordinates.cache_key}|{provider_id or '*'}"

        async def fetch_and_record() -> WeatherReport:
            report = await self._pipeline.current(
                location.coordinates, provider_id=provider_id
            )
            # Reached only for data that cleared the validation gate, so no
            # invalid record can be written — and no invalid record can reach a
            # rule. Alerts are evaluated after persistence so an alert always
            # has the observation behind it on record.
            if self._persistence is not None:
                await self._persistence.record_observation(report)
            if self._alerts is not None:
                await self._alerts.evaluate_observation(report)
            return report

        report, cached = await self._current_cache.get_or_load(key, fetch_and_record)
        return self._finalise(report, location=location, cached=cached)

    async def get_forecast(
        self,
        query: LocationQuery,
        *,
        days: int,
        include_hourly: bool = False,
        include_daily: bool = True,
        provider_id: str | None = None,
    ) -> Forecast:
        """A forecast for a resolved location.

        Raises:
            LocationNotFoundError: a place name could not be resolved.
            ForecastUnavailableError: no source produced a valid forecast.
        """
        location = await self._resolve(query)
        key = (
            f"forecast|{location.coordinates.cache_key}|{days}"
            f"|{int(include_hourly)}{int(include_daily)}|{provider_id or '*'}"
        )

        async def fetch_and_record() -> Forecast:
            forecast = await self._pipeline.forecast(
                location.coordinates,
                days=days,
                include_hourly=include_hourly,
                include_daily=include_daily,
                provider_id=provider_id,
            )
            if self._persistence is not None:
                await self._persistence.record_forecast(forecast)
            if self._alerts is not None:
                await self._alerts.evaluate_forecast(forecast)
            return forecast

        forecast, cached = await self._forecast_cache.get_or_load(key, fetch_and_record)
        return self._finalise(forecast, location=location, cached=cached)

    # --- Internals ----------------------------------------------------------

    async def _resolve(self, query: LocationQuery) -> Location:
        if query.coordinates is not None:
            return Location(coordinates=query.coordinates)
        if query.place:
            return await self._geocoding.resolve(query.place)
        # Guarded at the API boundary; this is the last line of defence.
        raise ValueError("A location query needs either coordinates or a place name.")

    def _finalise(self, record, *, location: Location, cached: bool):
        """Attach place labels and mark whether the response was cached.

        The provider reports the grid point it actually served, which we keep;
        the human-facing labels come from geocoding, which the provider never
        saw. Both matter, so they are merged rather than one overwriting the
        other.
        """
        merged = record.location.model_copy(
            update={
                "name": location.name or record.location.name,
                "country": location.country or record.location.country,
                "country_code": location.country_code or record.location.country_code,
                "admin1": location.admin1 or record.location.admin1,
                "timezone": record.location.timezone or location.timezone,
                "population": location.population or record.location.population,
            }
        )
        return record.model_copy(
            update={
                "location": merged,
                "provenance": record.provenance.model_copy(update={"cached": cached}),
            }
        )
