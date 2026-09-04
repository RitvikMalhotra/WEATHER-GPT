"""Composition root.

Every concrete object the application runs on is constructed here, once, from
configuration. Nothing else in the codebase reaches for a global or builds its
own client — modules declare what they need in their constructor and receive it.

That is what keeps the layers honest: the pipeline can be handed a fake
provider, the service a fake pipeline, and a route a fake service, all without
patching module internals. It is also the seam where a future database session
factory or alert dispatcher will be added.

The container owns the HTTP connection pool, so it must be closed. The
application lifespan does that.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import httpx

from app.alerts.engine import AlertEngine
from app.alerts.rules import build_rules
from app.config.logging import get_logger
from app.config.settings import Settings
from app.db.engine import Database
from app.domain.forecast import Forecast
from app.domain.location import Location
from app.domain.weather import WeatherReport
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.validation import WeatherValidator
from app.providers.http import UpstreamHttpClient, build_http_client
from app.providers.open_meteo.client import PROVIDER_ID, OpenMeteoClient
from app.providers.open_meteo.geocoding import OpenMeteoGeocodingClient
from app.providers.open_meteo.provider import OpenMeteoProvider
from app.providers.registry import ProviderRegistry
from app.services.alerts import AlertService
from app.services.cache import TTLCache
from app.services.geocoding import GeocodingService
from app.services.history import HistoryService
from app.services.persistence import PersistenceService
from app.services.weather_service import WeatherService

logger = get_logger(__name__)


@dataclass(slots=True)
class ApplicationContainer:
    """The wired application graph."""

    settings: Settings
    registry: ProviderRegistry
    pipeline: IngestionPipeline
    weather: WeatherService
    geocoding: GeocodingService
    persistence: PersistenceService
    history: HistoryService
    alerts: AlertService
    http_client: httpx.AsyncClient
    database: Database | None = None

    async def aclose(self) -> None:
        """Release the shared connection pool and the database engine."""
        await self.http_client.aclose()
        if self.database is not None:
            await self.database.dispose()


def build_container(
    settings: Settings,
    *,
    http_client: httpx.AsyncClient | None = None,
    database: Database | None = None,
) -> ApplicationContainer:
    """Construct the application graph from configuration.

    Args:
        settings: validated configuration.
        http_client: connection pool to give the providers. Supplying one lets a
            test drive the whole graph through ``httpx.MockTransport`` — every
            layer runs for real, only the network is replaced.
        database: an already-built database, for tests that point the graph at a
            throwaway schema. Otherwise one is created when DATABASE_URL is set.
    """
    http_client = http_client or build_http_client(
        timeout_seconds=settings.PROVIDER_TIMEOUT_SECONDS,
        user_agent=settings.HTTP_USER_AGENT,
    )

    def upstream(provider_id: str) -> UpstreamHttpClient:
        return UpstreamHttpClient(
            http_client,
            provider_id=provider_id,
            max_retries=settings.PROVIDER_MAX_RETRIES,
        )

    registry = ProviderRegistry(default_provider_id=settings.DEFAULT_PROVIDER)
    registry.register(
        OpenMeteoProvider(
            OpenMeteoClient(
                upstream(PROVIDER_ID),
                forecast_url=settings.OPEN_METEO_FORECAST_URL,
            )
        )
    )

    validator = WeatherValidator(
        max_age=timedelta(minutes=settings.MAX_OBSERVATION_AGE_MINUTES),
        max_clock_skew=timedelta(minutes=settings.MAX_CLOCK_SKEW_MINUTES),
    )
    pipeline = IngestionPipeline(registry, validator)

    geocoding = GeocodingService(
        OpenMeteoGeocodingClient(
            upstream("open-meteo-geocoding"),
            search_url=settings.OPEN_METEO_GEOCODING_URL,
        ),
        cache=TTLCache[list[Location]](
            ttl_seconds=settings.GEOCODING_CACHE_TTL_SECONDS
        ),
    )

    if database is None and settings.persistence_active:
        database = Database(
            settings.DATABASE_URL,  # type: ignore[arg-type]  # guarded above
            echo=settings.DATABASE_ECHO,
            pool_size=settings.DATABASE_POOL_SIZE,
            max_overflow=settings.DATABASE_MAX_OVERFLOW,
            pool_timeout=settings.DATABASE_POOL_TIMEOUT_SECONDS,
        )

    persistence = PersistenceService(
        database if settings.PERSISTENCE_ENABLED else None,
        generation_bucket_minutes=settings.FORECAST_GENERATION_BUCKET_MINUTES,
    )
    history = HistoryService(database)

    alerts = AlertService(
        AlertEngine(
            build_rules(settings),
            observation_validity=timedelta(
                minutes=settings.ALERT_OBSERVATION_VALIDITY_MINUTES
            ),
            forecast_lookahead=timedelta(
                hours=settings.ALERT_FORECAST_LOOKAHEAD_HOURS
            ),
        ),
        database,
        enabled=settings.ALERT_EVALUATION_ENABLED,
        max_per_evaluation=settings.ALERT_MAX_PER_EVALUATION,
    )

    weather = WeatherService(
        pipeline=pipeline,
        geocoding=geocoding,
        persistence=persistence,
        alerts=alerts,
        current_cache=TTLCache[WeatherReport](
            ttl_seconds=settings.CURRENT_CACHE_TTL_SECONDS
        ),
        forecast_cache=TTLCache[Forecast](
            ttl_seconds=settings.FORECAST_CACHE_TTL_SECONDS
        ),
    )

    logger.info(
        "container.built",
        extra={
            "providers": [p.metadata.provider_id for p in registry.all()],
            "default_provider": settings.DEFAULT_PROVIDER,
            "persistence": persistence.enabled,
            "alert_rules": len(alerts.engine.rules),
            "alert_evaluation": alerts.enabled,
        },
    )
    return ApplicationContainer(
        settings=settings,
        registry=registry,
        pipeline=pipeline,
        weather=weather,
        geocoding=geocoding,
        persistence=persistence,
        history=history,
        alerts=alerts,
        http_client=http_client,
        database=database,
    )
