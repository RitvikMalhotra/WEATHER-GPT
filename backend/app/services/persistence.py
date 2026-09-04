"""Writing validated meteorological data to durable storage.

Where this sits matters. The service is invoked by :class:`WeatherService`
*inside the cache loader*, which means it only ever receives a value that has
already come back through the ingestion pipeline:

    provider -> normalise -> validate -> [here] -> respond

The pipeline returns only records that passed the validation gate, so "invalid
data is never persisted" is a structural property of where this code is called
from, not a check it performs. There is no path from a provider to this table
that bypasses validation.

Two further rules:

* **A write failure never breaks a read.** Current conditions and forecasts are
  answered from provider data that is already in hand and already valid. If the
  database is unreachable, the caller still gets their weather; the failure is
  logged and the response is unaffected. Persistence is a side effect of serving
  a request, not a precondition for it.
* **Cache hits do not rewrite.** Because persistence runs in the cache loader, a
  repeated request inside the TTL touches neither the provider nor the database.

Responsibilities are split with the cache, and stay split:

    cache     avoid repeating an upstream request  (per-instance, seconds)
    database  durable history and spatial queries  (shared, indefinite)
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.config.logging import get_logger
from app.db.engine import Database
from app.db.mappers import daily_forecast_values, hourly_forecast_values
from app.db.repositories import ForecastRepository, ObservationRepository
from app.domain.forecast import Forecast
from app.domain.weather import WeatherReport

logger = get_logger(__name__)


def generation_bucket(moment: datetime, *, minutes: int) -> datetime:
    """Round an instant down to a generation window.

    Providers rarely publish which model run produced a forecast, so we cannot
    record the true generation time. Bucketing the fetch time gives a stable
    stand-in: every request inside the same window updates one set of rows,
    while a later window appends a new generation. That keeps repeated API calls
    from multiplying rows, without collapsing the forecast history that makes
    skill analysis possible.
    """
    if minutes <= 0:
        return moment
    span = minutes * 60
    epoch_seconds = int(moment.timestamp())
    return datetime.fromtimestamp(
        epoch_seconds - (epoch_seconds % span), tz=timezone.utc
    )


class PersistenceService:
    """Records validated observations and forecasts."""

    def __init__(
        self,
        database: Database | None,
        *,
        generation_bucket_minutes: int = 60,
    ) -> None:
        self._database = database
        self._bucket_minutes = generation_bucket_minutes

    @property
    def enabled(self) -> bool:
        """False when the deployment runs without a configured database."""
        return self._database is not None

    async def record_observation(self, report: WeatherReport) -> bool:
        """Store one validated observation. Never raises.

        Returns:
            True when the row was written.
        """
        if self._database is None:
            return False
        try:
            async with self._database.session() as session:
                await ObservationRepository(session).upsert(report)
        except Exception:  # noqa: BLE001 - a write must not break the response
            logger.exception(
                "persistence.observation_failed",
                extra={
                    "provider": report.provenance.provider_id,
                    "location": report.location.coordinates.cache_key,
                },
            )
            return False

        logger.info(
            "persistence.observation_recorded",
            extra={
                "provider": report.provenance.provider_id,
                "location": report.location.coordinates.cache_key,
                "observed_at": report.current.observed_at.isoformat(),
            },
        )
        return True

    async def record_forecast(self, forecast: Forecast) -> int:
        """Store every point of a validated forecast. Never raises.

        Returns:
            The number of points written.
        """
        if self._database is None:
            return 0

        created_at = generation_bucket(
            forecast.provenance.fetched_at, minutes=self._bucket_minutes
        )
        rows = [
            hourly_forecast_values(forecast, entry, created_at=created_at)
            for entry in forecast.hourly
        ] + [
            daily_forecast_values(forecast, entry, created_at=created_at)
            for entry in forecast.daily
        ]
        if not rows:
            return 0

        try:
            async with self._database.session() as session:
                repository = ForecastRepository(session)
                # Hourly and daily rows differ in which columns they populate,
                # so they go in as two homogeneous batches.
                written = 0
                for batch in _group_by_resolution(rows):
                    written += await repository.upsert_many(batch)
        except Exception:  # noqa: BLE001 - a write must not break the response
            logger.exception(
                "persistence.forecast_failed",
                extra={
                    "provider": forecast.provenance.provider_id,
                    "location": forecast.location.coordinates.cache_key,
                    "points": len(rows),
                },
            )
            return 0

        logger.info(
            "persistence.forecast_recorded",
            extra={
                "provider": forecast.provenance.provider_id,
                "location": forecast.location.coordinates.cache_key,
                "points": written,
                "forecast_created_at": created_at.isoformat(),
            },
        )
        return written


def _group_by_resolution(rows: list[dict]) -> list[list[dict]]:
    """Split mixed rows into batches that share a column set."""
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["resolution"], []).append(row)
    return list(grouped.values())


__all__ = ["PersistenceService", "generation_bucket"]
