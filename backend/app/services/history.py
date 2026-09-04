"""Reading stored meteorological history.

The counterpart to :mod:`app.services.persistence`, and the one place where the
database is the source of truth rather than a side effect — because the past is
the one thing a provider cannot re-fetch on demand.

Unlike the write path, a failure here is not recoverable: there is no provider
to fall back to. So database errors surface as
:class:`DatabaseUnavailableError`, already stripped of driver detail by the
engine layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from app.config.logging import get_logger
from app.core.exceptions import DatabaseUnavailableError
from app.db.engine import Database, translate_database_errors
from app.db.mappers import row_provenance, row_to_current_weather
from app.db.repositories import ObservationRepository
from app.domain.provenance import DataProvenance
from app.domain.weather import CurrentWeather

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class HistoricalRecord:
    """One stored observation, with its provenance and distance from the query."""

    weather: CurrentWeather
    provenance: DataProvenance
    latitude: float
    longitude: float
    distance_m: float


@dataclass(frozen=True, slots=True)
class HistoryQuery:
    """A resolved request for stored observations."""

    latitude: float
    longitude: float
    start: datetime
    end: datetime
    radius_m: float
    provider_id: str | None = None
    limit: int = 1000


class HistoryService:
    """Answers questions about stored observations."""

    def __init__(self, database: Database | None) -> None:
        self._database = database

    @property
    def enabled(self) -> bool:
        return self._database is not None

    async def observations(self, query: HistoryQuery) -> list[HistoricalRecord]:
        """Stored observations matching a spatial and temporal window.

        Raises:
            DatabaseUnavailableError: persistence is unconfigured or unreachable.
        """
        if self._database is None:
            raise DatabaseUnavailableError(
                "Historical weather requires a configured database.",
                details={"reason": "persistence_disabled"},
            )

        async with translate_database_errors("history.observations"):
            async with self._database.session() as session:
                found = await ObservationRepository(session).find_in_range(
                    latitude=query.latitude,
                    longitude=query.longitude,
                    start=query.start,
                    end=query.end,
                    radius_m=query.radius_m,
                    provider_id=query.provider_id,
                    limit=query.limit,
                )

        logger.info(
            "history.observations",
            extra={
                "location": f"{query.latitude:.4f},{query.longitude:.4f}",
                "radius_m": query.radius_m,
                "results": len(found),
            },
        )
        return [
            HistoricalRecord(
                weather=row_to_current_weather(entry.observation),
                provenance=row_provenance(entry.observation),
                latitude=entry.observation.latitude,
                longitude=entry.observation.longitude,
                distance_m=entry.distance_m,
            )
            for entry in found
        ]
