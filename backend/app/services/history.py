"""Reading meteorological history.

The counterpart to :mod:`app.services.persistence`, and the one place where the
database is preferred over a provider: a record we validated on the way in is
better evidence than the same hour fetched again later.

Preferred, not required. A deployment that was not running yesterday stored
nothing about yesterday, and reporting that as "0 observations" answers a
question about our storage rather than the one that was asked. So a window the
database cannot cover falls through to a provider archive, read through the
same interface and stamped with the same provenance. A database that is
configured but unreachable takes the same path, for the same reason. Only when
there is nothing left behind it does the failure surface, as
:class:`DatabaseUnavailableError`, stripped of driver detail by the engine
layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import TYPE_CHECKING

from app.config.logging import get_logger
from app.core.exceptions import DatabaseUnavailableError
from app.db.engine import Database, translate_database_errors
from app.db.mappers import row_provenance, row_to_current_weather
from app.db.repositories import ObservationRepository
from app.domain.provenance import DataProvenance
from app.domain.weather import CurrentWeather

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.services.archive import ArchiveReader

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
    """A resolved request for past observations."""

    latitude: float
    longitude: float
    start: datetime
    end: datetime
    radius_m: float
    provider_id: str | None = None
    limit: int = 1000
    #: Set when the caller asked in calendar dates rather than in instants.
    #: The archive resolves these against the *location's* day, because a UTC
    #: midnight is the wrong midnight for the person asking.
    calendar_start: date | None = None
    calendar_end: date | None = None
    #: Local hours of day to keep, e.g. "yesterday evening" -> (17, 21).
    local_hours: tuple[int, int] | None = None


class HistoryService:
    """Answers questions about the past.

    Two sources, in order. What this deployment recorded is preferred, because
    it is what we can vouch for having validated on the way in. What it never
    recorded — which, on a fresh install, is everything — comes from a
    provider's archive through the same normalisation and the same provenance.

    The order matters and the fallback matters: "stored observations: 0
    returned" is a true statement about our storage and a useless answer to a
    question about yesterday's rain.
    """

    def __init__(self, database: Database | None, archive: "ArchiveReader | None" = None) -> None:
        self._database = database
        self._archive = archive

    @property
    def enabled(self) -> bool:
        """True when *some* source can answer a question about the past."""
        return self._database is not None or (
            self._archive is not None and self._archive.enabled
        )

    async def observations(self, query: HistoryQuery) -> list[HistoricalRecord]:
        """Past observations matching a spatial and temporal window.

        Raises:
            DatabaseUnavailableError: nothing can answer — no database and no
                archive-capable provider.
        """
        try:
            stored = await self._stored(query)
        except DatabaseUnavailableError:
            # A database that is configured but unreachable holds nothing we
            # can read right now, which for this question is the same position
            # as having recorded nothing at all — and that case already falls
            # through to the archive. Failing here instead would make a
            # deployment *with* a database less able to answer than one
            # without. With no archive behind it the failure is still the
            # answer, so it is re-raised.
            if self._archive is None or not self._archive.enabled:
                raise
            logger.warning("history.database_unavailable", exc_info=True)
            stored = []

        if stored or self._archive is None:
            if not stored and self._database is None:
                raise DatabaseUnavailableError(
                    "Historical weather requires a configured database or an "
                    "archive-capable provider.",
                    details={"reason": "no_history_source"},
                )
            return stored

        archived = await self._archive.observations(query)
        if archived:
            logger.info(
                "history.served_from_archive",
                extra={"location": f"{query.latitude:.4f},{query.longitude:.4f}"},
            )
        return archived

    async def _stored(self, query: HistoryQuery) -> list[HistoricalRecord]:
        """What this deployment recorded, or nothing when it recorded nothing."""
        if self._database is None:
            return []

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
