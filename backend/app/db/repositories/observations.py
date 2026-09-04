"""Reads and writes for stored observations.

The spatial query here is the reason this system uses PostGIS rather than two
float columns. ``ST_DWithin`` on a geography column is index-assisted by the
GIST index on ``geom``, so a radius search touches only candidate rows. The
equivalent in Python — fetch everything, compute great-circle distance per row,
discard most of it — would read the whole table for every request.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from geoalchemy2.functions import ST_Distance, ST_DWithin
from sqlalchemy import Select, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.mappers import observation_values, point
from app.db.models import WeatherObservation
from app.domain.weather import WeatherReport

#: Columns refreshed when the same observation is ingested again. The identity
#: columns and ``ingested_at`` are excluded — the first ingestion time is the
#: honest one to keep.
_UPSERT_EXCLUDED = frozenset(
    {"id", "provider_id", "model", "location_key", "observed_at", "ingested_at"}
)


@dataclass(frozen=True, slots=True)
class NearbyObservation:
    """An observation together with how far it sits from the query point."""

    observation: WeatherObservation
    distance_m: float


class ObservationRepository:
    """Persistence operations for :class:`WeatherObservation`."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert(self, report: WeatherReport) -> None:
        """Store one validated observation, updating any existing match.

        Idempotent on (provider, model, place, observed instant): re-requesting
        the same observation refreshes it rather than appending a duplicate.
        """
        values = observation_values(report)
        statement = insert(WeatherObservation).values(**values)
        await self._session.execute(
            statement.on_conflict_do_update(
                constraint="uq_weather_observations_identity",
                set_={
                    column: statement.excluded[column]
                    for column in values
                    if column not in _UPSERT_EXCLUDED
                },
            )
        )

    async def find_in_range(
        self,
        *,
        latitude: float,
        longitude: float,
        start: datetime,
        end: datetime,
        radius_m: float,
        provider_id: str | None = None,
        limit: int = 1000,
    ) -> list[NearbyObservation]:
        """Observations near a point within a time window, closest-in-time first.

        Radius filtering happens in PostGIS via ``ST_DWithin``; the distance
        returned to the caller comes from ``ST_Distance`` on the geography type,
        which is metres on the spheroid.
        """
        origin = point(latitude, longitude)
        distance = ST_Distance(WeatherObservation.geom, origin).label("distance_m")

        statement: Select = (
            select(WeatherObservation, distance)
            .where(
                ST_DWithin(WeatherObservation.geom, origin, radius_m),
                WeatherObservation.observed_at >= start,
                WeatherObservation.observed_at <= end,
            )
            .order_by(WeatherObservation.observed_at.asc(), distance.asc())
            .limit(limit)
        )
        if provider_id is not None:
            statement = statement.where(WeatherObservation.provider_id == provider_id)

        result = await self._session.execute(statement)
        return [
            NearbyObservation(observation=row[0], distance_m=float(row[1]))
            for row in result.all()
        ]

    async def find_nearest(
        self, *, latitude: float, longitude: float, radius_m: float
    ) -> NearbyObservation | None:
        """The closest stored observation to a point, if one is in range."""
        origin = point(latitude, longitude)
        distance = ST_Distance(WeatherObservation.geom, origin).label("distance_m")

        result = await self._session.execute(
            select(WeatherObservation, distance)
            .where(ST_DWithin(WeatherObservation.geom, origin, radius_m))
            .order_by(distance.asc())
            .limit(1)
        )
        row = result.first()
        return (
            None
            if row is None
            else NearbyObservation(observation=row[0], distance_m=float(row[1]))
        )

    async def count(self) -> int:
        """Total stored observations. Used by diagnostics and tests."""
        result = await self._session.execute(
            select(func.count()).select_from(WeatherObservation)
        )
        return int(result.scalar_one())
