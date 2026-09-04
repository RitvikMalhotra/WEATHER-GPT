"""Reads and writes for stored forecast points."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import WeatherForecast

#: Identity columns and first-seen bookkeeping are never overwritten.
_UPSERT_EXCLUDED = frozenset(
    {
        "id",
        "provider_id",
        "model",
        "location_key",
        "resolution",
        "forecast_created_at",
        "forecast_for",
        "ingested_at",
    }
)


class ForecastRepository:
    """Persistence operations for :class:`WeatherForecast`."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert_many(self, rows: Sequence[dict[str, Any]]) -> int:
        """Store a batch of forecast points in one round trip.

        A forecast is tens to hundreds of points; inserting them individually
        would multiply latency by the horizon. Conflicts update in place, so
        re-requesting the same forecast within a generation window refreshes the
        values instead of duplicating them.

        Returns:
            The number of rows submitted.
        """
        if not rows:
            return 0

        # Every row in a batch shares a shape, so one statement covers them all.
        statement = insert(WeatherForecast).values(list(rows))
        updatable = [column for column in rows[0] if column not in _UPSERT_EXCLUDED]
        await self._session.execute(
            statement.on_conflict_do_update(
                constraint="uq_weather_forecasts_identity",
                set_={column: statement.excluded[column] for column in updatable},
            )
        )
        return len(rows)

    async def find_for_location(
        self,
        *,
        location_key: str,
        start: datetime,
        end: datetime,
        resolution: str | None = None,
        limit: int = 2000,
    ) -> list[WeatherForecast]:
        """Stored predictions for a place over a target window.

        Ordered newest-generation-first so the most recent prediction for each
        target time comes first.
        """
        statement = (
            select(WeatherForecast)
            .where(
                WeatherForecast.location_key == location_key,
                WeatherForecast.forecast_for >= start,
                WeatherForecast.forecast_for <= end,
            )
            .order_by(
                WeatherForecast.forecast_for.asc(),
                WeatherForecast.forecast_created_at.desc(),
            )
            .limit(limit)
        )
        if resolution is not None:
            statement = statement.where(WeatherForecast.resolution == resolution)

        result = await self._session.execute(statement)
        return list(result.scalars().all())

    async def count(self) -> int:
        """Total stored forecast points. Used by diagnostics and tests."""
        result = await self._session.execute(
            select(func.count()).select_from(WeatherForecast)
        )
        return int(result.scalar_one())
