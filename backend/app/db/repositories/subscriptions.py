"""Persistence for the places a person asked WeatherGPT to watch.

Deliberately small. A subscription is a row saying *where to look*; everything
about what was found lives in :mod:`app.db.repositories.alerts`, written by the
deterministic engine on the ordinary path. Nothing here evaluates anything.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Sequence

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import SRID, AlertSubscription
from app.domain.location import Coordinates, Location


def _point(latitude: float, longitude: float):
    """A PostGIS point, built the way every other spatial write here builds one."""
    return func.ST_SetSRID(func.ST_MakePoint(longitude, latitude), SRID)


def subscription_values(
    *,
    owner_key: str,
    location: Location,
    alert_types: Sequence[str],
    enabled: bool,
) -> dict[str, Any]:
    """Column values for one subscription, from a resolved location."""
    coordinates = location.coordinates
    return {
        "owner_key": owner_key,
        "location_key": coordinates.cache_key,
        "latitude": coordinates.latitude,
        "longitude": coordinates.longitude,
        "geom": _point(coordinates.latitude, coordinates.longitude),
        # The bare name, not the display name. `admin1` and `country` are
        # columns of their own, and storing them here too makes every later
        # `display_name` read "Hyderabad, Telangana, India, Telangana, India".
        "label": (location.name or coordinates.cache_key)[:200],
        "admin1": location.admin1,
        "country": location.country,
        "timezone": location.timezone,
        "alert_types": list(alert_types),
        "enabled": enabled,
    }


def row_to_location(row: AlertSubscription) -> Location:
    """The resolved place a subscription was created for."""
    return Location(
        coordinates=Coordinates(latitude=row.latitude, longitude=row.longitude),
        name=row.label,
        admin1=row.admin1,
        country=row.country,
        timezone=row.timezone,
    )


class SubscriptionRepository:
    """Queries over :class:`AlertSubscription`."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert(self, values: dict[str, Any]) -> AlertSubscription:
        """Create a watch, or revive and update the one already on that point.

        Asking twice for the same place is the same instruction, not a second
        one. The unique index makes that a database guarantee rather than a
        check-then-write race.
        """
        statement = (
            pg_insert(AlertSubscription)
            .values(**values)
            .on_conflict_do_update(
                index_elements=[AlertSubscription.owner_key, AlertSubscription.location_key],
                set_={
                    "label": values["label"],
                    "admin1": values["admin1"],
                    "country": values["country"],
                    "timezone": values["timezone"],
                    "alert_types": values["alert_types"],
                    "enabled": values["enabled"],
                    "updated_at": datetime.now(timezone.utc),
                },
            )
            .returning(AlertSubscription)
        )
        result = await self._session.execute(statement)
        return result.scalar_one()

    async def list_for(self, owner_key: str) -> list[AlertSubscription]:
        result = await self._session.execute(
            select(AlertSubscription)
            .where(AlertSubscription.owner_key == owner_key)
            .order_by(AlertSubscription.created_at.desc())
        )
        return list(result.scalars().all())

    async def get(self, owner_key: str, subscription_id: uuid.UUID) -> AlertSubscription | None:
        result = await self._session.execute(
            select(AlertSubscription).where(
                AlertSubscription.id == subscription_id,
                AlertSubscription.owner_key == owner_key,
            )
        )
        return result.scalar_one_or_none()

    async def due(self, *, limit: int = 100) -> list[AlertSubscription]:
        """Enabled watches, least recently evaluated first.

        Ordering by staleness is what keeps one busy location from starving the
        others when there are more watches than one pass can cover.
        """
        result = await self._session.execute(
            select(AlertSubscription)
            .where(AlertSubscription.enabled.is_(True))
            .order_by(AlertSubscription.last_evaluated_at.asc().nulls_first())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def set_enabled(
        self, owner_key: str, subscription_id: uuid.UUID, enabled: bool
    ) -> AlertSubscription | None:
        result = await self._session.execute(
            update(AlertSubscription)
            .where(
                AlertSubscription.id == subscription_id,
                AlertSubscription.owner_key == owner_key,
            )
            .values(enabled=enabled, updated_at=datetime.now(timezone.utc))
            .returning(AlertSubscription)
        )
        return result.scalar_one_or_none()

    async def mark_evaluated(self, subscription_id: uuid.UUID, when: datetime) -> None:
        await self._session.execute(
            update(AlertSubscription)
            .where(AlertSubscription.id == subscription_id)
            .values(last_evaluated_at=when)
        )

    async def delete(self, owner_key: str, subscription_id: uuid.UUID) -> bool:
        """Remove a watch. The alerts it produced are history and stay."""
        row = await self.get(owner_key, subscription_id)
        if row is None:
            return False
        await self._session.delete(row)
        return True
