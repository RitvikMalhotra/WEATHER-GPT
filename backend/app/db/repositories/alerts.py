"""Reads and writes for alerts.

Two things distinguish this repository from the observation and forecast ones:

* **Deduplication is a read-then-write**, not a blind upsert. An ongoing
  condition must land on the *existing* active alert so its validity extends and
  its severity updates, and that needs the current row first. A partial unique
  index on active rows makes the guarantee hold under concurrency.
* **Lifecycle transitions are set-based.** Expiring stale alerts and resolving
  ones whose condition has lifted are single UPDATE statements, not row-by-row
  work in Python.

Spatial filtering reuses the Phase 3 pattern: ``ST_DWithin`` on the geography
column, index-assisted by GIST, with ``ST_Distance`` for the reported distance.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence

from geoalchemy2.functions import ST_Distance, ST_DWithin
from sqlalchemy import Float, Select, and_, case, func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.mappers import point
from app.db.models import WeatherAlert
from app.domain.alert import AlertStatus

#: Columns refreshed when an ongoing condition is re-evaluated. Identity, the
#: original trigger time and the first-seen timestamp are preserved: an alert
#: that has been active for three hours must not look three seconds old.
_REFRESHABLE = (
    "severity",
    "description",
    "observed_value",
    "threshold",
    "unit",
    "comparison",
    "sample_window",
    "evidence_context",
    "valid_until",
    "fetched_at",
    "provider_name",
    "source_url",
    "license",
    "attribution",
    "elevation_m",
    "timezone",
)


@dataclass(frozen=True, slots=True)
class NearbyAlert:
    """An alert together with how far it sits from the query point."""

    alert: WeatherAlert
    distance_m: float


@dataclass(frozen=True, slots=True)
class AlertFilter:
    """Criteria for an alert search. Every field is optional."""

    latitude: float | None = None
    longitude: float | None = None
    radius_m: float | None = None
    severities: tuple[str, ...] = ()
    alert_types: tuple[str, ...] = ()
    statuses: tuple[str, ...] = ()
    kinds: tuple[str, ...] = ()
    provider_id: str | None = None
    rule_id: str | None = None
    valid_at: datetime | None = None
    limit: int = 200

    @property
    def is_spatial(self) -> bool:
        return (
            self.latitude is not None
            and self.longitude is not None
            and self.radius_m is not None
        )


class AlertRepository:
    """Persistence operations for :class:`WeatherAlert`."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # --- Writes -------------------------------------------------------------

    async def find_active_by_dedup_key(self, dedup_key: str) -> WeatherAlert | None:
        """The open alert for this identity, if one exists.

        Locked for update so two concurrent evaluations of the same condition
        cannot both decide to insert.
        """
        result = await self._session.execute(
            select(WeatherAlert)
            .where(
                WeatherAlert.dedup_key == dedup_key,
                WeatherAlert.status == AlertStatus.ACTIVE.value,
            )
            .with_for_update()
        )
        return result.scalars().first()

    async def insert(self, values: dict[str, Any]) -> WeatherAlert:
        """Open a new alert."""
        result = await self._session.execute(
            insert(WeatherAlert).values(**values).returning(WeatherAlert)
        )
        return result.scalars().one()

    async def refresh(self, alert: WeatherAlert, values: dict[str, Any]) -> None:
        """Update an open alert with the latest evaluation of the same condition.

        Severity is allowed to move in both directions: a storm that eases is as
        important to reflect as one that intensifies.
        """
        for column in _REFRESHABLE:
            if column in values:
                setattr(alert, column, values[column])

    # --- Lifecycle ----------------------------------------------------------

    async def expire_stale(self, *, now: datetime) -> int:
        """Mark active alerts whose validity window has elapsed as expired.

        Rows are never deleted. An expired alert is the record of something that
        was true and then stopped being true, which is exactly what false-alarm
        analysis needs.
        """
        result = await self._session.execute(
            update(WeatherAlert)
            .where(
                WeatherAlert.status == AlertStatus.ACTIVE.value,
                WeatherAlert.valid_until < now,
            )
            .values(status=AlertStatus.EXPIRED.value)
        )
        return int(result.rowcount or 0)

    async def resolve_except(
        self,
        *,
        location_key: str,
        provider_id: str,
        kind: str,
        keep: Sequence[str],
        now: datetime,
    ) -> int:
        """Resolve active alerts at a place that this evaluation did not re-raise.

        Called after evaluating observed conditions: any alert still open for
        this place and source whose rule did not fire again has had its
        condition lift, which is a *resolution* rather than an expiry.
        """
        conditions = [
            WeatherAlert.status == AlertStatus.ACTIVE.value,
            WeatherAlert.location_key == location_key,
            WeatherAlert.provider_id == provider_id,
            WeatherAlert.kind == kind,
        ]
        if keep:
            conditions.append(WeatherAlert.dedup_key.notin_(list(keep)))

        result = await self._session.execute(
            update(WeatherAlert)
            .where(and_(*conditions))
            .values(status=AlertStatus.RESOLVED.value, resolved_at=now)
        )
        return int(result.rowcount or 0)

    # --- Reads --------------------------------------------------------------

    async def search(self, criteria: AlertFilter) -> list[NearbyAlert]:
        """Alerts matching a filter, most severe and most recent first.

        Distance is reported as zero for a non-spatial search: there is no
        reference point to measure from.
        """
        if criteria.is_spatial:
            origin = point(criteria.latitude, criteria.longitude)  # type: ignore[arg-type]
            distance = ST_Distance(WeatherAlert.geom, origin).label("distance_m")
            statement: Select = select(WeatherAlert, distance).where(
                ST_DWithin(WeatherAlert.geom, origin, criteria.radius_m)
            )
        else:
            statement = select(WeatherAlert, func.cast(0.0, Float).label("distance_m"))

        statement = statement.where(*_predicates(criteria))
        statement = statement.order_by(
            _severity_order().desc(),
            WeatherAlert.valid_from.desc(),
        ).limit(criteria.limit)

        result = await self._session.execute(statement)
        return [
            NearbyAlert(alert=row[0], distance_m=float(row[1] or 0.0))
            for row in result.all()
        ]

    async def count(self) -> int:
        """Total stored alerts, of any status. Used by diagnostics and tests."""
        result = await self._session.execute(
            select(func.count()).select_from(WeatherAlert)
        )
        return int(result.scalar_one())


def _predicates(criteria: AlertFilter) -> list[Any]:
    """Non-spatial filters, applied only where the caller supplied them."""
    predicates: list[Any] = []
    if criteria.statuses:
        predicates.append(WeatherAlert.status.in_(criteria.statuses))
    if criteria.severities:
        predicates.append(WeatherAlert.severity.in_(criteria.severities))
    if criteria.alert_types:
        predicates.append(WeatherAlert.alert_type.in_(criteria.alert_types))
    if criteria.kinds:
        predicates.append(WeatherAlert.kind.in_(criteria.kinds))
    if criteria.provider_id is not None:
        predicates.append(WeatherAlert.provider_id == criteria.provider_id)
    if criteria.rule_id is not None:
        predicates.append(WeatherAlert.rule_id == criteria.rule_id)
    if criteria.valid_at is not None:
        predicates.append(WeatherAlert.valid_from <= criteria.valid_at)
        predicates.append(WeatherAlert.valid_until >= criteria.valid_at)
    return predicates


def _severity_order():
    """Order by the severity ladder rather than alphabetically.

    Sorting the stored strings would put "extreme" below "info", which is
    exactly backwards for a list a person scans top-down.
    """
    return case(
        {
            "info": 0,
            "watch": 1,
            "warning": 2,
            "severe": 3,
            "extreme": 4,
        },
        value=WeatherAlert.severity,
        else_=0,
    )
