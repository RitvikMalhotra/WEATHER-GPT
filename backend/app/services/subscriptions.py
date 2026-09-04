"""Watching a place, rather than being asked about it each time.

The feature this adds is *standing intent*. Everything else already existed:

    subscription -> WeatherService.get_current / get_forecast
                        -> ingestion + validation
                        -> AlertService -> AlertEngine -> rules
                        -> weather_alerts
                    AlertService.search -> what to show

So evaluating a watch is exactly one thing: run the ordinary weather request
for its point. The pipeline's own side effects do the rest — validation gates
the data, the deterministic engine decides whether a threshold was crossed, and
the alert service handles deduplication, resolution and expiry. No rule, no
threshold and no severity is computed here, and none may be: this module exists
so a person does not have to keep asking, not so anything new can be decided.

A language model is not involved at any point in this path.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from app.config.logging import get_logger
from app.core.exceptions import DatabaseUnavailableError, LocationNotFoundError
from app.db.engine import Database, translate_database_errors
from app.db.repositories import SubscriptionRepository, row_to_location, subscription_values
from app.db.repositories.alerts import AlertFilter
from app.domain.alert import AlertStatus, AlertType
from app.domain.location import Coordinates, Location
from app.services.alerts import AlertMatch, AlertService
from app.services.geocoding import GeocodingService
from app.services.weather_service import LocationQuery, WeatherService

logger = get_logger(__name__)

#: How far around a watched point an alert is considered to be "here". Matches
#: the radius the alerts endpoint uses, so the panel and the API agree.
WATCH_RADIUS_KM = 25.0


class AmbiguousLocationError(LocationNotFoundError):
    """The place named matches several real places in different regions.

    Not an error in the usual sense: it carries the candidates so the caller
    can ask which one was meant. Choosing silently is the failure this exists
    to prevent.
    """

    code = "AMBIGUOUS_LOCATION"
    status_code = 409
    message = "Multiple locations matched that name."


@dataclass(frozen=True, slots=True)
class Watch:
    """A saved subscription, as the rest of the application sees it."""

    id: uuid.UUID
    location: Location
    alert_types: tuple[str, ...]
    enabled: bool
    created_at: datetime
    updated_at: datetime
    last_evaluated_at: datetime | None


@dataclass(frozen=True, slots=True)
class WatchStatus:
    """A watch together with what the engine currently has for its point."""

    watch: Watch
    alerts: list[AlertMatch]
    #: False when nothing has run for this point yet, which is a different
    #: thing from "nothing was found" and must not be shown as if it were.
    evaluated: bool


def _to_watch(row) -> Watch:
    return Watch(
        id=row.id,
        location=row_to_location(row),
        alert_types=tuple(row.alert_types or ()),
        enabled=row.enabled,
        created_at=row.created_at,
        updated_at=row.updated_at,
        last_evaluated_at=row.last_evaluated_at,
    )


class SubscriptionService:
    """Create, read and evaluate the places a person is watching."""

    def __init__(
        self,
        database: Database | None,
        *,
        weather: WeatherService,
        alerts: AlertService,
        geocoding: GeocodingService,
    ) -> None:
        self._database = database
        self._weather = weather
        self._alerts = alerts
        self._geocoding = geocoding

    @property
    def enabled(self) -> bool:
        return self._database is not None

    def _require_database(self) -> Database:
        if self._database is None:
            raise DatabaseUnavailableError(
                "Watching a location requires a configured database.",
                details={"reason": "persistence_disabled"},
            )
        return self._database

    # ---------------------------------------------------------------- create

    async def resolve(self, query: str, *, near: Coordinates | None = None) -> Location:
        """Resolve a typed place to the one location a watch will be created for.

        Raises:
            LocationNotFoundError: nothing matched.
            AmbiguousLocationError: several places carry that name, in
                different regions. The candidates travel on the exception.
        """
        candidates = await self._geocoding.search(query, limit=5, near=near)
        if not candidates:
            raise LocationNotFoundError(
                f"No location matched {query.strip()!r}.",
                details={"query": query.strip()},
            )
        if GeocodingService.ambiguous(candidates):
            raise AmbiguousLocationError(
                "Multiple locations matched that name.",
                details={
                    "query": query.strip(),
                    "candidates": [c.model_dump(mode="json") for c in candidates[:5]],
                },
            )
        return candidates[0]

    async def create(
        self,
        *,
        owner_key: str,
        location: Location,
        alert_types: tuple[str, ...] = (),
        enabled: bool = True,
    ) -> Watch:
        """Start watching a resolved place.

        The location must already be resolved: a watch stores coordinates, not
        the words that were typed, so it cannot drift onto a different place of
        the same name later.
        """
        database = self._require_database()
        values = subscription_values(
            owner_key=owner_key,
            location=location,
            alert_types=alert_types,
            enabled=enabled,
        )

        async with translate_database_errors("subscriptions.create"):
            async with database.session() as session:
                row = await SubscriptionRepository(session).upsert(values)
                watch = _to_watch(row)

        logger.info(
            "subscriptions.created",
            extra={"location": watch.location.display_name, "types": list(alert_types)},
        )
        return watch

    # ------------------------------------------------------------------ read

    async def list_for(self, owner_key: str) -> list[Watch]:
        database = self._require_database()
        async with translate_database_errors("subscriptions.list"):
            async with database.session() as session:
                rows = await SubscriptionRepository(session).list_for(owner_key)
                return [_to_watch(row) for row in rows]

    async def set_enabled(
        self, owner_key: str, subscription_id: uuid.UUID, enabled: bool
    ) -> Watch | None:
        database = self._require_database()
        async with translate_database_errors("subscriptions.toggle"):
            async with database.session() as session:
                row = await SubscriptionRepository(session).set_enabled(
                    owner_key, subscription_id, enabled
                )
                return _to_watch(row) if row is not None else None

    async def delete(self, owner_key: str, subscription_id: uuid.UUID) -> bool:
        database = self._require_database()
        async with translate_database_errors("subscriptions.delete"):
            async with database.session() as session:
                return await SubscriptionRepository(session).delete(owner_key, subscription_id)

    # -------------------------------------------------------------- evaluate

    async def alerts_for(self, watch: Watch) -> list[AlertMatch]:
        """What the deterministic engine currently holds for a watched point.

        A plain read of ``weather_alerts``. The rows were written by the engine
        during evaluation; nothing is decided here, and a watch that filters by
        type filters the *display*, never the evaluation.
        """
        criteria = AlertFilter(
            latitude=watch.location.coordinates.latitude,
            longitude=watch.location.coordinates.longitude,
            radius_m=WATCH_RADIUS_KM * 1000.0,
            statuses=(AlertStatus.ACTIVE.value,),
            alert_types=watch.alert_types,
            valid_at=datetime.now(timezone.utc),
        )
        return await self._alerts.search(criteria)

    async def evaluate(self, watch: Watch) -> list[AlertMatch]:
        """Run the ordinary weather pipeline for a watched point, then read back.

        Current conditions and the forecast are both requested because the
        engine draws different alerts from each — "it is raining" and "rain is
        forecast" are different claims, and the rule set covers both.

        A provider or database failure is logged and swallowed: a watch that
        could not be refreshed this minute must not take the panel down, and
        the previous evaluation's alerts remain valid until they expire.
        """
        # The stored point, never the stored label: re-resolving a name here
        # could land on a different place of that name than the one watched.
        query = LocationQuery(coordinates=watch.location.coordinates)
        try:
            # Evaluation is a side effect of these two calls, inside the
            # weather service, downstream of validation. That is the whole
            # integration: no alert logic is duplicated here.
            await asyncio.gather(
                self._weather.get_current(query),
                self._weather.get_forecast(query, days=2, include_hourly=True),
            )
        except Exception:  # noqa: BLE001 - a refresh failure is not fatal
            logger.warning(
                "subscriptions.evaluation_failed",
                extra={"location": watch.location.display_name},
            )

        now = datetime.now(timezone.utc)
        if self._database is not None:
            try:
                async with translate_database_errors("subscriptions.mark_evaluated"):
                    async with self._database.session() as session:
                        await SubscriptionRepository(session).mark_evaluated(watch.id, now)
            except DatabaseUnavailableError:
                logger.warning("subscriptions.mark_evaluated_failed")

        return await self.alerts_for(watch)

    async def status_for(self, owner_key: str) -> list[WatchStatus]:
        """Every watch a client holds, with the alerts standing against each."""
        watches = await self.list_for(owner_key)
        statuses: list[WatchStatus] = []
        for watch in watches:
            alerts = await self.alerts_for(watch) if watch.enabled else []
            statuses.append(
                WatchStatus(
                    watch=watch,
                    alerts=alerts,
                    evaluated=watch.last_evaluated_at is not None,
                )
            )
        return statuses

    async def evaluate_due(self, *, limit: int = 100) -> int:
        """Refresh the watches that have gone longest without a look.

        Called by the monitor. Returns how many points were evaluated.
        """
        database = self._require_database()
        async with translate_database_errors("subscriptions.due"):
            async with database.session() as session:
                rows = await SubscriptionRepository(session).due(limit=limit)
                watches = [_to_watch(row) for row in rows]

        for watch in watches:
            await self.evaluate(watch)
        if watches:
            logger.info("subscriptions.swept", extra={"evaluated": len(watches)})
        return len(watches)


#: Alert types a client may ask to be notified about. Read from the domain
#: enum, so a rule added to the engine appears in the UI without a second list
#: being kept in step by hand.
def available_alert_types() -> list[str]:
    return [item.value for item in AlertType]
