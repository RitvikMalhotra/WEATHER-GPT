"""Watched locations: the standing instruction behind the Alerts panel.

The property that matters most here is a negative one. This feature adds a way
to *ask* for evaluation on a schedule; it must not add a way to *decide* that
weather is dangerous. So alongside the ordinary create/read/toggle/delete
tests, these pin that evaluation goes through the existing weather pipeline and
that every alert shown came out of the alert store rather than out of this
layer.

The database is faked. A subscription is a row and a session, and PostGIS adds
nothing to the questions being asked — the real spatial behaviour is already
covered by the integration suite.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pytest

from app.core.exceptions import DatabaseUnavailableError, LocationNotFoundError
from app.domain.alert import (
    Alert,
    AlertEvidence,
    AlertKind,
    AlertSeverity,
    AlertSourceType,
    AlertStatus,
    AlertType,
)
from app.domain.location import Coordinates, Location
from app.domain.provenance import DataProvenance
from app.services.alerts import AlertMatch
from app.services.monitor import AlertMonitor
from app.services.subscriptions import (
    AmbiguousLocationError,
    SubscriptionService,
    available_alert_types,
)

NOW = datetime(2026, 9, 5, 12, 0, tzinfo=timezone.utc)
OWNER = "client-abcdef12"


def place(name: str, latitude: float, longitude: float, admin1: str = "Telangana") -> Location:
    return Location(
        coordinates=Coordinates(latitude=latitude, longitude=longitude),
        name=name,
        admin1=admin1,
        country="India",
        timezone="Asia/Kolkata",
    )


HYDERABAD = place("Hyderabad", 17.385, 78.4867)
MIYAPUR = place("Miyapur, Hyderabad", 17.4948, 78.3908)


# --- doubles -----------------------------------------------------------------


@dataclass
class Row:
    """A stand-in for one ``alert_subscriptions`` row."""

    id: uuid.UUID
    owner_key: str
    location_key: str
    latitude: float
    longitude: float
    label: str
    admin1: str | None
    country: str | None
    timezone: str | None
    alert_types: list
    enabled: bool
    created_at: datetime = NOW
    updated_at: datetime = NOW
    last_evaluated_at: datetime | None = None


class FakeDatabase:
    """Holds rows in a dict and hands out a session that writes to it."""

    def __init__(self) -> None:
        self.rows: dict[uuid.UUID, Row] = {}

    @asynccontextmanager
    async def session(self):
        yield self


class FakeRepository:
    """The queries :class:`SubscriptionService` actually runs."""

    def __init__(self, database: FakeDatabase) -> None:
        self._db = database

    async def upsert(self, values: dict) -> Row:
        # The unique index, in miniature: one watch per owner per point.
        for row in self._db.rows.values():
            if (
                row.owner_key == values["owner_key"]
                and row.location_key == values["location_key"]
            ):
                row.label = values["label"]
                row.alert_types = values["alert_types"]
                row.enabled = values["enabled"]
                row.updated_at = NOW
                return row
        row = Row(
            id=uuid.uuid4(),
            owner_key=values["owner_key"],
            location_key=values["location_key"],
            latitude=values["latitude"],
            longitude=values["longitude"],
            label=values["label"],
            admin1=values["admin1"],
            country=values["country"],
            timezone=values["timezone"],
            alert_types=values["alert_types"],
            enabled=values["enabled"],
        )
        self._db.rows[row.id] = row
        return row

    async def list_for(self, owner_key: str) -> list[Row]:
        return [row for row in self._db.rows.values() if row.owner_key == owner_key]

    async def get(self, owner_key: str, subscription_id: uuid.UUID) -> Row | None:
        row = self._db.rows.get(subscription_id)
        return row if row is not None and row.owner_key == owner_key else None

    async def due(self, *, limit: int = 100) -> list[Row]:
        enabled = [row for row in self._db.rows.values() if row.enabled]
        enabled.sort(key=lambda row: row.last_evaluated_at or datetime.min.replace(tzinfo=timezone.utc))
        return enabled[:limit]

    async def set_enabled(self, owner_key: str, subscription_id: uuid.UUID, enabled: bool):
        row = await self.get(owner_key, subscription_id)
        if row is None:
            return None
        row.enabled = enabled
        return row

    async def mark_evaluated(self, subscription_id: uuid.UUID, when: datetime) -> None:
        row = self._db.rows.get(subscription_id)
        if row is not None:
            row.last_evaluated_at = when

    async def delete(self, owner_key: str, subscription_id: uuid.UUID) -> bool:
        row = await self.get(owner_key, subscription_id)
        if row is None:
            return False
        del self._db.rows[subscription_id]
        return True


class FakeWeather:
    """Records that the ordinary weather pipeline was driven, and for where."""

    def __init__(self, fail: bool = False) -> None:
        self.current_calls: list = []
        self.forecast_calls: list = []
        self._fail = fail

    async def get_current(self, query, **kwargs):
        if self._fail:
            raise RuntimeError("provider down")
        self.current_calls.append(query)
        return None

    async def get_forecast(self, query, **kwargs):
        if self._fail:
            raise RuntimeError("provider down")
        self.forecast_calls.append(query)
        return None


class FakeAlerts:
    """Stands in for the alert store. Returns only what it is given."""

    def __init__(self, matches: list[AlertMatch] | None = None) -> None:
        self.matches = matches or []
        self.filters: list = []

    async def search(self, criteria):
        self.filters.append(criteria)
        return list(self.matches)


class FakeGeocoding:
    def __init__(self, results: list[Location]) -> None:
        self.results = results

    async def search(self, query, *, limit=5, near=None):
        return list(self.results)


def build(
    *,
    database: FakeDatabase | None = None,
    weather: FakeWeather | None = None,
    alerts: FakeAlerts | None = None,
    geocoding: FakeGeocoding | None = None,
) -> SubscriptionService:
    service = SubscriptionService(
        database if database is not None else FakeDatabase(),
        weather=weather or FakeWeather(),
        alerts=alerts or FakeAlerts(),
        geocoding=geocoding or FakeGeocoding([HYDERABAD]),
    )
    return service


@pytest.fixture(autouse=True)
def _fake_repository(monkeypatch):
    """Point the service's repository at the fake database it was handed."""
    import app.services.subscriptions as module

    def repository(session):
        return FakeRepository(session)

    monkeypatch.setattr(module, "SubscriptionRepository", repository)

    @asynccontextmanager
    async def passthrough(_name):
        yield

    monkeypatch.setattr(module, "translate_database_errors", passthrough)


def alert_match(
    severity: AlertSeverity = AlertSeverity.WARNING,
    alert_type: AlertType = AlertType.HEAVY_RAINFALL,
    distance_m: float = 0.0,
) -> AlertMatch:
    return AlertMatch(
        alert=Alert(
            id=uuid.uuid4(),
            location=HYDERABAD,
            alert_type=alert_type,
            severity=severity,
            status=AlertStatus.ACTIVE,
            source_type=AlertSourceType.DETERMINISTIC_RULE,
            kind=AlertKind.OBSERVED,
            rule_id="HEAVY_RAINFALL_01",
            title="Heavy rainfall",
            description="Observed rainfall of 32 mm meets the warning threshold of 30 mm.",
            triggered_at=NOW,
            valid_from=NOW,
            valid_until=NOW + timedelta(hours=3),
            evidence=AlertEvidence(
                rule_id="HEAVY_RAINFALL_01",
                variable="precipitation_mm",
                observed_value=32.0,
                threshold=30.0,
                unit="mm",
                comparison=">=",
                sample_window="hour",
            ),
            provenance=DataProvenance(
                provider_id="open-meteo",
                provider_name="Open-Meteo",
                fetched_at=NOW,
                source_url="https://open-meteo.com",
            ),
        ),
        distance_m=distance_m,
    )


# --- creating and resolving ---------------------------------------------------


@pytest.mark.asyncio
async def test_a_watch_stores_the_resolved_point_not_the_typed_words():
    """Re-resolving a name later could land on a different place of that name."""
    database = FakeDatabase()
    service = build(database=database)

    watch = await service.create(owner_key=OWNER, location=MIYAPUR)

    assert watch.location.coordinates.latitude == pytest.approx(17.4948)
    assert watch.location.coordinates.longitude == pytest.approx(78.3908)
    [row] = database.rows.values()
    assert row.latitude == pytest.approx(17.4948)
    assert row.label == MIYAPUR.name


@pytest.mark.asyncio
async def test_a_place_is_resolved_through_the_existing_gazetteer():
    service = build(geocoding=FakeGeocoding([HYDERABAD]))

    resolved = await service.resolve("Hyderabad")

    assert resolved.name == "Hyderabad"
    assert resolved.coordinates.latitude == pytest.approx(17.385)


@pytest.mark.asyncio
async def test_an_ambiguous_name_is_refused_with_its_candidates():
    """Watching the wrong town for a week is worse than one extra question."""
    candidates = [
        place("Miyapur", 26.9, 75.8, admin1="Rajasthan"),
        place("Miyapur", 17.49, 78.39, admin1="Telangana"),
    ]
    service = build(geocoding=FakeGeocoding(candidates))

    with pytest.raises(AmbiguousLocationError) as raised:
        await service.resolve("Miyapur")

    assert raised.value.code == "AMBIGUOUS_LOCATION"
    assert len(raised.value.details["candidates"]) == 2


@pytest.mark.asyncio
async def test_an_unmatched_name_is_a_clean_not_found():
    service = build(geocoding=FakeGeocoding([]))

    with pytest.raises(LocationNotFoundError):
        await service.resolve("qqqqqqqq")


@pytest.mark.asyncio
async def test_watching_the_same_place_twice_updates_one_row():
    database = FakeDatabase()
    service = build(database=database)

    first = await service.create(owner_key=OWNER, location=HYDERABAD)
    second = await service.create(
        owner_key=OWNER, location=HYDERABAD, alert_types=("high_wind",)
    )

    assert first.id == second.id
    assert len(database.rows) == 1
    assert second.alert_types == ("high_wind",)


# --- lifecycle ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_watch_can_be_paused_and_resumed():
    service = build()
    watch = await service.create(owner_key=OWNER, location=HYDERABAD)

    paused = await service.set_enabled(OWNER, watch.id, False)
    assert paused is not None and paused.enabled is False

    resumed = await service.set_enabled(OWNER, watch.id, True)
    assert resumed is not None and resumed.enabled is True


@pytest.mark.asyncio
async def test_a_watch_can_be_removed():
    database = FakeDatabase()
    service = build(database=database)
    watch = await service.create(owner_key=OWNER, location=HYDERABAD)

    assert await service.delete(OWNER, watch.id) is True
    assert database.rows == {}
    assert await service.delete(OWNER, watch.id) is False


@pytest.mark.asyncio
async def test_one_client_cannot_read_or_change_another_clients_watches():
    service = build()
    watch = await service.create(owner_key=OWNER, location=HYDERABAD)

    assert await service.list_for("someone-else-000") == []
    assert await service.set_enabled("someone-else-000", watch.id, False) is None
    assert await service.delete("someone-else-000", watch.id) is False


@pytest.mark.asyncio
async def test_a_saved_watch_reads_back_with_its_configuration():
    service = build()
    await service.create(
        owner_key=OWNER, location=HYDERABAD, alert_types=("heavy_rainfall", "high_wind")
    )

    [watch] = await service.list_for(OWNER)

    assert watch.location.display_name == HYDERABAD.display_name
    assert watch.alert_types == ("heavy_rainfall", "high_wind")
    assert watch.enabled is True
    assert watch.last_evaluated_at is None


# --- evaluation ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_evaluating_a_watch_drives_the_ordinary_weather_pipeline():
    """The whole integration. No alert logic lives in this layer.

    Current conditions *and* forecast, because the engine draws different
    alerts from each: "it is raining" and "rain is forecast" are different
    claims and the rule set covers both.
    """
    weather = FakeWeather()
    service = build(weather=weather)
    watch = await service.create(owner_key=OWNER, location=HYDERABAD)

    await service.evaluate(watch)

    assert len(weather.current_calls) == 1
    assert len(weather.forecast_calls) == 1
    # The stored point, not the stored label.
    assert weather.current_calls[0].coordinates.latitude == pytest.approx(17.385)
    assert weather.current_calls[0].place is None


@pytest.mark.asyncio
async def test_alerts_come_from_the_alert_store_and_are_returned_unchanged():
    match = alert_match()
    service = build(alerts=FakeAlerts([match]))
    watch = await service.create(owner_key=OWNER, location=HYDERABAD)

    found = await service.evaluate(watch)

    assert [item.alert.id for item in found] == [match.alert.id]
    assert found[0].alert.severity is AlertSeverity.WARNING
    assert found[0].alert.evidence.observed_value == 32.0
    assert found[0].alert.source_type is AlertSourceType.DETERMINISTIC_RULE


@pytest.mark.asyncio
async def test_only_active_alerts_around_the_watched_point_are_asked_for():
    alerts = FakeAlerts()
    service = build(alerts=alerts)
    watch = await service.create(
        owner_key=OWNER, location=HYDERABAD, alert_types=("high_wind",)
    )

    await service.alerts_for(watch)

    [criteria] = alerts.filters
    assert criteria.statuses == (AlertStatus.ACTIVE.value,)
    assert criteria.alert_types == ("high_wind",)
    assert criteria.is_spatial
    assert criteria.latitude == pytest.approx(17.385)


@pytest.mark.asyncio
async def test_no_alerts_is_reported_as_no_alerts_not_as_a_failure():
    service = build(alerts=FakeAlerts([]))
    watch = await service.create(owner_key=OWNER, location=HYDERABAD)

    assert await service.evaluate(watch) == []


@pytest.mark.asyncio
async def test_a_watch_records_when_it_was_last_looked_at():
    """"Nothing found" and "not checked yet" must not look the same."""
    database = FakeDatabase()
    service = build(database=database)
    watch = await service.create(owner_key=OWNER, location=HYDERABAD)
    assert watch.last_evaluated_at is None

    await service.evaluate(watch)

    [refreshed] = await service.list_for(OWNER)
    assert refreshed.last_evaluated_at is not None


@pytest.mark.asyncio
async def test_a_provider_failure_does_not_take_the_panel_down():
    """The previous evaluation's alerts stay valid until they expire."""
    service = build(weather=FakeWeather(fail=True), alerts=FakeAlerts([alert_match()]))
    watch = await service.create(owner_key=OWNER, location=HYDERABAD)

    found = await service.evaluate(watch)

    assert len(found) == 1


@pytest.mark.asyncio
async def test_a_paused_watch_reports_no_alerts_without_asking_the_store():
    alerts = FakeAlerts([alert_match()])
    service = build(alerts=alerts)
    watch = await service.create(owner_key=OWNER, location=HYDERABAD)
    await service.set_enabled(OWNER, watch.id, False)

    [status] = await service.status_for(OWNER)

    assert status.watch.enabled is False
    assert status.alerts == []
    assert alerts.filters == []


# --- automatic evaluation -----------------------------------------------------


@pytest.mark.asyncio
async def test_a_sweep_evaluates_every_enabled_watch():
    weather = FakeWeather()
    service = build(weather=weather)
    await service.create(owner_key=OWNER, location=HYDERABAD)
    await service.create(owner_key=OWNER, location=MIYAPUR)

    evaluated = await service.evaluate_due()

    assert evaluated == 2
    assert len(weather.current_calls) == 2


@pytest.mark.asyncio
async def test_a_sweep_skips_paused_watches():
    weather = FakeWeather()
    service = build(weather=weather)
    watch = await service.create(owner_key=OWNER, location=HYDERABAD)
    await service.set_enabled(OWNER, watch.id, False)

    assert await service.evaluate_due() == 0
    assert weather.current_calls == []


@pytest.mark.asyncio
async def test_the_monitor_sweeps_and_survives_a_bad_pass():
    """A monitor that dies silently is worse than one that logs and retries."""

    class Broken(SubscriptionService):
        def __init__(self):
            pass

        @property
        def enabled(self):
            return True

        async def evaluate_due(self, *, limit=100):
            raise RuntimeError("database went away")

    monitor = AlertMonitor(Broken(), interval_seconds=30.0)

    assert await monitor.sweep_once() == 0
    assert monitor.last_sweep_at is None


@pytest.mark.asyncio
async def test_the_monitor_records_a_successful_sweep():
    service = build()
    await service.create(owner_key=OWNER, location=HYDERABAD)
    monitor = AlertMonitor(service, interval_seconds=30.0)

    assert await monitor.sweep_once() == 1
    assert monitor.last_sweep_at is not None
    assert monitor.last_sweep_count == 1


@pytest.mark.asyncio
async def test_the_monitor_does_not_start_without_somewhere_to_store_watches():
    service = SubscriptionService(
        None, weather=FakeWeather(), alerts=FakeAlerts(), geocoding=FakeGeocoding([])
    )
    monitor = AlertMonitor(service)

    monitor.start()

    assert monitor.running is False


# --- degradation --------------------------------------------------------------


@pytest.mark.asyncio
async def test_without_a_database_watching_is_refused_honestly():
    service = SubscriptionService(
        None, weather=FakeWeather(), alerts=FakeAlerts(), geocoding=FakeGeocoding([HYDERABAD])
    )

    assert service.enabled is False
    with pytest.raises(DatabaseUnavailableError):
        await service.create(owner_key=OWNER, location=HYDERABAD)
    with pytest.raises(DatabaseUnavailableError):
        await service.list_for(OWNER)


def test_the_panel_can_only_offer_rules_the_engine_actually_runs():
    """A type list kept by hand would eventually advertise a rule that is gone."""
    offered = available_alert_types()

    assert offered == [item.value for item in AlertType]
    assert "heavy_rainfall" in offered
