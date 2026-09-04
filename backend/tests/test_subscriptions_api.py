"""The HTTP surface behind the Alerts panel.

Exercised through the real app with the subscription service replaced, so the
routing, the client header, the error envelope and the response shape are the
ones a browser will actually meet.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app.core.dependencies import get_subscription_service
from app.core.exceptions import DatabaseUnavailableError, LocationNotFoundError
from app.domain.alert import (
    Alert,
    AlertEvidence,
    AlertKind,
    AlertSeverity,
    AlertSourceType,
    AlertStatus,
)
from app.domain.location import Coordinates, Location
from app.domain.provenance import DataProvenance
from app.main import create_app
from app.services.alerts import AlertMatch
from app.services.subscriptions import AmbiguousLocationError, Watch, WatchStatus

NOW = datetime(2026, 9, 5, 12, 0, tzinfo=timezone.utc)
CLIENT = {"X-WeatherGPT-Client": "client-abcdef12"}

HYDERABAD = Location(
    coordinates=Coordinates(latitude=17.385, longitude=78.4867),
    name="Hyderabad",
    admin1="Telangana",
    country="India",
    timezone="Asia/Kolkata",
)


def a_watch(enabled: bool = True, evaluated: bool = True) -> Watch:
    return Watch(
        id=uuid.UUID("11111111-2222-3333-4444-555555555555"),
        location=HYDERABAD,
        alert_types=(),
        enabled=enabled,
        created_at=NOW,
        updated_at=NOW,
        last_evaluated_at=NOW if evaluated else None,
    )


def a_match() -> AlertMatch:
    return AlertMatch(
        alert=Alert(
            id=uuid.uuid4(),
            location=HYDERABAD,
            alert_type="heavy_rainfall",
            severity=AlertSeverity.SEVERE,
            status=AlertStatus.ACTIVE,
            source_type=AlertSourceType.DETERMINISTIC_RULE,
            kind=AlertKind.OBSERVED,
            rule_id="HEAVY_RAINFALL_01",
            title="Heavy rainfall",
            description="Observed rainfall of 42 mm meets the severe threshold of 40 mm.",
            triggered_at=NOW,
            valid_from=NOW,
            valid_until=NOW + timedelta(hours=3),
            evidence=AlertEvidence(
                rule_id="HEAVY_RAINFALL_01",
                variable="precipitation_mm",
                observed_value=42.0,
                threshold=40.0,
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
        distance_m=0.0,
    )


class StubService:
    """Records what the routes asked for and answers with fixed data."""

    def __init__(self, **overrides) -> None:
        self.enabled = True
        self.created: list = []
        self.evaluated: list = []
        self.deleted: list = []
        self.toggled: list = []
        self.watches: list[Watch] = []
        self.alerts: list[AlertMatch] = []
        self.resolve_error: Exception | None = None
        self.__dict__.update(overrides)

    async def resolve(self, query, *, near=None):
        if self.resolve_error is not None:
            raise self.resolve_error
        return HYDERABAD

    async def create(self, *, owner_key, location, alert_types=(), enabled=True):
        self.created.append((owner_key, location, alert_types))
        watch = a_watch(evaluated=False)
        self.watches = [watch]
        return watch

    async def evaluate(self, watch):
        self.evaluated.append(watch.id)
        return list(self.alerts)

    async def alerts_for(self, watch):
        return list(self.alerts)

    async def list_for(self, owner_key):
        return list(self.watches)

    async def status_for(self, owner_key):
        return [
            WatchStatus(
                watch=watch,
                alerts=list(self.alerts) if watch.enabled else [],
                evaluated=watch.last_evaluated_at is not None,
            )
            for watch in self.watches
        ]

    async def set_enabled(self, owner_key, subscription_id, enabled):
        self.toggled.append((subscription_id, enabled))
        if not any(w.id == subscription_id for w in self.watches):
            return None
        watch = a_watch(enabled=enabled)
        self.watches = [watch]
        return watch

    async def delete(self, owner_key, subscription_id):
        self.deleted.append(subscription_id)
        return any(w.id == subscription_id for w in self.watches)


@pytest.fixture
def stub():
    return StubService()


@pytest.fixture
def client(settings, stub):
    app = create_app(settings)
    app.dependency_overrides[get_subscription_service] = lambda: stub
    with TestClient(app) as running:
        yield running


# --- creating -----------------------------------------------------------------


def test_watching_a_confirmed_place_stores_its_coordinates(client, stub):
    response = client.post(
        "/api/v1/alerts/subscriptions",
        headers=CLIENT,
        json={
            "latitude": 17.385,
            "longitude": 78.4867,
            "name": "Hyderabad",
            "admin1": "Telangana",
            "country": "India",
            "alert_types": [],
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["location"]["latitude"] == pytest.approx(17.385)
    assert body["enabled"] is True
    # The place the panel confirmed, not the words typed into it.
    _, location, _ = stub.created[0]
    assert location.coordinates.latitude == pytest.approx(17.385)


def test_a_new_watch_is_evaluated_immediately(client, stub):
    """An empty panel before anything has looked reads as an all-clear."""
    stub.alerts = [a_match()]

    body = client.post(
        "/api/v1/alerts/subscriptions",
        headers=CLIENT,
        json={"latitude": 17.385, "longitude": 78.4867, "name": "Hyderabad"},
    ).json()

    assert stub.evaluated, "the first evaluation must not wait for the next sweep"
    assert body["evaluated"] is True
    assert len(body["alerts"]) == 1


def test_a_name_can_be_resolved_by_the_backend(client, stub):
    response = client.post(
        "/api/v1/alerts/subscriptions", headers=CLIENT, json={"query": "Hyderabad"}
    )

    assert response.status_code == 201
    assert response.json()["location"]["name"].startswith("Hyderabad")


def test_an_ambiguous_name_is_refused_with_its_candidates(client, stub):
    stub.resolve_error = AmbiguousLocationError(
        "Multiple locations matched that name.",
        details={"candidates": [{"name": "Miyapur", "admin1": "Telangana"}]},
    )

    response = client.post(
        "/api/v1/alerts/subscriptions", headers=CLIENT, json={"query": "Miyapur"}
    )

    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "AMBIGUOUS_LOCATION"
    assert error["details"]["candidates"]


def test_an_unknown_place_is_a_clean_not_found(client, stub):
    stub.resolve_error = LocationNotFoundError("No location matched 'qqqq'.")

    response = client.post(
        "/api/v1/alerts/subscriptions", headers=CLIENT, json={"query": "qqqq"}
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "LOCATION_NOT_FOUND"


def test_a_request_without_a_client_header_is_rejected(client):
    response = client.post(
        "/api/v1/alerts/subscriptions", json={"latitude": 17.0, "longitude": 78.0}
    )

    assert response.status_code == 422


# --- reading ------------------------------------------------------------------


def test_the_panel_is_told_which_rules_the_engine_runs(client, stub):
    body = client.get("/api/v1/alerts/subscriptions", headers=CLIENT).json()

    assert "heavy_rainfall" in body["available_alert_types"]


def test_a_watch_with_no_alerts_says_so_rather_than_failing(client, stub):
    stub.watches = [a_watch()]

    body = client.get("/api/v1/alerts/subscriptions", headers=CLIENT).json()

    assert body["count"] == 1
    assert body["subscriptions"][0]["alerts"] == []
    assert body["subscriptions"][0]["evaluated"] is True


def test_an_alert_travels_with_its_severity_evidence_and_source(client, stub):
    """The panel must be able to show why a rule fired, not just that it did."""
    stub.watches = [a_watch()]
    stub.alerts = [a_match()]

    body = client.get("/api/v1/alerts/subscriptions", headers=CLIENT).json()
    [alert] = body["subscriptions"][0]["alerts"]

    assert alert["severity"] == "severe"
    assert alert["source_type"] == "deterministic_rule"
    assert alert["evidence"]["observed_value"] == 42.0
    assert alert["evidence"]["threshold"] == 40.0
    assert alert["provenance"]["provider_name"] == "Open-Meteo"


def test_a_watch_never_checked_is_not_reported_as_clear(client, stub):
    stub.watches = [a_watch(evaluated=False)]

    body = client.get("/api/v1/alerts/subscriptions", headers=CLIENT).json()

    assert body["subscriptions"][0]["evaluated"] is False
    assert body["subscriptions"][0]["last_evaluated_at"] is None


# --- lifecycle ----------------------------------------------------------------


def test_a_watch_can_be_paused(client, stub):
    stub.watches = [a_watch()]

    body = client.patch(
        f"/api/v1/alerts/subscriptions/{a_watch().id}", headers=CLIENT, json={"enabled": False}
    ).json()

    assert body["enabled"] is False
    assert body["alerts"] == []


def test_pausing_something_that_is_not_yours_is_a_404(client, stub):
    stub.watches = []

    response = client.patch(
        f"/api/v1/alerts/subscriptions/{a_watch().id}", headers=CLIENT, json={"enabled": False}
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "SUBSCRIPTION_NOT_FOUND"


def test_a_watch_can_be_removed(client, stub):
    stub.watches = [a_watch()]

    response = client.delete(f"/api/v1/alerts/subscriptions/{a_watch().id}", headers=CLIENT)

    assert response.status_code == 204
    assert stub.deleted == [a_watch().id]


def test_refresh_runs_an_evaluation_now(client, stub):
    stub.watches = [a_watch()]

    response = client.post(
        f"/api/v1/alerts/subscriptions/{a_watch().id}/refresh", headers=CLIENT
    )

    assert response.status_code == 200
    assert stub.evaluated == [a_watch().id]


def test_refreshing_something_that_is_not_yours_is_a_404(client, stub):
    stub.watches = []

    response = client.post(
        f"/api/v1/alerts/subscriptions/{a_watch().id}/refresh", headers=CLIENT
    )

    assert response.status_code == 404


# --- degradation --------------------------------------------------------------


def test_without_a_database_the_panel_is_told_plainly(client, stub):
    async def refuse(*args, **kwargs):
        raise DatabaseUnavailableError("Watching a location requires a database.")

    stub.status_for = refuse

    response = client.get("/api/v1/alerts/subscriptions", headers=CLIENT)

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "DATABASE_UNAVAILABLE"


# --- the existing alert surface is untouched ----------------------------------


def test_the_original_alerts_endpoint_still_answers(client):
    """This feature adds a way to ask; it changes nothing about the answer."""
    schema = client.get("/openapi.json").json()

    assert "/api/v1/weather/alerts" in schema["paths"]
    assert "/api/v1/alerts/subscriptions" in schema["paths"]
