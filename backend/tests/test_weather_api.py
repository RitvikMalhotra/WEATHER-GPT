"""End-to-end tests for the meteorological endpoints.

The full stack runs: route, dependency, service, cache, pipeline, validator,
provider, normaliser and HTTP client. Only the network is replaced, by a mock
transport serving recorded Open-Meteo payloads — so these tests exercise the
real wiring without ever leaving the process.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config.settings import Settings, get_settings
from app.core.container import build_container
from app.core.dependencies import get_container
from app.main import create_app

WEATHER = "/api/v1/weather/current"
FORECAST = "/api/v1/forecast"
SEARCH = "/api/v1/locations/search"


@pytest.fixture(autouse=True)
def _fresh_observation(current_payload):
    """Date the recorded observation to now.

    The payload is otherwise a fixed recording, and the validation engine
    legitimately cares about how old an observation is. Anchoring it to the
    clock keeps these tests about the API rather than about the calendar.
    """
    local_now = datetime.now(timezone.utc) + timedelta(seconds=19800)
    current_payload["current"]["time"] = local_now.strftime("%Y-%m-%dT%H:%M")


@pytest.fixture
def upstream(current_payload, forecast_payload, geocoding_payload):
    """Routes mock requests to the right recorded payload, and counts calls."""

    state: dict[str, Any] = {
        "forecast_calls": 0,
        "geocoding_calls": 0,
        "fail_with": None,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if state["fail_with"] is not None:
            return httpx.Response(state["fail_with"], json={"reason": "upstream down"})

        if "geocoding-api" in request.url.host:
            state["geocoding_calls"] += 1
            return httpx.Response(200, json=geocoding_payload)

        state["forecast_calls"] += 1
        params = request.url.params
        if "current" in params:
            return httpx.Response(200, json=current_payload)

        # Open-Meteo returns only the series that were requested; mirroring that
        # is what makes "hourly is opt-in" a real assertion rather than a
        # property of the fixture.
        body = dict(forecast_payload)
        if "hourly" not in params:
            body.pop("hourly", None)
            body.pop("hourly_units", None)
        if "daily" not in params:
            body.pop("daily", None)
            body.pop("daily_units", None)
        return httpx.Response(200, json=body)

    state["handler"] = handler
    return state


@pytest.fixture
def client(settings: Settings, upstream):
    """An app whose provider layer talks to a mock transport.

    The whole graph is built by the real composition root and only the socket is
    replaced, so route, service, cache, pipeline, validator, provider and
    normaliser all execute. Dispatch goes through ``upstream["handler"]`` on
    every call, so a test can swap the upstream behaviour mid-flight.
    """
    app = create_app(settings)
    mock_http = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: upstream["handler"](request))
    )
    container = build_container(settings, http_client=mock_http)
    app.dependency_overrides[get_container] = lambda: container

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def _body(response: httpx.Response) -> dict[str, Any]:
    assert response.status_code == 200, response.text
    return response.json()


# --- Current conditions ------------------------------------------------------


def test_current_weather_by_coordinates(client):
    body = _body(client.get(WEATHER, params={"latitude": 28.61, "longitude": 77.21}))

    assert body["current"]["temperature_c"] == pytest.approx(31.4)
    assert body["current"]["wind_speed_ms"] == pytest.approx(5.0)
    assert body["current"]["condition"] == "partly_cloudy"
    assert body["current"]["condition_description"] == "Partly cloudy"


def test_every_response_carries_provenance(client):
    body = _body(client.get(WEATHER, params={"latitude": 28.61, "longitude": 77.21}))
    provenance = body["provenance"]

    assert provenance["provider_id"] == "open-meteo"
    assert provenance["attribution"] == "Weather data by Open-Meteo.com"
    assert provenance["license"] == "CC-BY-4.0"
    assert provenance["fetched_at"]
    assert provenance["cached"] is False


def test_a_place_name_is_geocoded_and_its_labels_are_merged(client, upstream):
    body = _body(client.get(WEATHER, params={"location": "New Delhi"}))

    assert upstream["geocoding_calls"] == 1
    location = body["location"]
    # Labels come from the gazetteer...
    assert location["name"] == "New Delhi"
    assert location["country"] == "India"
    assert location["admin1"] == "Delhi"
    # ...while the coordinates are the grid point the provider actually served.
    assert location["coordinates"]["latitude"] == pytest.approx(28.625)


def test_repeated_requests_are_served_from_cache(client, upstream):
    client.get(WEATHER, params={"latitude": 28.61, "longitude": 77.21})
    first_calls = upstream["forecast_calls"]

    body = _body(client.get(WEATHER, params={"latitude": 28.61, "longitude": 77.21}))

    assert upstream["forecast_calls"] == first_calls
    assert body["provenance"]["cached"] is True


def test_nearby_coordinates_share_a_cache_entry(client, upstream):
    """Cache keys are rounded, so sub-metre jitter is not a cache miss."""
    client.get(WEATHER, params={"latitude": 28.613901, "longitude": 77.209001})
    calls = upstream["forecast_calls"]

    client.get(WEATHER, params={"latitude": 28.613902, "longitude": 77.209002})

    assert upstream["forecast_calls"] == calls


# --- Forecast ----------------------------------------------------------------


def test_daily_forecast_is_returned_by_default(client):
    body = _body(client.get(FORECAST, params={"latitude": 28.61, "longitude": 77.21}))

    assert body["hourly"] == []
    assert len(body["daily"]) == 2
    first = body["daily"][0]
    assert first["date"] == "2026-09-04"
    assert first["temperature_max_c"] == pytest.approx(34.8)
    assert first["wind_speed_max_ms"] == pytest.approx(7.0)


def test_the_hourly_series_is_opt_in(client):
    body = _body(
        client.get(
            FORECAST,
            params={"latitude": 28.61, "longitude": 77.21, "hourly": True},
        )
    )

    assert len(body["hourly"]) == 3
    assert body["hourly"][0]["dew_point_c"] == pytest.approx(23.1)


def test_requesting_neither_series_is_rejected(client):
    response = client.get(
        FORECAST,
        params={"latitude": 28.61, "longitude": 77.21, "hourly": False, "daily": False},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "EMPTY_SERIES_SELECTION"


def test_the_forecast_horizon_is_bounded(client):
    response = client.get(
        FORECAST, params={"latitude": 28.61, "longitude": 77.21, "days": 99}
    )

    assert response.status_code == 422


# --- Location resolution -----------------------------------------------------


def test_location_search_returns_ranked_candidates(client):
    body = _body(client.get(SEARCH, params={"q": "Delhi"}))

    assert body["count"] == 1
    assert body["results"][0]["name"] == "New Delhi"
    assert body["results"][0]["timezone"] == "Asia/Kolkata"


def test_geocoding_results_are_cached(client, upstream):
    client.get(SEARCH, params={"q": "Delhi"})
    client.get(SEARCH, params={"q": "delhi"})  # matching is case-insensitive

    assert upstream["geocoding_calls"] == 1


def test_an_unmatched_place_name_is_a_404_not_a_crash(client, upstream):
    def empty(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"generationtime_ms": 0.1})

    upstream["handler"] = empty
    response = client.get(WEATHER, params={"location": "Atlantis"})

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "LOCATION_NOT_FOUND"


# --- Argument validation -----------------------------------------------------


def test_a_request_with_no_location_is_rejected(client):
    response = client.get(WEATHER)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_LOCATION_QUERY"


def test_half_a_coordinate_pair_is_rejected(client):
    response = client.get(WEATHER, params={"latitude": 28.61})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_LOCATION_QUERY"


def test_out_of_range_coordinates_are_rejected_by_the_schema(client):
    response = client.get(WEATHER, params={"latitude": 500, "longitude": 77.21})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


# --- Upstream failure --------------------------------------------------------


def test_an_upstream_outage_surfaces_as_a_service_error_with_a_request_id(
    client, upstream
):
    upstream["fail_with"] = 503

    response = client.get(WEATHER, params={"latitude": 1.0, "longitude": 1.0})

    assert response.status_code == 503
    error = response.json()["error"]
    assert error["code"] == "WEATHER_DATA_UNAVAILABLE"
    assert error["request_id"] == response.headers["X-Request-ID"]
    assert error["details"]["attempts"][0]["provider"] == "open-meteo"


def test_data_failing_validation_is_never_served_over_http(
    client, upstream, current_payload
):
    """An impossible temperature must not reach the client."""
    current_payload["current"]["temperature_2m"] = 812.0

    response = client.get(WEATHER, params={"latitude": 2.0, "longitude": 2.0})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "WEATHER_DATA_UNAVAILABLE"


def test_an_unknown_provider_is_a_client_error(client):
    response = client.get(
        WEATHER,
        params={"latitude": 28.61, "longitude": 77.21, "provider": "imd"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "PROVIDER_NOT_FOUND"


# --- Provider discovery ------------------------------------------------------


def test_providers_endpoint_describes_the_source_layer(client):
    body = _body(client.get("/api/v1/providers"))

    assert body["default_provider"] == "open-meteo"
    provider = body["providers"][0]
    assert provider["provider_id"] == "open-meteo"
    assert set(provider["capabilities"]) == {
        "current",
        "daily_forecast",
        "hourly_forecast",
    }
    assert provider["is_default"] is True


# --- Staleness ---------------------------------------------------------------


def test_a_stale_observation_is_still_served(client, current_payload):
    """Warnings are logged, not fatal — old data beats no data."""
    stale = datetime.now(timezone.utc) - timedelta(hours=10) + timedelta(seconds=19800)
    current_payload["current"]["time"] = stale.strftime("%Y-%m-%dT%H:%M")

    response = client.get(WEATHER, params={"latitude": 3.0, "longitude": 3.0})

    assert response.status_code == 200
