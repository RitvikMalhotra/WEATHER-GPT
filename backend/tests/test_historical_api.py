"""The historical endpoint's HTTP contract.

Runs offline against a stub history service, so these tests cover argument
parsing, response shape and error behaviour without a database. The SQL behind
the service is covered by ``test_db_integration.py``.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app.config.settings import Settings
from app.core.dependencies import get_history_service
from app.core.exceptions import DatabaseUnavailableError
from app.domain.provenance import DataProvenance
from app.domain.weather import CurrentWeather, WeatherCondition
from app.main import create_app
from app.services.history import HistoricalRecord, HistoryQuery

HISTORICAL = "/api/v1/weather/historical"
HYDERABAD = {"latitude": 17.385, "longitude": 78.4867}


def _record(*, observed_at: datetime, distance_m: float = 0.0) -> HistoricalRecord:
    return HistoricalRecord(
        weather=CurrentWeather(
            observed_at=observed_at,
            temperature_c=28.4,
            relative_humidity_pct=81.0,
            wind_speed_ms=3.2,
            condition=WeatherCondition.RAIN,
            condition_description="Moderate rain",
            wmo_code=63,
        ),
        provenance=DataProvenance(
            provider_id="open-meteo",
            provider_name="Open-Meteo",
            model="best_match",
            fetched_at=observed_at,
            license="CC-BY-4.0",
            attribution="Weather data by Open-Meteo.com",
        ),
        latitude=17.385,
        longitude=78.4867,
        distance_m=distance_m,
    )


class _StubHistory:
    """Captures the query it was given and returns a scripted result."""

    def __init__(self, records=None, error: Exception | None = None) -> None:
        self._records = records or []
        self._error = error
        self.queries: list[HistoryQuery] = []

    @property
    def enabled(self) -> bool:
        return self._error is None

    async def observations(self, query: HistoryQuery):
        self.queries.append(query)
        if self._error:
            raise self._error
        return self._records


@pytest.fixture
def history() -> _StubHistory:
    return _StubHistory(
        records=[
            _record(observed_at=datetime(2026, 8, 15, 6, 0, tzinfo=timezone.utc)),
            _record(
                observed_at=datetime(2026, 8, 15, 7, 0, tzinfo=timezone.utc),
                distance_m=1420.5,
            ),
        ]
    )


@pytest.fixture
def client(settings: Settings, history: _StubHistory):
    app = create_app(settings)
    app.dependency_overrides[get_history_service] = lambda: history
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _get(client, **params):
    return client.get(HISTORICAL, params={**HYDERABAD, **params})


# --- Happy path ---------------------------------------------------------------


def test_a_valid_range_returns_records(client):
    response = _get(client, start="2026-08-01", end="2026-08-31")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["count"] == 2
    assert body["requested"] == HYDERABAD
    assert len(body["observations"]) == 2


def test_records_carry_canonical_weather_and_provenance(client):
    body = _get(client, start="2026-08-01", end="2026-08-31").json()
    first = body["observations"][0]

    assert first["weather"]["temperature_c"] == pytest.approx(28.4)
    assert first["weather"]["condition"] == "rain"
    assert first["provenance"]["provider_id"] == "open-meteo"
    assert first["provenance"]["attribution"] == "Weather data by Open-Meteo.com"
    assert first["provenance"]["cached"] is False


def test_the_response_reports_the_window_it_actually_searched(client):
    body = _get(client, start="2026-08-01", end="2026-08-31").json()

    assert body["range"]["start"].startswith("2026-08-01T00:00:00")
    # A bare end date covers the whole day rather than stopping at midnight.
    assert body["range"]["end"].startswith("2026-08-31T23:59:59")


def test_distance_is_reported_in_kilometres(client):
    body = _get(client, start="2026-08-01", end="2026-08-31").json()

    assert body["observations"][0]["distance_km"] == pytest.approx(0.0)
    assert body["observations"][1]["distance_km"] == pytest.approx(1.421)


def test_no_database_row_internals_are_exposed(client):
    """Response schemas, not ORM rows: no ids, no column-name leakage."""
    first = _get(client, start="2026-08-01", end="2026-08-31").json()["observations"][0]

    assert set(first) == {
        "latitude",
        "longitude",
        "distance_km",
        "weather",
        "provenance",
    }
    assert "id" not in first
    assert "geom" not in first
    assert "location_key" not in first


def test_timestamps_are_accepted_as_well_as_dates(client, history):
    response = _get(client, start="2026-08-01T06:00:00Z", end="2026-08-01T18:00:00Z")

    assert response.status_code == 200
    query = history.queries[-1]
    assert query.start == datetime(2026, 8, 1, 6, 0, tzinfo=timezone.utc)
    assert query.end == datetime(2026, 8, 1, 18, 0, tzinfo=timezone.utc)


def test_a_naive_timestamp_is_read_as_utc(client, history):
    _get(client, start="2026-08-01T06:00:00", end="2026-08-01T18:00:00")

    assert history.queries[-1].start.tzinfo is not None


# --- Search radius ------------------------------------------------------------


def test_the_default_radius_is_applied(client, history, settings):
    _get(client, start="2026-08-01", end="2026-08-02")

    assert history.queries[-1].radius_m == pytest.approx(
        settings.HISTORY_DEFAULT_RADIUS_KM * 1000
    )


def test_a_requested_radius_is_honoured(client, history):
    body = _get(client, start="2026-08-01", end="2026-08-02", radius_km=5).json()

    assert history.queries[-1].radius_m == pytest.approx(5000.0)
    assert body["search_radius_km"] == pytest.approx(5.0)


def test_the_radius_is_capped(client, history, settings):
    _get(client, start="2026-08-01", end="2026-08-02", radius_km=99_999)

    assert history.queries[-1].radius_m == pytest.approx(
        settings.HISTORY_MAX_RADIUS_KM * 1000
    )


def test_a_provider_filter_is_passed_through(client, history):
    _get(client, start="2026-08-01", end="2026-08-02", provider="open-meteo")

    assert history.queries[-1].provider_id == "open-meteo"


# --- Empty results ------------------------------------------------------------


def test_an_empty_result_is_a_200_not_an_error(settings):
    """'Nothing recorded there' is a valid answer to a historical query."""
    app = create_app(settings)
    app.dependency_overrides[get_history_service] = lambda: _StubHistory(records=[])

    with TestClient(app) as client:
        response = _get(client, start="2026-08-01", end="2026-08-31")

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 0
    assert body["observations"] == []
    assert body["truncated"] is False


# --- Argument validation ------------------------------------------------------


def test_an_inverted_range_is_rejected(client):
    response = _get(client, start="2026-08-31", end="2026-08-01")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_TIME_RANGE"


def test_an_unparseable_date_is_rejected(client):
    response = _get(client, start="last tuesday", end="2026-08-31")

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "INVALID_TIME_RANGE"
    assert error["details"]["field"] == "start"


def test_an_excessive_range_is_rejected(client, settings):
    response = _get(client, start="2000-01-01", end="2026-08-31")

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "INVALID_TIME_RANGE"
    assert error["details"]["max_days"] == settings.HISTORY_MAX_RANGE_DAYS


@pytest.mark.parametrize(
    "params",
    [
        {"latitude": 500, "longitude": 78.4867},
        {"latitude": 17.385, "longitude": -400},
    ],
)
def test_invalid_coordinates_are_rejected(client, params):
    response = client.get(
        HISTORICAL, params={**params, "start": "2026-08-01", "end": "2026-08-31"}
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_missing_coordinates_are_rejected(client):
    response = client.get(HISTORICAL, params={"start": "2026-08-01", "end": "2026-08-31"})

    assert response.status_code == 422


def test_a_zero_or_negative_radius_is_rejected(client):
    response = _get(client, start="2026-08-01", end="2026-08-02", radius_km=0)

    assert response.status_code == 422


# --- Database failure ---------------------------------------------------------


def test_a_database_failure_is_a_structured_error(settings):
    app = create_app(settings)
    app.dependency_overrides[get_history_service] = lambda: _StubHistory(
        error=DatabaseUnavailableError(details={"operation": "history.observations"})
    )

    with TestClient(app) as client:
        response = _get(client, start="2026-08-01", end="2026-08-31")

    assert response.status_code == 503
    error = response.json()["error"]
    assert error["code"] == "DATABASE_UNAVAILABLE"
    assert error["message"] == "Weather persistence is temporarily unavailable."
    assert error["request_id"] == response.headers["X-Request-ID"]


def test_a_database_failure_leaks_nothing_internal(settings):
    """No DSN, no SQL, no traceback in the response body."""
    app = create_app(settings)
    app.dependency_overrides[get_history_service] = lambda: _StubHistory(
        error=DatabaseUnavailableError(details={"operation": "history.observations"})
    )

    with TestClient(app) as client:
        body = _get(client, start="2026-08-01", end="2026-08-31").text

    lowered = body.lower()
    for leak in ("postgres", "asyncpg", "password", "select", "traceback", "sqlalchemy"):
        assert leak not in lowered, f"response leaked {leak!r}: {body}"


def test_history_without_a_database_falls_through_to_a_provider_archive(settings):
    """A deployment that stored nothing still answers questions about the past.

    "0 observations" is a true statement about our storage and a useless answer
    to "how much rain fell yesterday?", so a window the database cannot cover is
    served from a provider archive — normalised and stamped with provenance
    exactly like a live response.
    """
    app = create_app(settings)  # settings carry no DATABASE_URL in the test env

    with TestClient(app) as client:
        response = _get(client, start="2026-08-01", end="2026-08-31")

    assert response.status_code == 200
    body = response.json()
    # Whatever the stub upstream returns, every record names where it came from
    # and nothing was invented to fill the window.
    for observation in body["observations"]:
        assert observation["provenance"]["provider_id"]
        assert observation["provenance"]["fetched_at"]


def test_history_with_no_source_at_all_is_a_clean_503(settings):
    """No database and no archive-capable provider is an honest 503."""
    app = create_app(settings)

    with TestClient(app) as client:
        app.state.container.history._archive = None  # noqa: SLF001 - no source at all
        response = _get(client, start="2026-08-01", end="2026-08-31")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "DATABASE_UNAVAILABLE"


# --- API surface --------------------------------------------------------------


def test_the_endpoint_is_documented(client):
    schema = client.get("/openapi.json").json()

    operation = schema["paths"]["/api/v1/weather/historical"]["get"]
    assert operation["summary"]
    assert "200" in operation["responses"]
    assert "503" in operation["responses"]


def test_phase_two_endpoints_are_untouched(client):
    """Persistence must not have changed the existing contract."""
    schema = client.get("/openapi.json").json()

    for path in (
        "/api/v1/weather/current",
        "/api/v1/forecast",
        "/api/v1/locations/search",
        "/api/v1/providers",
        "/api/v1/health",
    ):
        assert path in schema["paths"], f"{path} disappeared"
