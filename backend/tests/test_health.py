"""Tests for the system status endpoints and the API contract around them.

These run entirely in-process: no PostgreSQL, no Redis, no weather provider and
no network access.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config.settings import Settings, get_settings
from app.core.dependencies import ComponentStatus, get_readiness_registry
from app.main import create_app


@pytest.fixture(scope="module")
def settings() -> Settings:
    return get_settings()


@pytest.fixture(scope="module")
def client(settings: Settings):
    """App instance with the lifespan executed, as in production."""
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture
def registry():
    """Readiness registry, emptied before and after each test that uses it."""
    registry = get_readiness_registry()
    registry.clear()
    yield registry
    registry.clear()


@pytest.fixture
def health_url(settings: Settings) -> str:
    return f"{settings.API_V1_PREFIX}/health"


# --- GET /api/v1/health -----------------------------------------------------


def test_health_returns_documented_payload(client, health_url, settings):
    response = client.get(health_url)

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "service": settings.SERVICE_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT.value,
    }


def test_health_reports_configured_identity(client, health_url):
    body = client.get(health_url).json()

    assert body["service"] == "weathergpt-backend"
    assert body["version"] == "1.0.0"
    assert body["environment"] in {"development", "staging", "production"}


# --- GET /api/v1/health/live ------------------------------------------------


def test_liveness_returns_alive(client, health_url):
    response = client.get(f"{health_url}/live")

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


# --- GET /api/v1/health/ready -----------------------------------------------


def test_readiness_returns_ready_with_no_registered_checks(
    client, health_url, registry
):
    response = client.get(f"{health_url}/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_readiness_reports_ready_when_every_check_passes(client, health_url, registry):
    registry.register("datastore", lambda: ComponentStatus("datastore", healthy=True))

    response = client.get(f"{health_url}/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_readiness_returns_503_when_a_check_fails(client, health_url, registry):
    registry.register(
        "datastore",
        lambda: ComponentStatus("datastore", healthy=False, detail="connection refused"),
    )

    response = client.get(f"{health_url}/ready")

    assert response.status_code == 503
    error = response.json()["error"]
    assert error["code"] == "SERVICE_UNAVAILABLE"
    assert error["details"]["components"] == [
        {"name": "datastore", "detail": "connection refused"}
    ]


def test_readiness_treats_a_raising_check_as_unhealthy(client, health_url, registry):
    def exploding_check() -> ComponentStatus:
        raise RuntimeError("probe blew up")

    registry.register("provider", exploding_check)

    response = client.get(f"{health_url}/ready")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "SERVICE_UNAVAILABLE"


# --- Request correlation ----------------------------------------------------


def test_every_response_carries_a_request_id(client, health_url):
    response = client.get(f"{health_url}/live")

    assert response.headers["X-Request-ID"]


def test_inbound_request_id_is_preserved(client, health_url):
    response = client.get(
        f"{health_url}/live", headers={"X-Request-ID": "trace-me-please"}
    )

    assert response.headers["X-Request-ID"] == "trace-me-please"


# --- Error contract ---------------------------------------------------------


def test_unknown_route_uses_the_error_envelope(client, settings):
    response = client.get(f"{settings.API_V1_PREFIX}/does-not-exist")

    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "NOT_FOUND"
    assert error["request_id"] == response.headers["X-Request-ID"]


# --- API surface ------------------------------------------------------------


def test_endpoints_are_versioned_under_the_v1_prefix(client, settings):
    assert settings.API_V1_PREFIX == "/api/v1"
    assert client.get("/health").status_code == 404


def test_openapi_schema_documents_the_health_endpoints(client, settings, health_url):
    schema = client.get("/openapi.json").json()

    assert schema["info"]["title"] == "WeatherGPT API"
    assert schema["info"]["version"] == settings.APP_VERSION

    for path in (health_url, f"{health_url}/live", f"{health_url}/ready"):
        assert path in schema["paths"], f"{path} missing from OpenAPI schema"
        operation = schema["paths"][path]["get"]
        assert operation["summary"]
        assert operation["description"]
        assert "200" in operation["responses"]

    assert "503" in schema["paths"][f"{health_url}/ready"]["get"]["responses"]


def test_interactive_docs_are_served(client):
    assert client.get("/docs").status_code == 200
