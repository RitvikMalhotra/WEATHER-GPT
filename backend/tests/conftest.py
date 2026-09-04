"""Shared test fixtures.

Every test in this suite runs offline. Upstream sources are driven through
``httpx.MockTransport``, which intercepts requests inside the client rather than
patching our own code — so the provider, the retry policy and the normaliser all
execute for real against a recorded payload.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import httpx
import pytest

from app.config.settings import Settings, get_settings
from app.core.dependencies import get_readiness_registry
from app.ingestion.validation import WeatherValidator
from app.providers.http import UpstreamHttpClient
from app.providers.open_meteo.client import OpenMeteoClient
from app.providers.open_meteo.provider import OpenMeteoProvider

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"


@pytest.fixture(scope="session")
def settings() -> Settings:
    return get_settings()


@pytest.fixture(autouse=True)
def _isolated_readiness_registry():
    """Keep readiness probes from leaking between tests."""
    registry = get_readiness_registry()
    registry.clear()
    yield registry
    registry.clear()


# --- Recorded upstream payloads ---------------------------------------------


def _iso_local(moment: datetime) -> str:
    """Render an instant the way Open-Meteo does with ``timezone=auto``."""
    return moment.strftime("%Y-%m-%dT%H:%M")


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 9, 4, 7, 30, tzinfo=timezone.utc)


@pytest.fixture
def current_payload(now: datetime) -> dict[str, Any]:
    """A realistic Open-Meteo current-conditions response for New Delhi.

    Wind is reported in km/h and temperature in °C — the units the upstream
    actually declares — so the normaliser has real conversion work to do.
    """
    local = now + timedelta(seconds=19800)  # Asia/Kolkata, UTC+05:30
    return {
        "latitude": 28.625,
        "longitude": 77.25,
        "elevation": 216.0,
        "generationtime_ms": 0.05,
        "utc_offset_seconds": 19800,
        "timezone": "Asia/Kolkata",
        "timezone_abbreviation": "IST",
        "current_units": {
            "time": "iso8601",
            "interval": "seconds",
            "temperature_2m": "°C",
            "relative_humidity_2m": "%",
            "apparent_temperature": "°C",
            "is_day": "",
            "precipitation": "mm",
            "weather_code": "wmo code",
            "cloud_cover": "%",
            "pressure_msl": "hPa",
            "surface_pressure": "hPa",
            "wind_speed_10m": "km/h",
            "wind_direction_10m": "°",
            "wind_gusts_10m": "km/h",
        },
        "current": {
            "time": _iso_local(local),
            "interval": 900,
            "temperature_2m": 31.4,
            "relative_humidity_2m": 62,
            "apparent_temperature": 35.2,
            "is_day": 1,
            "precipitation": 0.0,
            "weather_code": 2,
            "cloud_cover": 40,
            "pressure_msl": 1004.2,
            "surface_pressure": 987.6,
            "wind_speed_10m": 18.0,  # km/h -> 5.0 m/s
            "wind_direction_10m": 115,
            "wind_gusts_10m": 36.0,  # km/h -> 10.0 m/s
        },
    }


@pytest.fixture
def forecast_payload(now: datetime) -> dict[str, Any]:
    """A two-day daily plus three-hour hourly Open-Meteo forecast response."""
    local = now + timedelta(seconds=19800)
    hours = [_iso_local(local.replace(minute=0) + timedelta(hours=i)) for i in range(3)]
    return {
        "latitude": 28.625,
        "longitude": 77.25,
        "elevation": 216.0,
        "utc_offset_seconds": 19800,
        "timezone": "Asia/Kolkata",
        "timezone_abbreviation": "IST",
        "hourly_units": {
            "time": "iso8601",
            "temperature_2m": "°C",
            "relative_humidity_2m": "%",
            "dew_point_2m": "°C",
            "apparent_temperature": "°C",
            "precipitation_probability": "%",
            "precipitation": "mm",
            "weather_code": "wmo code",
            "pressure_msl": "hPa",
            "cloud_cover": "%",
            "visibility": "m",
            "wind_speed_10m": "km/h",
            "wind_direction_10m": "°",
            "wind_gusts_10m": "km/h",
            "uv_index": "",
            "is_day": "",
        },
        "hourly": {
            "time": hours,
            "temperature_2m": [31.4, 32.1, 32.8],
            "relative_humidity_2m": [62, 59, 55],
            "dew_point_2m": [23.1, 22.8, 22.4],
            "apparent_temperature": [35.2, 36.0, 36.6],
            "precipitation_probability": [10, 15, 25],
            "precipitation": [0.0, 0.0, 0.4],
            "weather_code": [2, 3, 80],
            "pressure_msl": [1004.2, 1003.8, 1003.1],
            "cloud_cover": [40, 65, 80],
            "visibility": [24000.0, 20000.0, 14000.0],
            "wind_speed_10m": [18.0, 21.6, 25.2],
            "wind_direction_10m": [115, 120, 128],
            "wind_gusts_10m": [36.0, 39.6, 46.8],
            "uv_index": [6.2, 7.1, 7.4],
            "is_day": [1, 1, 1],
        },
        "daily_units": {
            "time": "iso8601",
            "weather_code": "wmo code",
            "temperature_2m_max": "°C",
            "temperature_2m_min": "°C",
            "apparent_temperature_max": "°C",
            "apparent_temperature_min": "°C",
            "sunrise": "iso8601",
            "sunset": "iso8601",
            "uv_index_max": "",
            "precipitation_sum": "mm",
            "precipitation_hours": "h",
            "precipitation_probability_max": "%",
            "wind_speed_10m_max": "km/h",
            "wind_gusts_10m_max": "km/h",
            "wind_direction_10m_dominant": "°",
        },
        "daily": {
            "time": ["2026-09-04", "2026-09-05"],
            "weather_code": [80, 61],
            "temperature_2m_max": [34.8, 33.2],
            "temperature_2m_min": [26.1, 25.4],
            "apparent_temperature_max": [39.1, 37.5],
            "apparent_temperature_min": [28.0, 27.2],
            "sunrise": ["2026-09-04T06:02", "2026-09-05T06:02"],
            "sunset": ["2026-09-04T18:39", "2026-09-05T18:38"],
            "uv_index_max": [7.4, 6.9],
            "precipitation_sum": [4.2, 11.8],
            "precipitation_hours": [3.0, 6.0],
            "precipitation_probability_max": [45, 70],
            "wind_speed_10m_max": [25.2, 28.8],  # km/h -> 7.0, 8.0 m/s
            "wind_gusts_10m_max": [46.8, 54.0],
            "wind_direction_10m_dominant": [128, 135],
        },
    }


@pytest.fixture
def geocoding_payload() -> dict[str, Any]:
    return {
        "results": [
            {
                "id": 1261481,
                "name": "New Delhi",
                "latitude": 28.63576,
                "longitude": 77.22445,
                "elevation": 216.0,
                "country": "India",
                "country_code": "IN",
                "admin1": "Delhi",
                "timezone": "Asia/Kolkata",
                "population": 317797,
            }
        ]
    }


# --- Wiring helpers ----------------------------------------------------------

Handler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture
def make_client() -> Callable[[Handler], httpx.AsyncClient]:
    """Build an httpx client whose transport is a caller-supplied handler."""

    def factory(handler: Handler) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    return factory


@pytest.fixture
def make_provider(make_client) -> Callable[[Handler], OpenMeteoProvider]:
    """Build a real OpenMeteoProvider backed by a mocked transport."""

    def factory(handler: Handler, *, max_retries: int = 0) -> OpenMeteoProvider:
        http = UpstreamHttpClient(
            make_client(handler),
            provider_id="open-meteo",
            max_retries=max_retries,
            backoff_seconds=0.0,
        )
        return OpenMeteoProvider(OpenMeteoClient(http, forecast_url=FORECAST_URL))

    return factory


@pytest.fixture
def json_handler() -> Callable[[dict[str, Any]], Handler]:
    """A handler that always answers with the given JSON body."""

    def factory(body: dict[str, Any], status_code: int = 200) -> Handler:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status_code, json=body)

        return handler

    return factory


@pytest.fixture
def validator() -> WeatherValidator:
    return WeatherValidator()
