"""The provider layer: HTTP resilience, the registry, and readiness."""

from __future__ import annotations

import httpx
import pytest

from app.core.dependencies import (
    ReadinessRegistry,
    get_readiness_registry,
    register_provider_probes,
)
from app.core.exceptions import (
    ProviderNotFoundError,
    WeatherProviderError,
    WeatherProviderTimeoutError,
)
from app.domain.location import Coordinates
from app.providers.http import UpstreamHttpClient
from app.providers.registry import ProviderRegistry

DELHI = Coordinates(latitude=28.6, longitude=77.2)
URL = "https://example.test/forecast"


def _http(handler, *, max_retries: int = 2) -> UpstreamHttpClient:
    return UpstreamHttpClient(
        httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        provider_id="test-provider",
        max_retries=max_retries,
        backoff_seconds=0.0,
    )


# --- Retry policy ------------------------------------------------------------


async def test_a_transient_failure_is_retried_and_then_succeeds():
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503, json={"reason": "temporarily unavailable"})
        return httpx.Response(200, json={"ok": True})

    body = await _http(handler).get_json(URL)

    assert body == {"ok": True}
    assert attempts == 2


async def test_retries_are_bounded():
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(503, json={"reason": "down"})

    with pytest.raises(WeatherProviderError):
        await _http(handler, max_retries=2).get_json(URL)

    assert attempts == 3  # the first call plus two retries


async def test_a_client_error_is_not_retried():
    """A rejected request will be rejected identically next time."""
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(400, json={"reason": "latitude out of range"})

    with pytest.raises(WeatherProviderError) as exc_info:
        await _http(handler).get_json(URL)

    assert attempts == 1
    assert exc_info.value.details["upstream_reason"] == "latitude out of range"


async def test_a_timeout_is_reported_distinctly():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    with pytest.raises(WeatherProviderTimeoutError) as exc_info:
        await _http(handler, max_retries=1).get_json(URL)

    assert exc_info.value.status_code == 504
    assert exc_info.value.details["attempts"] == 2


async def test_an_unreadable_body_is_a_provider_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>not json</html>")

    with pytest.raises(WeatherProviderError):
        await _http(handler).get_json(URL)


async def test_a_json_array_where_an_object_was_expected_is_rejected():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[1, 2, 3])

    with pytest.raises(WeatherProviderError):
        await _http(handler).get_json(URL)


# --- Provider integration ----------------------------------------------------


async def test_the_provider_normalises_a_real_payload(make_provider, json_handler, current_payload):
    """Client, retry layer and normaliser composed, against a recorded payload."""
    provider = make_provider(json_handler(current_payload))

    report = await provider.fetch_current(DELHI)

    assert report.current.temperature_c == pytest.approx(31.4)
    assert report.current.wind_speed_ms == pytest.approx(5.0)
    assert report.provenance.provider_id == "open-meteo"
    assert report.provenance.attribution == "Weather data by Open-Meteo.com"


async def test_the_provider_requests_only_the_series_it_was_asked_for(
    make_provider, forecast_payload
):
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(request.url.params)
        return httpx.Response(200, json=forecast_payload)

    await make_provider(handler).fetch_forecast(
        DELHI, days=3, include_hourly=False, include_daily=True
    )

    assert "daily" in captured
    assert "hourly" not in captured
    assert captured["forecast_days"] == "3"


async def test_the_provider_caps_the_horizon_at_its_declared_maximum(
    make_provider, forecast_payload
):
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(request.url.params)
        return httpx.Response(200, json=forecast_payload)

    await make_provider(handler).fetch_forecast(
        DELHI, days=90, include_hourly=False, include_daily=True
    )

    assert captured["forecast_days"] == "16"


# --- Registry ----------------------------------------------------------------


def test_an_unknown_provider_lists_what_is_available():
    registry = ProviderRegistry()

    with pytest.raises(ProviderNotFoundError) as exc_info:
        registry.get("imd")

    assert exc_info.value.details["available"] == []


def test_registering_the_same_id_twice_replaces_rather_than_duplicates(make_provider, json_handler, current_payload):
    registry = ProviderRegistry()
    handler = json_handler(current_payload)

    registry.register(make_provider(handler))
    registry.register(make_provider(handler))

    assert len(registry) == 1
    assert "open-meteo" in registry


# --- Readiness ---------------------------------------------------------------


def test_readiness_passes_when_a_source_can_serve_current_conditions(
    make_provider, json_handler, current_payload
):
    providers = ProviderRegistry()
    providers.register(make_provider(json_handler(current_payload)))
    readiness = ReadinessRegistry()

    register_provider_probes(providers, readiness)

    assert readiness.names == ("weather-providers",)


async def test_readiness_fails_when_no_source_is_registered():
    """A misconfigured instance genuinely cannot serve weather."""
    readiness = ReadinessRegistry()
    register_provider_probes(ProviderRegistry(), readiness)

    statuses = await readiness.evaluate()

    assert [status.healthy for status in statuses] == [False]
    assert "No registered provider" in statuses[0].detail


async def test_the_readiness_probe_makes_no_upstream_call(current_payload):
    """Gating readiness on a third party would empty the fleet during their outage."""
    calls = 0

    def counting_handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=current_payload)

    from app.providers.open_meteo.client import OpenMeteoClient
    from app.providers.open_meteo.provider import OpenMeteoProvider

    providers = ProviderRegistry()
    providers.register(
        OpenMeteoProvider(OpenMeteoClient(_http(counting_handler), forecast_url=URL))
    )
    readiness = get_readiness_registry()
    register_provider_probes(providers, readiness)

    statuses = await readiness.evaluate()

    assert [status.healthy for status in statuses] == [True]
    assert calls == 0
