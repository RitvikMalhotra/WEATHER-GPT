"""IMD provider: parsing, mapping, station selection, routing and fallback.

Every test here is offline. Payloads follow the field names in IMD's published
API reference; no live IMD call is made, and none is needed to pin the
behaviour that matters — which source answers, and what the answer claims about
where it came from.
"""

from __future__ import annotations

from datetime import timedelta

import httpx
import pytest

from app.core.exceptions import WeatherProviderError
from app.domain.location import Coordinates
from app.domain.weather import WeatherCondition
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.validation import WeatherValidator
from app.providers.base import ProviderCapability
from app.providers.gfs.provider import GFS_PROVIDER_ID, GfsProvider
from app.providers.http import UpstreamHttpClient
from app.providers.imd.client import ImdClient
from app.providers.imd.provider import IMD_PROVIDER_ID, ImdProvider
from app.providers.open_meteo.client import PROVIDER_ID as OPEN_METEO_ID
from app.providers.open_meteo.client import OpenMeteoClient
from app.providers.open_meteo.provider import OpenMeteoProvider
from app.providers.registry import ProviderRegistry

HYDERABAD = Coordinates(latitude=17.385, longitude=78.4867)
LONDON = Coordinates(latitude=51.5072, longitude=-0.1276)

IMD_BASE = "https://api.imd.gov.in/api/v1"

STATIONS = [
    {"Station_Code": "42182", "Station_Name": "Hyderabad", "Latitude": 17.45, "Longitude": 78.47},
    {"Station_Code": "42181", "Station_Name": "New Delhi", "Latitude": 28.58, "Longitude": 77.20},
    {"Station_Code": "43003", "Station_Name": "Mumbai", "Latitude": 19.11, "Longitude": 72.85},
]

CURRENT_WX = {
    "Station Id": "42182",
    "Station": "Hyderabad",
    "Date of Observation": "2026-09-04",
    "Time of Observation": "1200",
    "M.S.L.P": "1008.4",
    "Wind Direction": "230",
    "Wind Speed": "18",          # km/h -> 5 m/s
    "Temperature": "29.6",
    "Weather Code": "02",
    "Nebulosity": "4",           # oktas -> 50%
    "Humidity": "62",
    "Last 24 hrs Rainfall": "3.2",
    "Weather Description": "Generally cloudy sky with light rain",
}

CITY_FORECAST = {
    "Date": "2026-09-04",
    "Station_Code": "42182",
    "Station_Name": "Hyderabad",
    "Latitude": 17.45,
    "Longitude": 78.47,
    "Todays_Forecast_Max_Temp": "31.0",
    "Todays_Forecast_Min_temp": "22.4",
    "Todays_Forecast": "Generally cloudy sky with light rain",
    "Day_2_Max_Temp": "30.5",
    "Day_2_Min_temp": "22.0",
    "Day_2_Forecast": "Thunderstorm with lightning",
    "Day_3_Max_Temp": "31.2",
    "Day_3_Min_temp": "22.6",
    "Day_3_Forecast": "Partly cloudy sky",
}


def _imd(handler, **kwargs) -> ImdProvider:
    http = UpstreamHttpClient(
        httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        provider_id=IMD_PROVIDER_ID,
        max_retries=0,
    )
    return ImdProvider(
        ImdClient(http, base_url=IMD_BASE, api_key="test-key"), **kwargs
    )


def imd_handler(*, current=CURRENT_WX, forecast=CITY_FORECAST, stations=STATIONS):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/cityforecast_mapping"):
            return httpx.Response(200, json=stations)
        if path.endswith("/current_wx"):
            return httpx.Response(200, json=[current])
        if path.endswith("/cityforecast"):
            return httpx.Response(200, json=[forecast])
        raise AssertionError(f"unexpected IMD path {path}")

    return handler


# --- Parsing and canonical mapping ------------------------------------------


@pytest.mark.asyncio
async def test_imd_observation_maps_onto_the_canonical_model_in_si_units():
    provider = _imd(imd_handler())
    report = await provider.fetch_current(HYDERABAD)

    current = report.current
    assert current.temperature_c == 29.6
    assert current.wind_speed_ms == 5.0          # 18 km/h, converted not assumed
    assert current.relative_humidity_pct == 62.0
    assert current.pressure_msl_hpa == 1008.4
    assert current.precipitation_mm == 3.2
    assert current.cloud_cover_pct == 50.0        # 4 oktas of 8
    assert current.observed_at.isoformat() == "2026-09-04T12:00:00+00:00"
    assert current.condition is WeatherCondition.RAIN
    assert current.condition_description == "Generally cloudy sky with light rain"


@pytest.mark.asyncio
async def test_an_imd_weather_code_is_never_read_as_a_wmo_code():
    # IMD publishes its own 01-99 list with different meanings. Code "02" is
    # not WMO 2; storing it as one would put a wrong condition on real data.
    provider = _imd(imd_handler())
    report = await provider.fetch_current(HYDERABAD)
    assert report.current.wmo_code is None


@pytest.mark.asyncio
async def test_imd_city_forecast_becomes_dated_daily_points():
    provider = _imd(imd_handler())
    forecast = await provider.fetch_forecast(
        HYDERABAD, days=3, include_hourly=False, include_daily=True
    )

    assert [point.date.isoformat() for point in forecast.daily] == [
        "2026-09-04", "2026-09-05", "2026-09-06"
    ]
    assert forecast.daily[0].temperature_max_c == 31.0
    assert forecast.daily[1].condition is WeatherCondition.THUNDERSTORM


@pytest.mark.asyncio
async def test_missing_imd_values_become_none_rather_than_zero():
    sparse = {**CURRENT_WX, "Humidity": "NA", "Last 24 hrs Rainfall": "", "M.S.L.P": "-"}
    provider = _imd(imd_handler(current=sparse))
    report = await provider.fetch_current(HYDERABAD)

    assert report.current.relative_humidity_pct is None
    assert report.current.precipitation_mm is None
    assert report.current.pressure_msl_hpa is None
    assert report.current.temperature_c == 29.6


# --- Provenance --------------------------------------------------------------


@pytest.mark.asyncio
async def test_imd_provenance_names_imd_and_the_station_that_answered():
    provider = _imd(imd_handler())
    report = await provider.fetch_current(HYDERABAD)

    provenance = report.provenance
    assert provenance.provider_id == "imd"
    assert provenance.provider_name == "India Meteorological Department"
    assert "42182" in (provenance.model or "")
    assert "IMD" in (provenance.attribution or "")
    # A station-based reading is only as local as its station.
    assert "km from the requested point" in (provenance.attribution or "")


# --- Station mapping ---------------------------------------------------------


@pytest.mark.asyncio
async def test_a_location_maps_to_its_nearest_published_station():
    provider = _imd(imd_handler())
    report = await provider.fetch_current(Coordinates(latitude=19.07, longitude=72.87))
    assert "43003" in (report.provenance.model or "")  # Mumbai, not Hyderabad


@pytest.mark.asyncio
async def test_a_point_with_no_nearby_station_is_refused_rather_than_mismapped():
    # Deep in the Thar desert: inside India, far from every station listed.
    remote = Coordinates(latitude=27.0, longitude=71.0)
    provider = _imd(imd_handler())

    with pytest.raises(WeatherProviderError) as excinfo:
        await provider.fetch_current(remote)
    assert "close enough" in str(excinfo.value)


@pytest.mark.asyncio
async def test_the_station_catalogue_is_fetched_once_and_reused():
    calls: list[str] = []
    base = imd_handler()

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return base(request)

    provider = _imd(handler, catalogue_ttl=timedelta(hours=12))
    await provider.fetch_current(HYDERABAD)
    await provider.fetch_current(HYDERABAD)

    assert calls.count("/api/v1/cityforecast_mapping") == 1


@pytest.mark.asyncio
async def test_an_hourly_request_is_declined_because_imd_publishes_daily_only():
    provider = _imd(imd_handler())
    with pytest.raises(WeatherProviderError):
        await provider.fetch_forecast(
            HYDERABAD, days=3, include_hourly=True, include_daily=False
        )


@pytest.mark.asyncio
async def test_a_missing_key_is_reported_as_a_provider_failure_not_as_data():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "API key missing"})

    provider = _imd(handler)
    with pytest.raises(WeatherProviderError):
        await provider.fetch_current(HYDERABAD)


# --- Routing and fallback ----------------------------------------------------


def _registry(*, with_imd=True, imd_handler_fn=None) -> ProviderRegistry:
    registry = ProviderRegistry(default_provider_id=OPEN_METEO_ID)

    def upstream(provider_id: str, handler) -> UpstreamHttpClient:
        return UpstreamHttpClient(
            httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            provider_id=provider_id,
            max_retries=0,
        )

    def open_meteo_handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("open-meteo should not be called in this test")

    registry.register(
        OpenMeteoProvider(
            OpenMeteoClient(
                upstream(OPEN_METEO_ID, open_meteo_handler),
                forecast_url="https://api.open-meteo.com/v1/forecast",
            )
        )
    )
    registry.register(
        GfsProvider(
            OpenMeteoClient(
                upstream(GFS_PROVIDER_ID, open_meteo_handler),
                forecast_url="https://api.open-meteo.com/v1/gfs",
            )
        )
    )
    if with_imd:
        http = UpstreamHttpClient(
            httpx.AsyncClient(transport=httpx.MockTransport(imd_handler_fn or imd_handler())),
            provider_id=IMD_PROVIDER_ID,
            max_retries=0,
        )
        registry.register(ImdProvider(ImdClient(http, base_url=IMD_BASE, api_key="k")))
    return registry


def test_an_indian_point_puts_imd_first_and_keeps_the_global_sources_behind_it():
    order = [
        provider.metadata.provider_id
        for provider in _registry().for_capability(
            ProviderCapability.CURRENT,
            latitude=HYDERABAD.latitude,
            longitude=HYDERABAD.longitude,
        )
    ]
    assert order[0] == "imd"
    assert "open-meteo" in order  # fallback remains available


def test_a_point_outside_india_drops_imd_entirely():
    order = [
        provider.metadata.provider_id
        for provider in _registry().for_capability(
            ProviderCapability.CURRENT,
            latitude=LONDON.latitude,
            longitude=LONDON.longitude,
        )
    ]
    assert "imd" not in order
    assert order[0] == "open-meteo"


def test_without_coordinates_ordering_is_unchanged_from_before_imd_existed():
    order = [
        provider.metadata.provider_id
        for provider in _registry().for_capability(ProviderCapability.CURRENT)
    ]
    assert order[0] == "open-meteo"


@pytest.mark.asyncio
async def test_when_imd_fails_the_answer_comes_from_open_meteo_and_says_so(
    current_payload, make_client
):
    """The fallback rule that matters: never IMD's name on another source's data."""

    def failing_imd(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "upstream down"})

    registry = ProviderRegistry(default_provider_id=OPEN_METEO_ID)
    registry.register(
        OpenMeteoProvider(
            OpenMeteoClient(
                UpstreamHttpClient(
                    make_client(lambda request: httpx.Response(200, json=current_payload)),
                    provider_id=OPEN_METEO_ID,
                    max_retries=0,
                ),
                forecast_url="https://api.open-meteo.com/v1/forecast",
            )
        )
    )
    registry.register(
        ImdProvider(
            ImdClient(
                UpstreamHttpClient(
                    make_client(failing_imd), provider_id=IMD_PROVIDER_ID, max_retries=0
                ),
                base_url=IMD_BASE,
                api_key="k",
            )
        )
    )

    pipeline = IngestionPipeline(registry, WeatherValidator())
    report = await pipeline.current(HYDERABAD)

    assert report.provenance.provider_id == "open-meteo"
    assert "imd" not in (report.provenance.provider_name or "").casefold()
    assert "IMD" not in (report.provenance.attribution or "")


@pytest.mark.asyncio
async def test_an_explicitly_requested_source_is_never_substituted():
    """Explicit GFS that fails must fail, not quietly become another source."""

    def failing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "down"})

    registry = ProviderRegistry(default_provider_id=OPEN_METEO_ID)
    registry.register(
        OpenMeteoProvider(
            OpenMeteoClient(
                UpstreamHttpClient(
                    httpx.AsyncClient(transport=httpx.MockTransport(failing)),
                    provider_id=OPEN_METEO_ID,
                    max_retries=0,
                ),
                forecast_url="https://api.open-meteo.com/v1/forecast",
            )
        )
    )
    registry.register(
        GfsProvider(
            OpenMeteoClient(
                UpstreamHttpClient(
                    httpx.AsyncClient(transport=httpx.MockTransport(failing)),
                    provider_id=GFS_PROVIDER_ID,
                    max_retries=0,
                ),
                forecast_url="https://api.open-meteo.com/v1/gfs",
            )
        )
    )

    pipeline = IngestionPipeline(registry, WeatherValidator())
    with pytest.raises(Exception) as excinfo:
        await pipeline.current(HYDERABAD, provider_id=GFS_PROVIDER_ID)
    # It failed rather than returning someone else's data under GFS's name.
    assert "open-meteo" not in str(excinfo.value).casefold()


@pytest.mark.asyncio
async def test_imd_data_still_passes_the_shared_validation_gate():
    impossible = {**CURRENT_WX, "Temperature": "451"}
    registry = ProviderRegistry(default_provider_id=IMD_PROVIDER_ID)
    http = UpstreamHttpClient(
        httpx.AsyncClient(transport=httpx.MockTransport(imd_handler(current=impossible))),
        provider_id=IMD_PROVIDER_ID,
        max_retries=0,
    )
    registry.register(ImdProvider(ImdClient(http, base_url=IMD_BASE, api_key="k")))

    pipeline = IngestionPipeline(registry, WeatherValidator())
    with pytest.raises(Exception):
        # No source left after the gate rejects it: better than serving 451 °C.
        await pipeline.current(HYDERABAD)
