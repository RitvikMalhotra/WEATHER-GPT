"""The ingestion pipeline: provider selection, the validation gate and fallback.

Drives the pipeline with stub providers so the behaviour under test is the
pipeline's own policy, not any particular upstream.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from app.core.exceptions import (
    ForecastUnavailableError,
    ProviderNotFoundError,
    WeatherDataUnavailableError,
    WeatherProviderError,
)
from app.domain.forecast import DailyForecastPoint, Forecast
from app.domain.location import Coordinates, Location
from app.domain.provenance import DataProvenance
from app.domain.weather import CurrentWeather, WeatherReport
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.validation import WeatherValidator
from app.providers.base import ProviderCapability, ProviderMetadata, WeatherProvider
from app.providers.registry import ProviderRegistry

NOW = datetime(2026, 9, 4, 7, 30, tzinfo=timezone.utc)
DELHI = Coordinates(latitude=28.6, longitude=77.2)

ALL_CAPABILITIES = frozenset(
    {
        ProviderCapability.CURRENT,
        ProviderCapability.HOURLY_FORECAST,
        ProviderCapability.DAILY_FORECAST,
    }
)


def _report(provider_id: str, *, temperature: float = 21.0) -> WeatherReport:
    return WeatherReport(
        location=Location(coordinates=DELHI),
        current=CurrentWeather(observed_at=NOW, temperature_c=temperature),
        provenance=DataProvenance(
            provider_id=provider_id, provider_name=provider_id, fetched_at=NOW
        ),
    )


class StubProvider(WeatherProvider):
    """A provider that returns, or fails, exactly as a test instructs."""

    def __init__(
        self,
        provider_id: str,
        *,
        priority: int = 100,
        report: WeatherReport | None = None,
        error: Exception | None = None,
        capabilities: frozenset[ProviderCapability] = ALL_CAPABILITIES,
    ) -> None:
        self._metadata = ProviderMetadata(
            provider_id=provider_id,
            name=provider_id,
            capabilities=capabilities,
            priority=priority,
        )
        self._report = report if report is not None else _report(provider_id)
        self._error = error
        self.calls = 0

    @property
    def metadata(self) -> ProviderMetadata:
        return self._metadata

    async def fetch_current(self, coordinates: Coordinates) -> WeatherReport:
        self.calls += 1
        if self._error:
            raise self._error
        return self._report

    async def fetch_forecast(self, coordinates, *, days, include_hourly, include_daily):
        self.calls += 1
        if self._error:
            raise self._error
        return Forecast(
            location=Location(coordinates=DELHI),
            daily=[
                DailyForecastPoint(
                    date=date(2026, 9, 4),
                    temperature_min_c=26.0,
                    temperature_max_c=35.0,
                )
            ],
            provenance=self._report.provenance,
        )


def _pipeline(*providers: WeatherProvider, default: str | None = None) -> IngestionPipeline:
    registry = ProviderRegistry(default_provider_id=default)
    for provider in providers:
        registry.register(provider)
    return IngestionPipeline(registry, WeatherValidator())


# --- Happy path --------------------------------------------------------------


async def test_the_first_healthy_provider_serves_the_request():
    primary = StubProvider("primary", priority=10)
    secondary = StubProvider("secondary", priority=20)

    report = await _pipeline(primary, secondary).current(DELHI)

    assert report.provenance.provider_id == "primary"
    assert secondary.calls == 0


async def test_providers_are_tried_in_priority_order():
    low = StubProvider("low-priority", priority=90)
    high = StubProvider("high-priority", priority=5)

    report = await _pipeline(low, high).current(DELHI)

    assert report.provenance.provider_id == "high-priority"


async def test_the_configured_default_leads_the_chain_regardless_of_priority():
    fastest = StubProvider("fastest", priority=1)
    preferred = StubProvider("preferred", priority=50)

    report = await _pipeline(fastest, preferred, default="preferred").current(DELHI)

    assert report.provenance.provider_id == "preferred"


# --- Fallback ----------------------------------------------------------------


async def test_a_failing_provider_falls_through_to_the_next():
    broken = StubProvider(
        "broken", priority=10, error=WeatherProviderError("upstream down")
    )
    healthy = StubProvider("healthy", priority=20)

    report = await _pipeline(broken, healthy).current(DELHI)

    assert report.provenance.provider_id == "healthy"
    assert broken.calls == 1


async def test_data_that_fails_validation_is_never_served():
    """The point of the gate: a plausible-looking source with impossible data."""
    corrupt = StubProvider(
        "corrupt", priority=10, report=_report("corrupt", temperature=812.0)
    )
    healthy = StubProvider("healthy", priority=20)

    report = await _pipeline(corrupt, healthy).current(DELHI)

    assert report.provenance.provider_id == "healthy"
    assert report.current.temperature_c == pytest.approx(21.0)


async def test_exhausting_every_source_raises_rather_than_guessing():
    corrupt = StubProvider(
        "corrupt", priority=10, report=_report("corrupt", temperature=812.0)
    )
    broken = StubProvider("broken", priority=20, error=WeatherProviderError("down"))

    with pytest.raises(WeatherDataUnavailableError) as exc_info:
        await _pipeline(corrupt, broken).current(DELHI)

    attempts = exc_info.value.details["attempts"]
    assert [attempt["provider"] for attempt in attempts] == ["corrupt", "broken"]
    assert attempts[0]["reason"] == "WEATHER_DATA_VALIDATION_FAILED"
    assert attempts[1]["reason"] == "WEATHER_PROVIDER_ERROR"


async def test_an_empty_registry_fails_cleanly():
    with pytest.raises(WeatherDataUnavailableError):
        await _pipeline().current(DELHI)


# --- Explicit provider selection ---------------------------------------------


async def test_an_explicitly_requested_provider_is_never_substituted():
    """Silently serving another source would make the response provenance a lie."""
    requested = StubProvider("requested", error=WeatherProviderError("down"))
    healthy = StubProvider("healthy")

    with pytest.raises(WeatherDataUnavailableError):
        await _pipeline(requested, healthy).current(DELHI, provider_id="requested")

    assert healthy.calls == 0


async def test_requesting_an_unknown_provider_is_a_client_error():
    with pytest.raises(ProviderNotFoundError) as exc_info:
        await _pipeline(StubProvider("known")).current(DELHI, provider_id="nope")

    assert exc_info.value.status_code == 400
    assert exc_info.value.details["available"] == ["known"]


# --- Capability routing ------------------------------------------------------


async def test_only_capable_providers_are_considered():
    current_only = StubProvider(
        "current-only",
        priority=1,
        capabilities=frozenset({ProviderCapability.CURRENT}),
    )
    forecaster = StubProvider("forecaster", priority=50)

    forecast = await _pipeline(current_only, forecaster).forecast(DELHI, days=3)

    assert forecast.provenance.provider_id == "forecaster"
    assert current_only.calls == 0


async def test_a_forecast_with_no_capable_provider_raises():
    current_only = StubProvider(
        "current-only", capabilities=frozenset({ProviderCapability.CURRENT})
    )

    with pytest.raises(ForecastUnavailableError):
        await _pipeline(current_only).forecast(DELHI, days=3)
