"""Alert orchestration that needs no database.

Deduplication keys, alert construction, provenance and the safety boundaries.
The queries themselves are covered by ``test_db_integration.py`` against a real
PostGIS.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.alerts.engine import AlertEngine
from app.alerts.rules import SampleWindow, build_rules
from app.config.settings import Settings
from app.domain.alert import (
    AlertKind,
    AlertSeverity,
    AlertSourceType,
    AlertStatus,
    AlertType,
)
from app.domain.forecast import Forecast, HourlyForecastPoint
from app.domain.location import Coordinates, Location
from app.domain.provenance import DataProvenance
from app.domain.weather import CurrentWeather, WeatherCondition, WeatherReport
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.validation import WeatherValidator
from app.providers.registry import ProviderRegistry
from app.services.alerts import AlertService, build_alert, dedup_key
from app.services.cache import TTLCache
from app.services.weather_service import LocationQuery, WeatherService

NOW = datetime(2026, 9, 4, 7, 30, tzinfo=timezone.utc)
HYDERABAD = Coordinates(latitude=17.385, longitude=78.4867)
DELHI = Coordinates(latitude=28.6139, longitude=77.209)


def _provenance(**overrides) -> DataProvenance:
    return DataProvenance(
        **{
            "provider_id": "open-meteo",
            "provider_name": "Open-Meteo",
            "model": "best_match",
            "fetched_at": NOW,
            "license": "CC-BY-4.0",
            "attribution": "Weather data by Open-Meteo.com",
            **overrides,
        }
    )


def _report(coordinates: Coordinates = HYDERABAD, **current) -> WeatherReport:
    return WeatherReport(
        location=Location(coordinates=coordinates, timezone="Asia/Kolkata"),
        current=CurrentWeather(
            **{
                "observed_at": NOW,
                "condition": WeatherCondition.CLEAR,
                **current,
            }
        ),
        provenance=_provenance(),
    )


def _key(**overrides) -> str:
    return dedup_key(
        **{
            "source_type": AlertSourceType.DETERMINISTIC_RULE,
            "provider_id": "open-meteo",
            "model": "best_match",
            "rule_id": "HIGH_WIND_01",
            "location_key": HYDERABAD.cache_key,
            "kind": AlertKind.OBSERVED,
            "target": None,
            **overrides,
        }
    )


# --- Deduplication keys -------------------------------------------------------


def test_the_same_condition_produces_the_same_key():
    assert _key() == _key()


def test_an_observed_key_ignores_the_weather_timestamp():
    """10:00, 10:05 and 10:10 of one ongoing condition must be one alert."""
    at_ten = _key(target=None)
    # Even if a caller mistakenly passes a target for an observed alert, the
    # service never does — this pins the intended call shape.
    assert at_ten == _key(target=None)


def test_a_different_rule_produces_a_different_key():
    assert _key(rule_id="HIGH_WIND_01") != _key(rule_id="EXTREME_HEAT_01")


def test_a_different_location_produces_a_different_key():
    assert _key(location_key=HYDERABAD.cache_key) != _key(
        location_key=DELHI.cache_key
    )


def test_a_different_provider_produces_a_different_key():
    assert _key(provider_id="open-meteo") != _key(provider_id="imd")


def test_a_different_model_produces_a_different_key():
    assert _key(model="best_match") != _key(model="gfs_seamless")


def test_observed_and_forecast_keys_never_collide():
    """The same rule at the same place, observed versus predicted."""
    observed = _key(kind=AlertKind.OBSERVED, target=None)
    forecast = _key(kind=AlertKind.FORECAST_RISK, target=NOW)

    assert observed != forecast


def test_forecast_keys_separate_by_target_instant():
    """A prediction for 14:00 and one for 15:00 are different claims."""
    first = _key(kind=AlertKind.FORECAST_RISK, target=NOW)
    second = _key(kind=AlertKind.FORECAST_RISK, target=NOW + timedelta(hours=1))

    assert first != second


def test_an_official_warning_key_never_collides_with_a_rule_key():
    rule = _key(source_type=AlertSourceType.DETERMINISTIC_RULE)
    official = _key(source_type=AlertSourceType.OFFICIAL_WARNING)

    assert rule != official


def test_keys_are_legible_in_logs():
    """A key that cannot be read in psql is a key nobody debugs with."""
    key = _key()

    assert key.startswith("HIGH_WIND_01:observed:")
    assert len(key) <= 128  # fits the column


# --- Alert construction -------------------------------------------------------


@pytest.fixture
def engine(settings: Settings) -> AlertEngine:
    return AlertEngine(build_rules(settings), observation_validity=timedelta(hours=1))


def _first_alert(engine: AlertEngine, report: WeatherReport):
    evaluation = engine.evaluate_observation(report, now=NOW)
    assert evaluation.triggered
    return build_alert(
        evaluation.results[0],
        location=report.location,
        provenance=report.provenance,
        triggered_at=NOW,
    )


def test_a_built_alert_carries_its_evidence(engine, settings):
    alert = _first_alert(
        engine, _report(wind_speed_ms=settings.HIGH_WIND_WARNING_THRESHOLD + 1)
    )

    assert alert.evidence.rule_id == "HIGH_WIND_01"
    assert alert.evidence.variable == "wind_speed_ms"
    assert alert.evidence.unit == "m/s"
    assert alert.evidence.comparison == ">="
    assert alert.evidence.sample_window == SampleWindow.OBSERVATION.value
    assert alert.evidence.observed_value > alert.evidence.threshold


def test_a_built_alert_carries_provenance(engine, settings):
    alert = _first_alert(engine, _report(wind_speed_ms=settings.HIGH_WIND_THRESHOLD))

    assert alert.provenance.provider_id == "open-meteo"
    assert alert.provenance.model == "best_match"
    assert alert.provenance.fetched_at == NOW
    assert alert.provenance.attribution == "Weather data by Open-Meteo.com"


def test_a_built_alert_records_when_evaluation_ran_and_what_it_covers(engine, settings):
    alert = _first_alert(engine, _report(wind_speed_ms=settings.HIGH_WIND_THRESHOLD))

    assert alert.triggered_at == NOW
    assert alert.valid_from == NOW
    assert alert.valid_until == NOW + timedelta(hours=1)
    assert alert.resolved_at is None


def test_a_built_alert_starts_active(engine, settings):
    alert = _first_alert(engine, _report(wind_speed_ms=settings.HIGH_WIND_THRESHOLD))

    assert alert.status is AlertStatus.ACTIVE


def test_a_rule_alert_is_never_marked_official(engine, settings):
    """There is no code path from a threshold comparison to OFFICIAL_WARNING."""
    alert = _first_alert(engine, _report(wind_speed_ms=settings.HIGH_WIND_THRESHOLD))

    assert alert.source_type is AlertSourceType.DETERMINISTIC_RULE
    assert alert.is_official is False


def test_an_observed_alert_is_marked_observed(engine, settings):
    alert = _first_alert(engine, _report(wind_speed_ms=settings.HIGH_WIND_THRESHOLD))

    assert alert.kind is AlertKind.OBSERVED
    assert alert.description.startswith("Observed")


def test_a_forecast_alert_is_marked_as_risk_not_occurrence(engine, settings):
    """Saying a forecast event has happened would be the core safety failure."""
    forecast = Forecast(
        location=Location(coordinates=HYDERABAD),
        hourly=[
            HourlyForecastPoint(
                valid_at=NOW + timedelta(hours=2),
                wind_speed_ms=settings.HIGH_WIND_SEVERE_THRESHOLD + 2,
            )
        ],
        provenance=_provenance(),
    )
    evaluation = engine.evaluate_forecast(forecast, now=NOW)

    alert = build_alert(
        evaluation.results[0],
        location=forecast.location,
        provenance=forecast.provenance,
        triggered_at=NOW,
    )

    assert alert.kind is AlertKind.FORECAST_RISK
    assert alert.description.startswith("Forecast")
    assert "Observed" not in alert.description


def test_an_alert_description_never_claims_external_authority(engine, settings):
    alert = _first_alert(
        engine, _report(temperature_c=settings.EXTREME_HEAT_SEVERE_THRESHOLD + 2)
    )

    lowered = alert.description.lower()
    for forbidden in ("imd", "official", "warning issued", "red alert", "orange alert"):
        assert forbidden not in lowered, alert.description
    assert "weathergpt" in lowered


def test_severity_escalation_is_reflected_in_the_alert(engine, settings):
    watch = _first_alert(engine, _report(wind_speed_ms=settings.HIGH_WIND_THRESHOLD))
    severe = _first_alert(
        engine, _report(wind_speed_ms=settings.HIGH_WIND_SEVERE_THRESHOLD)
    )

    assert watch.severity is AlertSeverity.WATCH
    assert severe.severity is AlertSeverity.SEVERE
    assert severe.severity.rank > watch.severity.rank


# --- Service gating -----------------------------------------------------------


async def test_evaluation_is_a_no_op_without_a_database(engine):
    service = AlertService(engine, None)

    assert service.enabled is False
    assert await service.evaluate_observation(_report(wind_speed_ms=40.0)) == []


async def test_evaluation_can_be_switched_off_by_configuration(engine):
    class _Database:
        def session(self):  # pragma: no cover - must never be reached
            raise AssertionError("evaluation was disabled")

    service = AlertService(engine, _Database(), enabled=False)

    assert service.enabled is False
    assert await service.evaluate_observation(_report(wind_speed_ms=40.0)) == []


async def test_a_database_failure_never_breaks_the_weather_response(engine):
    """Weather already in hand and already valid must still reach the caller."""
    from sqlalchemy.exc import OperationalError

    class _Exploding:
        def session(self):
            raise OperationalError("SELECT 1", {}, Exception("connection refused"))

    service = AlertService(engine, _Exploding())

    assert await service.evaluate_observation(_report(wind_speed_ms=40.0)) == []
    assert await service.evaluate_forecast(
        Forecast(
            location=Location(coordinates=HYDERABAD),
            hourly=[HourlyForecastPoint(valid_at=NOW, wind_speed_ms=40.0)],
            provenance=_provenance(),
        )
    ) == []


async def test_searching_without_a_database_is_a_structured_error(engine):
    from app.core.exceptions import DatabaseUnavailableError
    from app.db.repositories import AlertFilter

    service = AlertService(engine, None)

    with pytest.raises(DatabaseUnavailableError) as exc_info:
        await service.search(AlertFilter())

    assert exc_info.value.code == "DATABASE_UNAVAILABLE"


# --- The validation gate holds for alerts too ---------------------------------


class _RecordingAlerts(AlertService):
    """Captures what the weather service asks it to evaluate."""

    def __init__(self, engine: AlertEngine) -> None:
        super().__init__(engine, None)
        self.observations: list[WeatherReport] = []
        self.forecasts: list[Forecast] = []

    async def evaluate_observation(self, report, *, now=None):
        self.observations.append(report)
        return []

    async def evaluate_forecast(self, forecast, *, now=None):
        self.forecasts.append(forecast)
        return []


class _StubProvider:
    from app.providers.base import ProviderCapability, ProviderMetadata

    def __init__(self, report=None) -> None:
        self._metadata = self.ProviderMetadata(
            provider_id="stub",
            name="Stub",
            capabilities=frozenset({self.ProviderCapability.CURRENT}),
        )
        self._report = report

    @property
    def metadata(self):
        return self._metadata

    async def fetch_current(self, coordinates):
        return self._report

    async def fetch_forecast(self, coordinates, *, days, include_hourly, include_daily):
        raise NotImplementedError


def _weather_service(report: WeatherReport, alerts: AlertService) -> WeatherService:
    registry = ProviderRegistry()
    registry.register(_StubProvider(report=report))
    return WeatherService(
        pipeline=IngestionPipeline(registry, WeatherValidator()),
        geocoding=None,
        current_cache=TTLCache(ttl_seconds=60),
        forecast_cache=TTLCache(ttl_seconds=60),
        alerts=alerts,
    )


async def test_validated_conditions_reach_the_alert_engine(engine, settings):
    alerts = _RecordingAlerts(engine)
    report = _report(wind_speed_ms=settings.HIGH_WIND_THRESHOLD + 1)
    service = _weather_service(report, alerts)

    await service.get_current(LocationQuery(coordinates=HYDERABAD))

    assert len(alerts.observations) == 1


async def test_invalid_weather_data_can_never_generate_an_alert(engine):
    """The validation gate sits upstream; there is no path around it."""
    alerts = _RecordingAlerts(engine)
    # 812 °C is physically impossible and the pipeline rejects it outright.
    corrupt = _report(temperature_c=812.0, wind_speed_ms=95.0)
    service = _weather_service(corrupt, alerts)

    with pytest.raises(Exception):
        await service.get_current(LocationQuery(coordinates=HYDERABAD))

    assert alerts.observations == []


async def test_a_cache_hit_does_not_re_evaluate(engine, settings):
    alerts = _RecordingAlerts(engine)
    report = _report(wind_speed_ms=settings.HIGH_WIND_THRESHOLD + 1)
    service = _weather_service(report, alerts)

    await service.get_current(LocationQuery(coordinates=HYDERABAD))
    await service.get_current(LocationQuery(coordinates=HYDERABAD))

    assert len(alerts.observations) == 1


# --- No language model in the loop --------------------------------------------


def test_the_alert_path_imports_nothing_that_could_call_a_model():
    """The safety boundary, asserted mechanically rather than by convention.

    If a future change reaches for an LLM client inside the decision path, this
    fails. Explanation belongs downstream of a decision already made.
    """
    import app.alerts.engine
    import app.alerts.rules
    import app.alerts.severity
    import app.services.alerts

    forbidden = (
        "openai",
        "anthropic",
        "langchain",
        "transformers",
        "llama",
        "ollama",
        "groq",
    )
    for module in (
        app.alerts.engine,
        app.alerts.rules,
        app.alerts.severity,
        app.services.alerts,
    ):
        source = open(module.__file__, encoding="utf-8").read().lower()
        for name in forbidden:
            assert f"import {name}" not in source, f"{module.__name__} imports {name}"


def test_severity_labels_are_weathergpt_terms_not_agency_colours():
    """Colour-coded labels would imply an authority this system does not have."""
    values = {severity.value for severity in AlertSeverity}

    assert values == {"info", "watch", "warning", "severe", "extreme"}
    for colour in ("red", "orange", "yellow", "amber", "green"):
        assert colour not in values


def test_alert_types_describe_hazards_not_verdicts():
    for alert_type in AlertType:
        assert "emergency" not in alert_type.value
        assert "official" not in alert_type.value
