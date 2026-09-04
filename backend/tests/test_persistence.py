"""Persistence behaviour that needs no database.

Everything here is either a pure mapping or a service decision, so it runs in
the default offline suite. The queries themselves are covered by
``test_db_integration.py`` against a real PostGIS.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest
from sqlalchemy.exc import OperationalError

from app.core.exceptions import DatabaseUnavailableError, WeatherProviderError
from app.db.engine import translate_database_errors
from app.db.mappers import (
    ForecastResolution,
    daily_forecast_values,
    hourly_forecast_values,
    observation_values,
    row_provenance,
    row_to_report,
)
from app.db.models import UNDISCLOSED_MODEL, WeatherObservation
from app.domain.forecast import DailyForecastPoint, Forecast, HourlyForecastPoint
from app.domain.location import Coordinates, Location
from app.domain.provenance import DataProvenance
from app.domain.weather import CurrentWeather, WeatherCondition, WeatherReport
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.validation import WeatherValidator
from app.providers.registry import ProviderRegistry
from app.services.cache import TTLCache
from app.services.persistence import PersistenceService, generation_bucket
from app.services.weather_service import LocationQuery, WeatherService

NOW = datetime(2026, 9, 4, 7, 30, tzinfo=timezone.utc)
DELHI = Coordinates(latitude=28.6139, longitude=77.209, elevation_m=216.0)


def _provenance(**overrides) -> DataProvenance:
    return DataProvenance(
        **{
            "provider_id": "open-meteo",
            "provider_name": "Open-Meteo",
            "model": "best_match",
            "fetched_at": NOW,
            "source_url": "https://api.open-meteo.com/v1/forecast",
            "license": "CC-BY-4.0",
            "attribution": "Weather data by Open-Meteo.com",
            **overrides,
        }
    )


def _report(**current) -> WeatherReport:
    return WeatherReport(
        location=Location(coordinates=DELHI, timezone="Asia/Kolkata"),
        current=CurrentWeather(
            **{
                "observed_at": NOW,
                "temperature_c": 31.4,
                "wind_speed_ms": 5.0,
                "condition": WeatherCondition.PARTLY_CLOUDY,
                "condition_description": "Partly cloudy",
                "wmo_code": 2,
                **current,
            }
        ),
        provenance=_provenance(),
    )


def _forecast(**overrides) -> Forecast:
    return Forecast(
        location=Location(coordinates=DELHI, timezone="Asia/Kolkata"),
        hourly=overrides.get(
            "hourly",
            [HourlyForecastPoint(valid_at=NOW, temperature_c=31.4, wind_speed_ms=5.0)],
        ),
        daily=overrides.get(
            "daily",
            [
                DailyForecastPoint(
                    date=date(2026, 9, 4),
                    temperature_min_c=26.1,
                    temperature_max_c=34.8,
                    precipitation_sum_mm=4.2,
                    wind_speed_max_ms=7.0,
                )
            ],
        ),
        provenance=overrides.get("provenance", _provenance()),
    )


# --- Mapping: canonical model -> row -----------------------------------------


def test_observation_values_carry_measurements_and_place():
    values = observation_values(_report())

    assert values["temperature_c"] == pytest.approx(31.4)
    assert values["wind_speed_ms"] == pytest.approx(5.0)
    assert values["observed_at"] == NOW
    assert values["latitude"] == pytest.approx(28.6139)
    assert values["elevation_m"] == pytest.approx(216.0)
    assert values["timezone"] == "Asia/Kolkata"
    assert values["condition"] == "partly_cloudy"
    assert values["wmo_code"] == 2


def test_persistence_never_strips_provenance():
    """The whole point of storing weather is being able to attribute it later."""
    values = observation_values(_report())

    assert values["provider_id"] == "open-meteo"
    assert values["provider_name"] == "Open-Meteo"
    assert values["model"] == "best_match"
    assert values["fetched_at"] == NOW
    assert values["source_url"] == "https://api.open-meteo.com/v1/forecast"
    assert values["license"] == "CC-BY-4.0"
    assert values["attribution"] == "Weather data by Open-Meteo.com"


def test_an_undisclosed_model_is_stored_explicitly_not_as_null():
    """A NULL would silently disable the uniqueness constraint."""
    report = _report()
    report = report.model_copy(
        update={"provenance": _provenance(model=None)}
    )

    assert observation_values(report)["model"] == UNDISCLOSED_MODEL


def test_a_missing_measurement_stays_null():
    values = observation_values(_report(precipitation_mm=None))

    assert values["precipitation_mm"] is None
    assert "precipitation_mm" in values  # present as NULL, not dropped


def test_the_geometry_is_a_wgs84_point_in_longitude_latitude_order():
    """WKT is (x y): getting this backwards puts Delhi in the Indian Ocean."""
    geom = observation_values(_report())["geom"]

    assert geom.srid == 4326
    assert geom.data == "POINT(77.209 28.6139)"


def test_the_location_key_matches_the_cache_key():
    """One notion of 'the same place' across the cache and the database."""
    values = observation_values(_report())

    assert values["location_key"] == DELHI.cache_key


def test_hourly_and_daily_points_are_stored_at_their_own_resolution():
    forecast = _forecast()

    hourly = hourly_forecast_values(forecast, forecast.hourly[0], created_at=NOW)
    daily = daily_forecast_values(forecast, forecast.daily[0], created_at=NOW)

    assert hourly["resolution"] == ForecastResolution.HOURLY
    assert daily["resolution"] == ForecastResolution.DAILY


def test_forecast_rows_separate_when_it_was_said_from_what_was_said():
    forecast = _forecast()
    created = datetime(2026, 9, 3, 6, 0, tzinfo=timezone.utc)

    row = hourly_forecast_values(forecast, forecast.hourly[0], created_at=created)

    assert row["forecast_created_at"] == created
    assert row["forecast_for"] == NOW
    assert row["forecast_created_at"] != row["forecast_for"]


def test_daily_aggregates_map_onto_their_shared_columns():
    forecast = _forecast()

    row = daily_forecast_values(forecast, forecast.daily[0], created_at=NOW)

    assert row["temperature_min_c"] == pytest.approx(26.1)
    assert row["temperature_max_c"] == pytest.approx(34.8)
    assert row["precipitation_mm"] == pytest.approx(4.2)  # from precipitation_sum_mm
    assert row["wind_speed_ms"] == pytest.approx(7.0)  # from wind_speed_max_ms
    assert row["forecast_for"] == datetime(2026, 9, 4, tzinfo=timezone.utc)


# --- Mapping: row -> canonical model -----------------------------------------


def test_a_stored_row_round_trips_back_to_the_canonical_model():
    row = WeatherObservation(**observation_values(_report()))

    report = row_to_report(row)

    assert report.current.temperature_c == pytest.approx(31.4)
    assert report.current.observed_at == NOW
    assert report.current.condition is WeatherCondition.PARTLY_CLOUDY
    assert report.location.coordinates.latitude == pytest.approx(28.6139)
    assert report.provenance.provider_id == "open-meteo"
    assert report.provenance.model == "best_match"


def test_the_undisclosed_model_sentinel_maps_back_to_none():
    row = WeatherObservation(**observation_values(_report()))
    row.model = UNDISCLOSED_MODEL

    assert row_provenance(row).model is None


def test_history_is_never_reported_as_a_cache_hit():
    """`cached` describes the response cache, and history does not come from it."""
    row = WeatherObservation(**observation_values(_report()))

    assert row_provenance(row).cached is False


def test_an_unrecognised_stored_condition_degrades_rather_than_raising():
    """A row written by a newer build must not break this one's read path."""
    row = WeatherObservation(**observation_values(_report()))
    row.condition = "sharknado"

    assert row_to_report(row).current.condition is WeatherCondition.UNKNOWN


# --- Forecast generation bucketing -------------------------------------------


def test_requests_within_a_window_share_a_generation():
    first = generation_bucket(datetime(2026, 9, 4, 7, 5, tzinfo=timezone.utc), minutes=60)
    second = generation_bucket(datetime(2026, 9, 4, 7, 55, tzinfo=timezone.utc), minutes=60)

    assert first == second == datetime(2026, 9, 4, 7, 0, tzinfo=timezone.utc)


def test_a_later_window_starts_a_new_generation():
    """Keeping successive generations is what allows forecast-skill analysis."""
    first = generation_bucket(datetime(2026, 9, 4, 7, 55, tzinfo=timezone.utc), minutes=60)
    second = generation_bucket(datetime(2026, 9, 4, 8, 5, tzinfo=timezone.utc), minutes=60)

    assert second == first + timedelta(hours=1)


# --- Service behaviour --------------------------------------------------------


async def test_persistence_is_a_no_op_without_a_configured_database():
    service = PersistenceService(None)

    assert service.enabled is False
    assert await service.record_observation(_report()) is False
    assert await service.record_forecast(_forecast()) == 0


class _ExplodingDatabase:
    """A database that fails on every session."""

    def session(self):
        raise OperationalError("SELECT 1", {}, Exception("connection refused"))


async def test_a_write_failure_never_breaks_the_response():
    """Weather already in hand and already valid must still reach the caller."""
    service = PersistenceService(_ExplodingDatabase())

    assert await service.record_observation(_report()) is False
    assert await service.record_forecast(_forecast()) == 0


async def test_an_empty_forecast_reaches_no_database_at_all():
    """No points means no statement, so a dead database is never touched."""
    service = PersistenceService(_ExplodingDatabase())

    assert await service.record_forecast(_forecast(hourly=[], daily=[])) == 0


# --- The validation gate is structural ----------------------------------------


class _RecordingPersistence(PersistenceService):
    """Captures what the weather service tries to persist."""

    def __init__(self) -> None:
        super().__init__(None)
        self.observations: list[WeatherReport] = []
        self.forecasts: list[Forecast] = []

    async def record_observation(self, report: WeatherReport) -> bool:
        self.observations.append(report)
        return True

    async def record_forecast(self, forecast: Forecast) -> int:
        self.forecasts.append(forecast)
        return len(forecast.hourly) + len(forecast.daily)


class _StubProvider:
    """Returns whatever a test hands it, valid or not."""

    from app.providers.base import ProviderCapability, ProviderMetadata

    def __init__(self, report=None, forecast=None, error=None) -> None:
        self._metadata = self.ProviderMetadata(
            provider_id="stub",
            name="Stub",
            capabilities=frozenset(
                {
                    self.ProviderCapability.CURRENT,
                    self.ProviderCapability.DAILY_FORECAST,
                    self.ProviderCapability.HOURLY_FORECAST,
                }
            ),
        )
        self._report = report
        self._forecast = forecast
        self._error = error

    @property
    def metadata(self):
        return self._metadata

    async def fetch_current(self, coordinates):
        if self._error:
            raise self._error
        return self._report

    async def fetch_forecast(self, coordinates, *, days, include_hourly, include_daily):
        if self._error:
            raise self._error
        return self._forecast


def _service(provider, persistence) -> WeatherService:
    registry = ProviderRegistry()
    registry.register(provider)
    return WeatherService(
        pipeline=IngestionPipeline(registry, WeatherValidator()),
        geocoding=None,  # unused: these tests pass coordinates
        current_cache=TTLCache(ttl_seconds=60),
        forecast_cache=TTLCache(ttl_seconds=60),
        persistence=persistence,
    )


async def test_a_validated_observation_is_persisted():
    persistence = _RecordingPersistence()
    service = _service(_StubProvider(report=_report()), persistence)

    await service.get_current(LocationQuery(coordinates=DELHI))

    assert len(persistence.observations) == 1
    assert persistence.observations[0].current.temperature_c == pytest.approx(31.4)


async def test_an_invalid_observation_is_never_persisted():
    """The database is not a bypass around the validation layer."""
    persistence = _RecordingPersistence()
    corrupt = _report(temperature_c=812.0)
    service = _service(_StubProvider(report=corrupt), persistence)

    with pytest.raises(Exception):
        await service.get_current(LocationQuery(coordinates=DELHI))

    assert persistence.observations == []


async def test_a_provider_failure_persists_nothing():
    persistence = _RecordingPersistence()
    service = _service(
        _StubProvider(error=WeatherProviderError("upstream down")), persistence
    )

    with pytest.raises(Exception):
        await service.get_current(LocationQuery(coordinates=DELHI))

    assert persistence.observations == []


async def test_a_validated_forecast_is_persisted():
    persistence = _RecordingPersistence()
    service = _service(_StubProvider(forecast=_forecast()), persistence)

    await service.get_forecast(LocationQuery(coordinates=DELHI), days=1)

    assert len(persistence.forecasts) == 1


async def test_an_invalid_forecast_is_never_persisted():
    persistence = _RecordingPersistence()
    broken = _forecast(
        daily=[
            DailyForecastPoint(
                date=date(2026, 9, 4), temperature_min_c=35.0, temperature_max_c=26.0
            )
        ],
        hourly=[],
    )
    service = _service(_StubProvider(forecast=broken), persistence)

    with pytest.raises(Exception):
        await service.get_forecast(LocationQuery(coordinates=DELHI), days=1)

    assert persistence.forecasts == []


async def test_a_cache_hit_does_not_rewrite_the_row():
    """Persistence runs in the cache loader, so repeats touch neither layer."""
    persistence = _RecordingPersistence()
    service = _service(_StubProvider(report=_report()), persistence)

    await service.get_current(LocationQuery(coordinates=DELHI))
    await service.get_current(LocationQuery(coordinates=DELHI))

    assert len(persistence.observations) == 1


# --- Error containment --------------------------------------------------------


async def test_driver_failures_are_translated_without_leaking_sql_or_credentials():
    """SQLAlchemy puts the statement, and sometimes the DSN, in its message."""
    secret_sql = "INSERT INTO weather_observations (id) VALUES ($1)"
    leaky = OperationalError(
        secret_sql,
        {"password": "hunter2"},
        Exception(
            "connection to postgresql://weathergpt:hunter2@db:5432/weathergpt failed"
        ),
    )

    with pytest.raises(DatabaseUnavailableError) as exc_info:
        async with translate_database_errors("history.observations"):
            raise leaky

    error = exc_info.value
    rendered = f"{error.code} {error.message} {error.details}"
    assert error.code == "DATABASE_UNAVAILABLE"
    assert "hunter2" not in rendered
    assert "INSERT INTO" not in rendered
    assert "postgresql://" not in rendered
    assert error.details == {"operation": "history.observations"}
