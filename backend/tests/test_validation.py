"""The meteorological validation engine.

The gate that decides whether retrieved data is fit to serve. These tests pin
down both halves of the contract: implausible data must be rejected, and
plausible-but-extreme data must not be.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from app.domain.forecast import DailyForecastPoint, Forecast, HourlyForecastPoint
from app.domain.location import Coordinates, Location
from app.domain.provenance import DataProvenance
from app.domain.weather import CurrentWeather, WeatherReport
from app.ingestion.validation import ValidationSeverity, WeatherValidator

NOW = datetime(2026, 9, 4, 7, 30, tzinfo=timezone.utc)


def _provenance() -> DataProvenance:
    return DataProvenance(
        provider_id="test", provider_name="Test Source", fetched_at=NOW
    )


def _report(**fields) -> WeatherReport:
    defaults = {"observed_at": NOW, "temperature_c": 21.0}
    return WeatherReport(
        location=Location(coordinates=Coordinates(latitude=28.6, longitude=77.2)),
        current=CurrentWeather(**{**defaults, **fields}),
        provenance=_provenance(),
    )


def _forecast(*, hourly=(), daily=()) -> Forecast:
    return Forecast(
        location=Location(coordinates=Coordinates(latitude=28.6, longitude=77.2)),
        hourly=list(hourly),
        daily=list(daily),
        provenance=_provenance(),
    )


# --- Range checks ------------------------------------------------------------


def test_plausible_conditions_pass(validator):
    result = validator.validate_current(_report(), now=NOW)

    assert result.is_valid
    assert result.issues == ()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature_c", 150.0),
        ("temperature_c", -120.0),
        ("relative_humidity_pct", 140.0),
        ("relative_humidity_pct", -5.0),
        ("pressure_msl_hpa", 3.0),
        ("wind_speed_ms", -2.0),
        ("wind_direction_deg", 400.0),
        ("cloud_cover_pct", 250.0),
        ("precipitation_mm", -1.0),
    ],
)
def test_physically_impossible_values_are_rejected(validator, field, value):
    result = validator.validate_current(_report(**{field: value}), now=NOW)

    assert not result.is_valid
    assert result.errors[0].field == f"current.{field}"
    assert result.errors[0].severity is ValidationSeverity.ERROR


@pytest.mark.parametrize("temperature", [-89.2, 56.7])
def test_record_breaking_but_real_temperatures_are_accepted(validator, temperature):
    """Bounds must reject corruption without rejecting actual world records."""
    result = validator.validate_current(_report(temperature_c=temperature), now=NOW)

    assert result.is_valid


def test_a_unit_confusion_that_looks_plausible_per_field_is_caught(validator):
    """Reporting km/h as m/s inflates a 300 km/h gust into an impossible 300 m/s."""
    result = validator.validate_current(
        _report(wind_speed_ms=300.0, wind_gust_ms=360.0), now=NOW
    )

    assert not result.is_valid
    assert {issue.field for issue in result.errors} == {
        "current.wind_speed_ms",
        "current.wind_gust_ms",
    }


# --- Cross-field consistency -------------------------------------------------


def test_dew_point_above_air_temperature_is_rejected(validator):
    result = validator.validate_current(
        _report(temperature_c=20.0, dew_point_c=26.0), now=NOW
    )

    assert not result.is_valid
    assert result.errors[0].field == "current.dew_point_c"


def test_a_rounding_scale_dew_point_crossover_is_tolerated(validator):
    result = validator.validate_current(
        _report(temperature_c=20.0, dew_point_c=20.3), now=NOW
    )

    assert result.is_valid


def test_a_gust_weaker_than_the_sustained_wind_warns_but_still_serves(validator):
    result = validator.validate_current(
        _report(wind_speed_ms=12.0, wind_gust_ms=6.0), now=NOW
    )

    assert result.is_valid  # warnings do not reject
    assert result.warnings[0].field == "current.wind_gust_ms"


def test_rain_at_very_low_humidity_warns(validator):
    result = validator.validate_current(
        _report(relative_humidity_pct=12.0, precipitation_mm=4.0), now=NOW
    )

    assert result.is_valid
    assert result.warnings[0].field == "current.relative_humidity_pct"


# --- Temporal checks ---------------------------------------------------------


def test_an_observation_from_the_future_is_rejected(validator):
    result = validator.validate_current(
        _report(observed_at=NOW + timedelta(hours=6)), now=NOW
    )

    assert not result.is_valid
    assert result.errors[0].field == "current.observed_at"


def test_modest_clock_skew_is_tolerated(validator):
    result = validator.validate_current(
        _report(observed_at=NOW + timedelta(minutes=20)), now=NOW
    )

    assert result.is_valid


def test_a_stale_observation_warns_but_still_serves(validator):
    result = validator.validate_current(
        _report(observed_at=NOW - timedelta(hours=9)), now=NOW
    )

    assert result.is_valid
    assert result.warnings[0].field == "current.observed_at"


# --- Forecasts ---------------------------------------------------------------


def test_a_valid_forecast_passes(validator):
    forecast = _forecast(
        hourly=[HourlyForecastPoint(valid_at=NOW, temperature_c=30.0)],
        daily=[
            DailyForecastPoint(
                date=date(2026, 9, 4), temperature_min_c=26.0, temperature_max_c=35.0
            )
        ],
    )

    assert validator.validate_forecast(forecast, now=NOW).is_valid


def test_a_daily_minimum_above_its_maximum_is_rejected(validator):
    forecast = _forecast(
        daily=[
            DailyForecastPoint(
                date=date(2026, 9, 4), temperature_min_c=35.0, temperature_max_c=26.0
            )
        ]
    )

    result = validator.validate_forecast(forecast, now=NOW)

    assert not result.is_valid
    assert result.errors[0].field == "daily[0].temperature_min_c"


def test_the_failing_point_is_identified_by_index(validator):
    forecast = _forecast(
        hourly=[
            HourlyForecastPoint(valid_at=NOW, temperature_c=30.0),
            HourlyForecastPoint(valid_at=NOW, temperature_c=30.0),
            HourlyForecastPoint(valid_at=NOW, temperature_c=999.0),
        ]
    )

    result = validator.validate_forecast(forecast, now=NOW)

    assert not result.is_valid
    assert result.errors[0].field == "hourly[2].temperature_c"


def test_an_empty_forecast_is_rejected(validator):
    result = validator.validate_forecast(_forecast(), now=NOW)

    assert not result.is_valid


def test_forecast_points_are_not_subject_to_the_staleness_check(validator):
    """A forecast is about the future; ageing rules make no sense for it."""
    forecast = _forecast(
        hourly=[
            HourlyForecastPoint(valid_at=NOW + timedelta(days=5), temperature_c=30.0)
        ]
    )

    assert validator.validate_forecast(forecast, now=NOW).is_valid
