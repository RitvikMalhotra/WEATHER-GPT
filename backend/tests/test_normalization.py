"""Mapping Open-Meteo payloads onto the canonical model.

Pure-function tests: no HTTP, no application, just payload in and domain object
out. This is where a provider's wire format is pinned down.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.domain.provenance import DataProvenance
from app.domain.weather import WeatherCondition, interpret_wmo_code
from app.providers.open_meteo.client import OpenMeteoPayload
from app.providers.open_meteo.normalizer import (
    NormalizationError,
    normalize_current,
    normalize_forecast,
)


@pytest.fixture
def provenance() -> DataProvenance:
    return DataProvenance(
        provider_id="open-meteo",
        provider_name="Open-Meteo",
        fetched_at=datetime(2026, 9, 4, 7, 30, tzinfo=timezone.utc),
    )


def _report(payload: dict, provenance: DataProvenance):
    return normalize_current(OpenMeteoPayload.model_validate(payload), provenance=provenance)


# --- Current conditions ------------------------------------------------------


def test_current_conditions_are_mapped_to_canonical_fields(current_payload, provenance):
    report = _report(current_payload, provenance)
    current = report.current

    assert current.temperature_c == pytest.approx(31.4)
    assert current.apparent_temperature_c == pytest.approx(35.2)
    assert current.relative_humidity_pct == pytest.approx(62.0)
    assert current.pressure_msl_hpa == pytest.approx(1004.2)
    assert current.surface_pressure_hpa == pytest.approx(987.6)
    assert current.cloud_cover_pct == pytest.approx(40.0)
    assert current.precipitation_mm == pytest.approx(0.0)
    assert current.wind_direction_deg == pytest.approx(115.0)
    assert current.is_day is True


def test_wind_is_converted_from_the_declared_kilometres_per_hour(current_payload, provenance):
    """The payload declares km/h; the model is metres per second."""
    current = _report(current_payload, provenance).current

    assert current.wind_speed_ms == pytest.approx(5.0)  # 18.0 km/h
    assert current.wind_gust_ms == pytest.approx(10.0)  # 36.0 km/h


def test_units_are_read_from_the_payload_rather_than_assumed(current_payload, provenance):
    """Same number, different declared unit, different canonical result."""
    current_payload["current_units"]["wind_speed_10m"] = "m/s"
    current = _report(current_payload, provenance).current

    assert current.wind_speed_ms == pytest.approx(18.0)


def test_local_timestamps_become_utc(current_payload, provenance):
    """Open-Meteo reports naive local time plus an offset; we store UTC."""
    report = _report(current_payload, provenance)

    assert report.current.observed_at == datetime(2026, 9, 4, 7, 30, tzinfo=timezone.utc)


def test_the_served_grid_point_is_recorded(current_payload, provenance):
    location = _report(current_payload, provenance).location

    assert location.coordinates.latitude == pytest.approx(28.625)
    assert location.coordinates.elevation_m == pytest.approx(216.0)
    assert location.timezone == "Asia/Kolkata"


def test_a_missing_variable_stays_missing_rather_than_becoming_zero(
    current_payload, provenance
):
    """Unknown precipitation and zero precipitation are different facts."""
    del current_payload["current"]["precipitation"]
    current = _report(current_payload, provenance).current

    assert current.precipitation_mm is None
    assert current.temperature_c is not None  # the rest still maps


def test_an_undeclared_unit_fails_normalisation(current_payload, provenance):
    del current_payload["current_units"]["temperature_2m"]

    with pytest.raises(NormalizationError) as exc_info:
        _report(current_payload, provenance)

    assert exc_info.value.details["variable"] == "temperature_2m"


def test_an_unsupported_unit_fails_normalisation(current_payload, provenance):
    current_payload["current_units"]["temperature_2m"] = "degrees Rankine"

    with pytest.raises(NormalizationError):
        _report(current_payload, provenance)


def test_a_payload_without_current_conditions_is_rejected(current_payload, provenance):
    del current_payload["current"]

    with pytest.raises(NormalizationError):
        _report(current_payload, provenance)


# --- Weather codes -----------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "condition"),
    [
        (0, WeatherCondition.CLEAR),
        (2, WeatherCondition.PARTLY_CLOUDY),
        (45, WeatherCondition.FOG),
        (65, WeatherCondition.RAIN),
        (80, WeatherCondition.RAIN_SHOWERS),
        (95, WeatherCondition.THUNDERSTORM),
        (99, WeatherCondition.THUNDERSTORM_WITH_HAIL),
    ],
)
def test_wmo_codes_map_to_conditions(code, condition):
    assert interpret_wmo_code(code)[0] is condition


def test_an_unrecognised_weather_code_degrades_rather_than_raising():
    """A strange code must not cost us an otherwise valid temperature."""
    condition, description = interpret_wmo_code(7)

    assert condition is WeatherCondition.UNKNOWN
    assert description is None


# --- Forecast ----------------------------------------------------------------


def _forecast(payload: dict, provenance: DataProvenance):
    return normalize_forecast(OpenMeteoPayload.model_validate(payload), provenance=provenance)


def test_column_arrays_are_transposed_into_hourly_points(forecast_payload, provenance):
    forecast = _forecast(forecast_payload, provenance)

    assert len(forecast.hourly) == 3
    first = forecast.hourly[0]
    assert first.temperature_c == pytest.approx(31.4)
    assert first.dew_point_c == pytest.approx(23.1)
    assert first.wind_speed_ms == pytest.approx(5.0)
    assert first.visibility_m == pytest.approx(24000.0)
    assert first.valid_at == datetime(2026, 9, 4, 7, 30, tzinfo=timezone.utc)
    assert forecast.hourly[2].condition is WeatherCondition.RAIN_SHOWERS


def test_daily_points_keep_the_local_calendar_date(forecast_payload, provenance):
    forecast = _forecast(forecast_payload, provenance)

    assert len(forecast.daily) == 2
    first = forecast.daily[0]
    assert first.date.isoformat() == "2026-09-04"
    assert first.temperature_min_c == pytest.approx(26.1)
    assert first.temperature_max_c == pytest.approx(34.8)
    assert first.precipitation_sum_mm == pytest.approx(4.2)
    assert first.wind_speed_max_ms == pytest.approx(7.0)  # 25.2 km/h
    assert first.condition is WeatherCondition.RAIN_SHOWERS
    assert first.sunrise == datetime(2026, 9, 4, 0, 32, tzinfo=timezone.utc)


def test_ragged_upstream_arrays_do_not_crash_the_mapping(forecast_payload, provenance):
    """A short column yields missing values, not an IndexError."""
    forecast_payload["hourly"]["temperature_2m"] = [31.4]

    forecast = _forecast(forecast_payload, provenance)

    assert len(forecast.hourly) == 3
    assert forecast.hourly[0].temperature_c == pytest.approx(31.4)
    assert forecast.hourly[1].temperature_c is None


def test_a_forecast_with_no_series_is_rejected(forecast_payload, provenance):
    del forecast_payload["hourly"]
    del forecast_payload["daily"]

    with pytest.raises(NormalizationError):
        _forecast(forecast_payload, provenance)
