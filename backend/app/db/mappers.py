"""Translation between the canonical model and persisted rows.

Kept as pure functions, separate from the repositories, for two reasons: the
mapping is the part most likely to silently drop a field, and pure functions can
be tested exhaustively without a database.

The invariant these functions exist to protect: **persistence must not strip
provenance**. A row written here can always answer where its numbers came from,
which model produced them, and when they were obtained.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from geoalchemy2 import WKTElement

from app.db.models import (
    SRID,
    UNDISCLOSED_MODEL,
    WeatherAlert,
    WeatherForecast,
    WeatherObservation,
)
from app.domain.alert import (
    Alert,
    AlertEvidence,
    AlertKind,
    AlertSeverity,
    AlertSourceType,
    AlertStatus,
    AlertType,
)
from app.domain.forecast import DailyForecastPoint, Forecast, HourlyForecastPoint
from app.domain.location import Coordinates, Location
from app.domain.provenance import DataProvenance
from app.domain.weather import CurrentWeather, WeatherCondition, WeatherReport


class ForecastResolution:
    """The two forecast series, as stored."""

    HOURLY = "hourly"
    DAILY = "daily"


def point(latitude: float, longitude: float) -> WKTElement:
    """A PostGIS point. Note the order: WKT is (x y), so longitude first."""
    return WKTElement(f"POINT({longitude} {latitude})", srid=SRID)


def _location_columns(location: Location) -> dict[str, Any]:
    coordinates = location.coordinates
    return {
        "location_key": coordinates.cache_key,
        "latitude": coordinates.latitude,
        "longitude": coordinates.longitude,
        "elevation_m": coordinates.elevation_m,
        "geom": point(coordinates.latitude, coordinates.longitude),
        "timezone": location.timezone,
    }


def _provenance_columns(provenance: DataProvenance) -> dict[str, Any]:
    return {
        "provider_id": provenance.provider_id,
        "provider_name": provenance.provider_name,
        "model": provenance.model or UNDISCLOSED_MODEL,
        "fetched_at": provenance.fetched_at,
        "model_run_at": provenance.model_run_at,
        "source_url": provenance.source_url,
        "license": provenance.license,
        "attribution": provenance.attribution,
    }


def _condition_columns(record: Any) -> dict[str, Any]:
    return {
        "condition": record.condition.value,
        "condition_description": record.condition_description,
        "wmo_code": record.wmo_code,
    }


OBSERVATION_MEASUREMENTS = (
    "temperature_c",
    "apparent_temperature_c",
    "dew_point_c",
    "relative_humidity_pct",
    "pressure_msl_hpa",
    "surface_pressure_hpa",
    "wind_speed_ms",
    "wind_gust_ms",
    "wind_direction_deg",
    "precipitation_mm",
    "cloud_cover_pct",
    "visibility_m",
    "uv_index",
    "is_day",
)

HOURLY_MEASUREMENTS = (
    "temperature_c",
    "apparent_temperature_c",
    "dew_point_c",
    "relative_humidity_pct",
    "pressure_msl_hpa",
    "wind_speed_ms",
    "wind_gust_ms",
    "wind_direction_deg",
    "precipitation_mm",
    "precipitation_probability_pct",
    "cloud_cover_pct",
    "visibility_m",
    "uv_index",
    "is_day",
)

# Daily aggregates keep their own column names; the pairs below map the
# canonical field onto the stored one.
DAILY_MEASUREMENTS = (
    ("temperature_min_c", "temperature_min_c"),
    ("temperature_max_c", "temperature_max_c"),
    ("apparent_temperature_min_c", "apparent_temperature_min_c"),
    ("apparent_temperature_max_c", "apparent_temperature_max_c"),
    ("precipitation_sum_mm", "precipitation_mm"),
    ("precipitation_hours", "precipitation_hours"),
    ("precipitation_probability_max_pct", "precipitation_probability_pct"),
    ("wind_speed_max_ms", "wind_speed_ms"),
    ("wind_gust_max_ms", "wind_gust_ms"),
    ("wind_direction_dominant_deg", "wind_direction_deg"),
    ("uv_index_max", "uv_index"),
)


# --- Canonical model -> row --------------------------------------------------


def observation_values(report: WeatherReport) -> dict[str, Any]:
    """Column values for one observation row."""
    current = report.current
    values: dict[str, Any] = {
        **_location_columns(report.location),
        **_provenance_columns(report.provenance),
        **_condition_columns(current),
        "observed_at": current.observed_at,
    }
    values.update(
        {field: getattr(current, field) for field in OBSERVATION_MEASUREMENTS}
    )
    return values


def hourly_forecast_values(
    forecast: Forecast, point_record: HourlyForecastPoint, *, created_at: datetime
) -> dict[str, Any]:
    """Column values for one hourly forecast row."""
    values: dict[str, Any] = {
        **_location_columns(forecast.location),
        **_provenance_columns(forecast.provenance),
        **_condition_columns(point_record),
        "resolution": ForecastResolution.HOURLY,
        "forecast_created_at": created_at,
        "forecast_for": point_record.valid_at,
    }
    values.update(
        {field: getattr(point_record, field) for field in HOURLY_MEASUREMENTS}
    )
    return values


def daily_forecast_values(
    forecast: Forecast, point_record: DailyForecastPoint, *, created_at: datetime
) -> dict[str, Any]:
    """Column values for one daily forecast row.

    A daily point covers a local calendar day; its ``forecast_for`` is that
    day's start. Sunrise is the best available anchor for local midnight, and
    the date itself is the fallback.
    """
    valid_at = datetime.combine(
        point_record.date, datetime.min.time(), tzinfo=timezone.utc
    )
    values: dict[str, Any] = {
        **_location_columns(forecast.location),
        **_provenance_columns(forecast.provenance),
        **_condition_columns(point_record),
        "resolution": ForecastResolution.DAILY,
        "forecast_created_at": created_at,
        "forecast_for": valid_at,
        "sunrise": point_record.sunrise,
        "sunset": point_record.sunset,
    }
    values.update(
        {
            column: getattr(point_record, field)
            for field, column in DAILY_MEASUREMENTS
        }
    )
    return values


# --- Row -> canonical model --------------------------------------------------


def _condition(row: Any) -> WeatherCondition:
    try:
        return WeatherCondition(row.condition)
    except ValueError:
        # A row written by a newer version with a condition this build does not
        # know must not break the read path.
        return WeatherCondition.UNKNOWN


def row_provenance(row: WeatherObservation | WeatherForecast | WeatherAlert) -> DataProvenance:
    """Reconstruct provenance from a stored row."""
    return DataProvenance(
        provider_id=row.provider_id,
        provider_name=row.provider_name,
        model=row.model or None,
        fetched_at=row.fetched_at,
        model_run_at=row.model_run_at,
        source_url=row.source_url,
        license=row.license,
        attribution=row.attribution,
        # Historical reads come from durable storage, not the response cache.
        cached=False,
    )


def row_location(row: WeatherObservation | WeatherForecast | WeatherAlert) -> Location:
    """Reconstruct the location of a stored row."""
    return Location(
        coordinates=Coordinates(
            latitude=row.latitude,
            longitude=row.longitude,
            elevation_m=row.elevation_m,
        ),
        timezone=row.timezone,
    )


def row_to_current_weather(row: WeatherObservation) -> CurrentWeather:
    """Reconstruct the canonical measurement record from a stored row."""
    return CurrentWeather(
        observed_at=row.observed_at,
        condition=_condition(row),
        condition_description=row.condition_description,
        wmo_code=row.wmo_code,
        **{field: getattr(row, field) for field in OBSERVATION_MEASUREMENTS},
    )


def row_to_report(row: WeatherObservation) -> WeatherReport:
    """Reconstruct a full canonical report — measurements plus provenance."""
    return WeatherReport(
        location=row_location(row),
        current=row_to_current_weather(row),
        provenance=row_provenance(row),
    )


# --- Alerts ------------------------------------------------------------------


def alert_values(alert: Alert, *, dedup_key: str) -> dict[str, Any]:
    """Column values for one alert row.

    Evidence is split deliberately: the variable, value, threshold and unit are
    columns because they are queried and audited, while only the open-ended
    supporting context becomes JSON.
    """
    evidence = alert.evidence
    return {
        **_location_columns(alert.location),
        **_provenance_columns(alert.provenance),
        "dedup_key": dedup_key,
        "alert_type": alert.alert_type.value,
        "severity": alert.severity.value,
        "status": alert.status.value,
        "source_type": alert.source_type.value,
        "kind": alert.kind.value,
        "rule_id": alert.rule_id,
        "title": alert.title,
        "description": alert.description,
        "triggered_at": alert.triggered_at,
        "valid_from": alert.valid_from,
        "valid_until": alert.valid_until,
        "resolved_at": alert.resolved_at,
        "variable": evidence.variable,
        "observed_value": evidence.observed_value,
        "threshold": evidence.threshold,
        "unit": evidence.unit,
        "comparison": evidence.comparison,
        "sample_window": evidence.sample_window,
        "evidence_context": evidence.context or None,
        # Alerts carry no condition of their own; the column is inherited from
        # the shared spatial record and records the hazard type.
        "condition": alert.alert_type.value,
    }


def _enum_or(value: str, enum_type, fallback):
    """Coerce a stored string, degrading rather than raising.

    A row written by a newer build with a value this one does not know must not
    break the read path.
    """
    try:
        return enum_type(value)
    except ValueError:
        return fallback


def row_to_alert(row: WeatherAlert) -> Alert:
    """Reconstruct a canonical alert from a stored row."""
    return Alert(
        id=row.id,
        alert_type=_enum_or(row.alert_type, AlertType, AlertType.HEAVY_RAINFALL),
        severity=_enum_or(row.severity, AlertSeverity, AlertSeverity.INFO),
        status=_enum_or(row.status, AlertStatus, AlertStatus.EXPIRED),
        source_type=_enum_or(
            row.source_type, AlertSourceType, AlertSourceType.DETERMINISTIC_RULE
        ),
        kind=_enum_or(row.kind, AlertKind, AlertKind.OBSERVED),
        rule_id=row.rule_id,
        title=row.title,
        description=row.description,
        location=row_location(row),
        triggered_at=row.triggered_at,
        valid_from=row.valid_from,
        valid_until=row.valid_until,
        resolved_at=row.resolved_at,
        evidence=AlertEvidence(
            rule_id=row.rule_id,
            variable=row.variable,
            observed_value=row.observed_value,
            threshold=row.threshold,
            unit=row.unit,
            comparison=row.comparison,
            sample_window=row.sample_window,
            context=row.evidence_context,
        ),
        provenance=row_provenance(row),
        created_at=row.ingested_at,
        updated_at=row.updated_at,
    )
