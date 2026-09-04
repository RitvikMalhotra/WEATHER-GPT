"""Normalisation: Open-Meteo payloads to the canonical domain model.

Pure functions — given a payload, they produce domain objects. No I/O, so every
mapping rule here is unit-testable without a network or a running application,
and a provider's mapping can be verified against a recorded payload.

Two rules this layer enforces:

* **Units are read, never assumed.** Each value is converted using the unit the
  payload declares beside it. A variable that arrives without a declared unit is
  a normalisation failure, not something to guess at.
* **Missing stays missing.** A variable the provider omitted becomes ``None``,
  never ``0``. Downstream, "no precipitation reported" and "0 mm of
  precipitation" are different facts.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Mapping, Sequence

from app.core.exceptions import WeatherProviderError
from app.domain.forecast import DailyForecastPoint, Forecast, HourlyForecastPoint
from app.domain.location import Coordinates, Location
from app.domain.provenance import DataProvenance
from app.domain.units import (
    UnknownUnitError,
    to_celsius,
    to_hectopascals,
    to_metres,
    to_metres_per_second,
    to_millimetres,
)
from app.domain.weather import CurrentWeather, WeatherReport, interpret_wmo_code

Converter = Callable[[float, str], float]

# (canonical field, Open-Meteo variable, converter or None for unitless values
# such as percentages, bearings, indices and codes).
_FieldSpec = tuple[str, str, Converter | None]

_CURRENT_SPECS: tuple[_FieldSpec, ...] = (
    ("temperature_c", "temperature_2m", to_celsius),
    ("apparent_temperature_c", "apparent_temperature", to_celsius),
    ("relative_humidity_pct", "relative_humidity_2m", None),
    ("pressure_msl_hpa", "pressure_msl", to_hectopascals),
    ("surface_pressure_hpa", "surface_pressure", to_hectopascals),
    ("wind_speed_ms", "wind_speed_10m", to_metres_per_second),
    ("wind_gust_ms", "wind_gusts_10m", to_metres_per_second),
    ("wind_direction_deg", "wind_direction_10m", None),
    ("precipitation_mm", "precipitation", to_millimetres),
    ("cloud_cover_pct", "cloud_cover", None),
)

_HOURLY_SPECS: tuple[_FieldSpec, ...] = (
    ("temperature_c", "temperature_2m", to_celsius),
    ("apparent_temperature_c", "apparent_temperature", to_celsius),
    ("dew_point_c", "dew_point_2m", to_celsius),
    ("relative_humidity_pct", "relative_humidity_2m", None),
    ("precipitation_mm", "precipitation", to_millimetres),
    ("precipitation_probability_pct", "precipitation_probability", None),
    ("pressure_msl_hpa", "pressure_msl", to_hectopascals),
    ("wind_speed_ms", "wind_speed_10m", to_metres_per_second),
    ("wind_gust_ms", "wind_gusts_10m", to_metres_per_second),
    ("wind_direction_deg", "wind_direction_10m", None),
    ("cloud_cover_pct", "cloud_cover", None),
    ("visibility_m", "visibility", to_metres),
    ("uv_index", "uv_index", None),
)

_DAILY_SPECS: tuple[_FieldSpec, ...] = (
    ("temperature_min_c", "temperature_2m_min", to_celsius),
    ("temperature_max_c", "temperature_2m_max", to_celsius),
    ("apparent_temperature_min_c", "apparent_temperature_min", to_celsius),
    ("apparent_temperature_max_c", "apparent_temperature_max", to_celsius),
    ("precipitation_sum_mm", "precipitation_sum", to_millimetres),
    ("precipitation_hours", "precipitation_hours", None),
    ("precipitation_probability_max_pct", "precipitation_probability_max", None),
    ("wind_speed_max_ms", "wind_speed_10m_max", to_metres_per_second),
    ("wind_gust_max_ms", "wind_gusts_10m_max", to_metres_per_second),
    ("wind_direction_dominant_deg", "wind_direction_10m_dominant", None),
    ("uv_index_max", "uv_index_max", None),
)


class NormalizationError(WeatherProviderError):
    """A payload could not be mapped onto the canonical model."""

    code = "WEATHER_DATA_NORMALIZATION_FAILED"
    message = "Upstream weather data could not be normalised."


def _tzinfo(utc_offset_seconds: int) -> timezone:
    return timezone(timedelta(seconds=utc_offset_seconds))


def _parse_instant(raw: Any, utc_offset_seconds: int) -> datetime | None:
    """Parse a local ISO-8601 timestamp into an aware UTC datetime."""
    if not isinstance(raw, str) or not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_tzinfo(utc_offset_seconds))
    return parsed.astimezone(timezone.utc)


def _parse_date(raw: Any) -> date | None:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return None


def _as_number(value: Any) -> float | None:
    """Numeric values only; JSON nulls and unexpected types become missing."""
    if isinstance(value, bool) or value is None:
        return None
    return float(value) if isinstance(value, (int, float)) else None


def _convert(
    field: str, variable: str, value: Any, units: Mapping[str, str], converter: Converter | None
) -> float | None:
    number = _as_number(value)
    if number is None:
        return None
    if converter is None:
        return number

    declared_unit = units.get(variable)
    # Unit conversion introduces binary float noise (11.1 km/h becomes
    # 3.0833333333333335 m/s). Four decimals is far finer than any instrument
    # or model resolution, and keeps the JSON readable.
    if not declared_unit:
        raise NormalizationError(
            f"Open-Meteo returned {variable!r} without declaring its unit.",
            provider_id="open-meteo",
            details={"variable": variable, "field": field},
        )
    try:
        return round(converter(number, declared_unit), 4)
    except UnknownUnitError as exc:
        raise NormalizationError(
            f"Open-Meteo declared an unsupported unit for {variable!r}: {declared_unit!r}.",
            provider_id="open-meteo",
            details={"variable": variable, "field": field, "unit": declared_unit},
        ) from exc


def _extract(
    values: Mapping[str, Any], units: Mapping[str, str], specs: Sequence[_FieldSpec]
) -> dict[str, float | None]:
    return {
        field: _convert(field, variable, values.get(variable), units, converter)
        for field, variable, converter in specs
    }


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    return int(value) if isinstance(value, (int, float)) else None


def _as_bool(value: Any) -> bool | None:
    number = _as_int(value)
    return None if number is None else bool(number)


def _condition_fields(raw_code: Any) -> dict[str, Any]:
    code = _as_int(raw_code)
    condition, description = interpret_wmo_code(code)
    return {
        "wmo_code": code,
        "condition": condition,
        "condition_description": description,
    }


def build_location(payload: Any) -> Location:
    """Location described by the payload itself.

    These are the coordinates of the grid point the provider actually served,
    which may differ slightly from the point that was requested. Place-name
    labels are not a provider concern; the service layer merges those in from
    geocoding after the fact.
    """
    return Location(
        coordinates=Coordinates(
            latitude=payload.latitude,
            longitude=payload.longitude,
            elevation_m=payload.elevation,
        ),
        timezone=payload.timezone,
    )


def build_provenance(
    *, fetched_at: datetime, source_url: str, metadata: Any
) -> DataProvenance:
    """Stamp a record with where and when it came from."""
    return DataProvenance(
        provider_id=metadata.provider_id,
        provider_name=metadata.name,
        model=metadata.model,
        fetched_at=fetched_at,
        source_url=source_url,
        license=metadata.license,
        attribution=metadata.attribution,
    )


def normalize_current(
    payload: Any, *, provenance: DataProvenance
) -> WeatherReport:
    """Map the ``current`` block onto a :class:`WeatherReport`."""
    if not payload.current:
        raise NormalizationError(
            "Open-Meteo response contained no current conditions.",
            provider_id="open-meteo",
        )

    observed_at = _parse_instant(payload.current.get("time"), payload.utc_offset_seconds)
    if observed_at is None:
        raise NormalizationError(
            "Open-Meteo current conditions had no usable timestamp.",
            provider_id="open-meteo",
            details={"time": payload.current.get("time")},
        )

    measurements = _extract(payload.current, payload.current_units, _CURRENT_SPECS)
    current = CurrentWeather(
        observed_at=observed_at,
        is_day=_as_bool(payload.current.get("is_day")),
        **measurements,
        **_condition_fields(payload.current.get("weather_code")),
    )
    return WeatherReport(
        location=build_location(payload),
        current=current,
        provenance=provenance,
    )


def _column(block: Mapping[str, Any], variable: str, index: int) -> Any:
    """Read one cell from a column-oriented series, tolerating ragged arrays."""
    column = block.get(variable)
    if isinstance(column, list) and index < len(column):
        return column[index]
    return None


def _normalize_hourly(payload: Any) -> list[HourlyForecastPoint]:
    block = payload.hourly or {}
    times = block.get("time") or []
    points: list[HourlyForecastPoint] = []

    for index, raw_time in enumerate(times):
        valid_at = _parse_instant(raw_time, payload.utc_offset_seconds)
        if valid_at is None:
            continue
        row = {
            variable: _column(block, variable, index)
            for _, variable, _ in _HOURLY_SPECS
        }
        points.append(
            HourlyForecastPoint(
                valid_at=valid_at,
                is_day=_as_bool(_column(block, "is_day", index)),
                **_extract(row, payload.hourly_units, _HOURLY_SPECS),
                **_condition_fields(_column(block, "weather_code", index)),
            )
        )
    return points


def _normalize_daily(payload: Any) -> list[DailyForecastPoint]:
    block = payload.daily or {}
    dates = block.get("time") or []
    points: list[DailyForecastPoint] = []

    for index, raw_date in enumerate(dates):
        day = _parse_date(raw_date)
        if day is None:
            continue
        row = {
            variable: _column(block, variable, index) for _, variable, _ in _DAILY_SPECS
        }
        points.append(
            DailyForecastPoint(
                date=day,
                sunrise=_parse_instant(
                    _column(block, "sunrise", index), payload.utc_offset_seconds
                ),
                sunset=_parse_instant(
                    _column(block, "sunset", index), payload.utc_offset_seconds
                ),
                **_extract(row, payload.daily_units, _DAILY_SPECS),
                **_condition_fields(_column(block, "weather_code", index)),
            )
        )
    return points


def normalize_forecast(payload: Any, *, provenance: DataProvenance) -> Forecast:
    """Map the ``hourly`` and ``daily`` blocks onto a :class:`Forecast`.

    The column-oriented upstream arrays are transposed into records here, so
    every consumer above works with points rather than parallel lists.
    """
    forecast = Forecast(
        location=build_location(payload),
        hourly=_normalize_hourly(payload),
        daily=_normalize_daily(payload),
        provenance=provenance,
    )
    if not forecast.hourly and not forecast.daily:
        raise NormalizationError(
            "Open-Meteo response contained no forecast series.",
            provider_id="open-meteo",
        )
    return forecast
