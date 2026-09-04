"""IMD payloads to the canonical domain model.

Field names are IMD's, taken from its published API reference. Units are the
ones IMD documents — temperature °C, wind km/h, pressure hPa, rainfall mm — and
the conversions to canonical SI happen here, in the only module that knows IMD's
wire format.

Weather codes are deliberately not mapped to WMO 4677. IMD publishes its own
01-99 code list with different meanings; treating one as the other would put a
wrong condition on a real observation. The condition comes from IMD's own text
where it exists, and is otherwise left unknown.
"""

from __future__ import annotations

from datetime import date as date_type
from datetime import datetime, time, timedelta, timezone
from typing import Any

from app.domain.forecast import DailyForecastPoint, Forecast
from app.domain.location import Coordinates, Location
from app.domain.provenance import DataProvenance
from app.domain.weather import CurrentWeather, WeatherCondition, WeatherReport

#: IMD reports in km/h; the canonical model is m/s.
_KMH_TO_MS = 1.0 / 3.6
#: IMD timestamps are IST unless the field says UTC.
_IST = timezone(timedelta(hours=5, minutes=30))


class ImdMappingError(ValueError):
    """An IMD record could not be mapped onto the canonical model."""


def _number(value: Any) -> float | None:
    """IMD sends numbers as strings, and absences as '', 'NA' or '-'."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip()
    if not text or text.upper() in {"NA", "N/A", "-", "--", "NIL"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _first(record: dict[str, Any], *names: str) -> Any:
    """IMD field names vary in spacing and case between products."""
    normalised = {str(k).strip().casefold().replace(" ", "_"): v for k, v in record.items()}
    for name in names:
        key = name.strip().casefold().replace(" ", "_")
        if key in normalised:
            return normalised[key]
    return None


def condition_from_text(description: str | None) -> WeatherCondition:
    """Classify from IMD's own wording, conservatively.

    Anything not clearly recognisable stays UNKNOWN: the description is shown
    to the user verbatim either way, so a guess buys nothing and can mislead.
    """
    if not description:
        return WeatherCondition.UNKNOWN
    text = description.casefold()
    if "thunder" in text and "hail" in text:
        return WeatherCondition.THUNDERSTORM_WITH_HAIL
    if "thunder" in text or "squall" in text:
        return WeatherCondition.THUNDERSTORM
    if "snow" in text:
        return WeatherCondition.SNOW
    if "drizzle" in text:
        return WeatherCondition.DRIZZLE
    if "shower" in text:
        return WeatherCondition.RAIN_SHOWERS
    if "rain" in text:
        return WeatherCondition.RAIN
    if "fog" in text or "mist" in text:
        return WeatherCondition.FOG
    if "overcast" in text:
        return WeatherCondition.OVERCAST
    if "partly" in text or "scattered" in text:
        return WeatherCondition.PARTLY_CLOUDY
    if "cloud" in text:
        return WeatherCondition.OVERCAST
    if "clear" in text or "sunny" in text:
        return WeatherCondition.CLEAR
    return WeatherCondition.UNKNOWN


def _observed_at(record: dict[str, Any]) -> datetime:
    """Observation instant, from IMD's documented UTC date and time fields."""
    day = _text(_first(record, "Date of Observation", "Date_of_Observation", "Date"))
    clock = _text(_first(record, "Time of Observation", "Time_of_Observation", "Time"))
    if not day:
        raise ImdMappingError("IMD record has no observation date.")
    try:
        parsed_day = date_type.fromisoformat(day[:10])
    except ValueError as exc:
        raise ImdMappingError(f"Unreadable IMD observation date: {day!r}.") from exc

    hour, minute = 0, 0
    if clock:
        digits = clock.replace(":", "").strip()
        if digits.isdigit() and len(digits) >= 3:
            hour, minute = int(digits[:-2]), int(digits[-2:])
        elif ":" in clock:
            parts = clock.split(":")
            hour, minute = int(parts[0]), int(parts[1][:2])
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ImdMappingError(f"Unreadable IMD observation time: {clock!r}.")
    # The reference documents this field as UTC.
    return datetime.combine(parsed_day, time(hour, minute), tzinfo=timezone.utc)


def station_location(record: dict[str, Any]) -> Location | None:
    """Coordinates IMD publishes for a station, when it publishes them."""
    latitude = _number(_first(record, "Latitude", "latitude", "lat"))
    longitude = _number(_first(record, "Longitude", "longitude", "long", "lon"))
    if latitude is None or longitude is None:
        return None
    if not (-90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0):
        return None
    return Location(
        coordinates=Coordinates(latitude=latitude, longitude=longitude),
        name=_text(_first(record, "Station_Name", "Station", "Station Name", "City")),
        country="India",
        country_code="IN",
        timezone="Asia/Kolkata",
    )


def map_current(
    record: dict[str, Any], *, location: Location, provenance: DataProvenance
) -> WeatherReport:
    """Map a ``current_wx`` record onto a :class:`WeatherReport`."""
    description = _text(_first(record, "Weather Description", "Weather_Description"))
    wind_kmh = _number(_first(record, "Wind Speed", "Wind_Speed"))
    temperature = _number(_first(record, "Temperature", "Temp"))
    if temperature is None:
        raise ImdMappingError("IMD observation carries no temperature.")

    current = CurrentWeather(
        observed_at=_observed_at(record),
        temperature_c=temperature,
        relative_humidity_pct=_number(_first(record, "Humidity", "Relative_Humidity")),
        pressure_msl_hpa=_number(_first(record, "M.S.L.P", "MSLP", "M_S_L_P")),
        wind_speed_ms=(
            round(wind_kmh * _KMH_TO_MS, 4) if wind_kmh is not None else None
        ),
        wind_direction_deg=_number(_first(record, "Wind Direction", "Wind_Direction")),
        precipitation_mm=_number(
            _first(record, "Last 24 hrs Rainfall", "Last_24_hrs_Rainfall")
        ),
        # Nebulosity is oktas (0-8); the canonical field is a percentage.
        cloud_cover_pct=_okta_to_percent(
            _number(_first(record, "Nebulosity", "Cloud_Cover"))
        ),
        condition=condition_from_text(description),
        condition_description=description,
    )
    return WeatherReport(location=location, current=current, provenance=provenance)


def _okta_to_percent(oktas: float | None) -> float | None:
    if oktas is None or not (0.0 <= oktas <= 8.0):
        return None
    return round(oktas / 8.0 * 100.0, 1)


def map_city_forecast(
    record: dict[str, Any],
    *,
    location: Location,
    provenance: DataProvenance,
    days: int,
) -> Forecast:
    """Map a ``cityforecast`` record onto a :class:`Forecast`.

    IMD returns one row holding today plus days 2-7 as separate columns, so the
    columns are transposed into dated points.
    """
    issued = _text(_first(record, "Date"))
    try:
        first_day = date_type.fromisoformat(issued[:10]) if issued else None
    except ValueError:
        first_day = None
    if first_day is None:
        raise ImdMappingError("IMD forecast record has no usable date.")

    points: list[DailyForecastPoint] = []
    for offset in range(min(max(days, 1), 7)):
        if offset == 0:
            maximum = _number(_first(record, "Todays_Forecast_Max_Temp"))
            minimum = _number(_first(record, "Todays_Forecast_Min_temp"))
            summary = _text(_first(record, "Todays_Forecast"))
        else:
            index = offset + 1
            maximum = _number(_first(record, f"Day_{index}_Max_Temp"))
            minimum = _number(_first(record, f"Day_{index}_Min_temp"))
            summary = _text(_first(record, f"Day_{index}_Forecast"))
        if maximum is None and minimum is None and summary is None:
            continue
        points.append(
            DailyForecastPoint(
                date=first_day + timedelta(days=offset),
                temperature_max_c=maximum,
                temperature_min_c=minimum,
                condition=condition_from_text(summary),
                condition_description=summary,
            )
        )

    if not points:
        raise ImdMappingError("IMD forecast record contained no usable days.")
    return Forecast(location=location, daily=points, provenance=provenance)
