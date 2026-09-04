"""Canonical current-conditions model.

Everything below is expressed in the units declared in :mod:`app.domain.units`.
Field names carry their unit as a suffix so a value can never be misread by a
downstream consumer — including the AI layer, which reads these records as
facts and must not have to guess whether a wind speed is m/s or km/h.

Every measurement is optional: providers differ in what they report, and a
missing value must stay missing rather than be defaulted to zero. Zero
precipitation and unknown precipitation are different facts.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from app.domain.location import Location
from app.domain.provenance import DataProvenance


class WeatherCondition(str, Enum):
    """Coarse condition category, suitable for icons and branching logic.

    Derived from the WMO 4677 present-weather code the provider reports. The
    precise wording lives alongside it in ``condition_description``.
    """

    CLEAR = "clear"
    MAINLY_CLEAR = "mainly_clear"
    PARTLY_CLOUDY = "partly_cloudy"
    OVERCAST = "overcast"
    FOG = "fog"
    DRIZZLE = "drizzle"
    FREEZING_DRIZZLE = "freezing_drizzle"
    RAIN = "rain"
    FREEZING_RAIN = "freezing_rain"
    SNOW = "snow"
    RAIN_SHOWERS = "rain_showers"
    SNOW_SHOWERS = "snow_showers"
    THUNDERSTORM = "thunderstorm"
    THUNDERSTORM_WITH_HAIL = "thunderstorm_with_hail"
    UNKNOWN = "unknown"


# WMO 4677 present-weather codes as emitted by Open-Meteo, GFS post-processing
# and most European services. Standard, so it lives in the domain rather than
# in any one provider.
_WMO_CODES: dict[int, tuple[WeatherCondition, str]] = {
    0: (WeatherCondition.CLEAR, "Clear sky"),
    1: (WeatherCondition.MAINLY_CLEAR, "Mainly clear"),
    2: (WeatherCondition.PARTLY_CLOUDY, "Partly cloudy"),
    3: (WeatherCondition.OVERCAST, "Overcast"),
    45: (WeatherCondition.FOG, "Fog"),
    48: (WeatherCondition.FOG, "Depositing rime fog"),
    51: (WeatherCondition.DRIZZLE, "Light drizzle"),
    53: (WeatherCondition.DRIZZLE, "Moderate drizzle"),
    55: (WeatherCondition.DRIZZLE, "Dense drizzle"),
    56: (WeatherCondition.FREEZING_DRIZZLE, "Light freezing drizzle"),
    57: (WeatherCondition.FREEZING_DRIZZLE, "Dense freezing drizzle"),
    61: (WeatherCondition.RAIN, "Slight rain"),
    63: (WeatherCondition.RAIN, "Moderate rain"),
    65: (WeatherCondition.RAIN, "Heavy rain"),
    66: (WeatherCondition.FREEZING_RAIN, "Light freezing rain"),
    67: (WeatherCondition.FREEZING_RAIN, "Heavy freezing rain"),
    71: (WeatherCondition.SNOW, "Slight snowfall"),
    73: (WeatherCondition.SNOW, "Moderate snowfall"),
    75: (WeatherCondition.SNOW, "Heavy snowfall"),
    77: (WeatherCondition.SNOW, "Snow grains"),
    80: (WeatherCondition.RAIN_SHOWERS, "Slight rain showers"),
    81: (WeatherCondition.RAIN_SHOWERS, "Moderate rain showers"),
    82: (WeatherCondition.RAIN_SHOWERS, "Violent rain showers"),
    85: (WeatherCondition.SNOW_SHOWERS, "Slight snow showers"),
    86: (WeatherCondition.SNOW_SHOWERS, "Heavy snow showers"),
    95: (WeatherCondition.THUNDERSTORM, "Thunderstorm"),
    96: (WeatherCondition.THUNDERSTORM_WITH_HAIL, "Thunderstorm with slight hail"),
    99: (WeatherCondition.THUNDERSTORM_WITH_HAIL, "Thunderstorm with heavy hail"),
}


def interpret_wmo_code(code: int | None) -> tuple[WeatherCondition, str | None]:
    """Map a WMO 4677 code to a condition category and its description.

    Unknown or absent codes degrade to ``UNKNOWN`` rather than raising: an
    unrecognised present-weather code should not cost us an otherwise valid
    temperature reading.
    """
    if code is None:
        return WeatherCondition.UNKNOWN, None
    return _WMO_CODES.get(code, (WeatherCondition.UNKNOWN, None))


class CurrentWeather(BaseModel):
    """Observed or analysed conditions at a single instant."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "observed_at": "2026-09-04T07:30:00Z",
                "temperature_c": 31.4,
                "apparent_temperature_c": 35.2,
                "relative_humidity_pct": 62.0,
                "pressure_msl_hpa": 1004.2,
                "wind_speed_ms": 3.6,
                "wind_direction_deg": 115.0,
                "precipitation_mm": 0.0,
                "cloud_cover_pct": 40.0,
                "condition": "partly_cloudy",
                "condition_description": "Partly cloudy",
                "wmo_code": 2,
                "is_day": True,
            }
        }
    )

    observed_at: datetime = Field(description="Instant the conditions apply to (UTC).")

    temperature_c: float | None = Field(
        default=None, description="Air temperature at 2 m, in degrees Celsius."
    )
    apparent_temperature_c: float | None = Field(
        default=None, description="Feels-like temperature, in degrees Celsius."
    )
    dew_point_c: float | None = Field(
        default=None, description="Dew point at 2 m, in degrees Celsius."
    )
    relative_humidity_pct: float | None = Field(
        default=None, description="Relative humidity at 2 m, as a percentage."
    )
    pressure_msl_hpa: float | None = Field(
        default=None, description="Pressure reduced to mean sea level, in hPa."
    )
    surface_pressure_hpa: float | None = Field(
        default=None, description="Pressure at station elevation, in hPa."
    )
    wind_speed_ms: float | None = Field(
        default=None, description="Wind speed at 10 m, in metres per second."
    )
    wind_gust_ms: float | None = Field(
        default=None, description="Peak gust at 10 m, in metres per second."
    )
    wind_direction_deg: float | None = Field(
        default=None, description="Direction the wind blows from, in degrees true."
    )
    precipitation_mm: float | None = Field(
        default=None, description="Precipitation accumulated in the reporting interval."
    )
    cloud_cover_pct: float | None = Field(
        default=None, description="Total cloud cover, as a percentage."
    )
    visibility_m: float | None = Field(
        default=None, description="Horizontal visibility, in metres."
    )
    uv_index: float | None = Field(default=None, description="UV index.")

    condition: WeatherCondition = Field(
        default=WeatherCondition.UNKNOWN, description="Coarse condition category."
    )
    condition_description: str | None = Field(
        default=None, description="Precise present-weather description."
    )
    wmo_code: int | None = Field(
        default=None, description="Raw WMO 4677 present-weather code."
    )
    is_day: bool | None = Field(
        default=None, description="True when the observation falls in daylight."
    )


class WeatherReport(BaseModel):
    """Current conditions for a location, with full provenance.

    This is the unit of exchange between the ingestion pipeline and everything
    above it: the API, the future alert engine, and the future AI layer.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "location": {
                    "coordinates": {"latitude": 28.6139, "longitude": 77.209},
                    "name": "New Delhi",
                    "country": "India",
                    "timezone": "Asia/Kolkata",
                },
                "current": {
                    "observed_at": "2026-09-04T07:30:00Z",
                    "temperature_c": 31.4,
                    "condition": "partly_cloudy",
                },
                "provenance": {
                    "provider_id": "open-meteo",
                    "provider_name": "Open-Meteo",
                    "fetched_at": "2026-09-04T07:30:12Z",
                },
            }
        }
    )

    location: Location = Field(description="Location the report applies to.")
    current: CurrentWeather = Field(description="Canonicalised current conditions.")
    provenance: DataProvenance = Field(description="Where the data came from.")
