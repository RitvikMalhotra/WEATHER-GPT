"""Canonical forecast model.

Two resolutions, deliberately kept as separate series rather than nested:
hourly points answer "when will it rain this afternoon", daily points answer
"what does the week look like". Consumers ask for one, the other, or both, and
the provider is asked only for what was requested.

Same unit discipline as :mod:`app.domain.weather`, and the same rule that a
missing value stays missing.
"""

from __future__ import annotations

# Imported under aliases: the daily model has a field literally named `date`,
# which would otherwise shadow the type in its own annotation.
from datetime import date as date_type
from datetime import datetime as datetime_type

from pydantic import BaseModel, ConfigDict, Field

from app.domain.location import Location
from app.domain.provenance import DataProvenance
from app.domain.weather import WeatherCondition


class HourlyForecastPoint(BaseModel):
    """Predicted conditions valid at a single hour."""

    valid_at: datetime_type = Field(description="Instant the prediction applies to (UTC).")
    temperature_c: float | None = Field(default=None, description="Air temperature at 2 m.")
    apparent_temperature_c: float | None = Field(
        default=None, description="Feels-like temperature."
    )
    relative_humidity_pct: float | None = Field(
        default=None, description="Relative humidity at 2 m."
    )
    dew_point_c: float | None = Field(default=None, description="Dew point at 2 m.")
    precipitation_mm: float | None = Field(
        default=None, description="Precipitation expected in the hour."
    )
    precipitation_probability_pct: float | None = Field(
        default=None, description="Probability of measurable precipitation."
    )
    wind_speed_ms: float | None = Field(default=None, description="Wind speed at 10 m.")
    wind_gust_ms: float | None = Field(default=None, description="Peak gust at 10 m.")
    wind_direction_deg: float | None = Field(
        default=None, description="Direction the wind blows from."
    )
    pressure_msl_hpa: float | None = Field(
        default=None, description="Pressure reduced to mean sea level."
    )
    cloud_cover_pct: float | None = Field(default=None, description="Total cloud cover.")
    visibility_m: float | None = Field(default=None, description="Horizontal visibility.")
    uv_index: float | None = Field(default=None, description="UV index.")
    condition: WeatherCondition = Field(
        default=WeatherCondition.UNKNOWN, description="Coarse condition category."
    )
    condition_description: str | None = Field(
        default=None, description="Precise present-weather description."
    )
    wmo_code: int | None = Field(default=None, description="Raw WMO 4677 code.")
    is_day: bool | None = Field(default=None, description="Daylight at this hour.")


class DailyForecastPoint(BaseModel):
    """Predicted conditions aggregated over one local calendar day."""

    date: date_type = Field(description="Local calendar date at the forecast location.")
    temperature_min_c: float | None = Field(default=None, description="Daily minimum.")
    temperature_max_c: float | None = Field(default=None, description="Daily maximum.")
    apparent_temperature_min_c: float | None = Field(
        default=None, description="Minimum feels-like temperature."
    )
    apparent_temperature_max_c: float | None = Field(
        default=None, description="Maximum feels-like temperature."
    )
    precipitation_sum_mm: float | None = Field(
        default=None, description="Total precipitation for the day."
    )
    precipitation_hours: float | None = Field(
        default=None, description="Hours with measurable precipitation."
    )
    precipitation_probability_max_pct: float | None = Field(
        default=None, description="Peak hourly probability of precipitation."
    )
    wind_speed_max_ms: float | None = Field(default=None, description="Maximum wind speed.")
    wind_gust_max_ms: float | None = Field(default=None, description="Maximum gust.")
    wind_direction_dominant_deg: float | None = Field(
        default=None, description="Dominant wind direction."
    )
    uv_index_max: float | None = Field(default=None, description="Peak UV index.")
    sunrise: datetime_type | None = Field(default=None, description="Sunrise instant.")
    sunset: datetime_type | None = Field(default=None, description="Sunset instant.")
    condition: WeatherCondition = Field(
        default=WeatherCondition.UNKNOWN, description="Representative condition."
    )
    condition_description: str | None = Field(
        default=None, description="Precise present-weather description."
    )
    wmo_code: int | None = Field(default=None, description="Raw WMO 4677 code.")


class Forecast(BaseModel):
    """A forecast for one location, with full provenance."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "location": {
                    "coordinates": {"latitude": 28.6139, "longitude": 77.209},
                    "name": "New Delhi",
                    "timezone": "Asia/Kolkata",
                },
                "daily": [
                    {
                        "date": "2026-09-04",
                        "temperature_min_c": 26.1,
                        "temperature_max_c": 34.8,
                        "precipitation_sum_mm": 4.2,
                        "condition": "rain_showers",
                    }
                ],
                "hourly": [],
                "provenance": {
                    "provider_id": "open-meteo",
                    "provider_name": "Open-Meteo",
                    "fetched_at": "2026-09-04T07:30:12Z",
                },
            }
        }
    )

    location: Location = Field(description="Location the forecast applies to.")
    hourly: list[HourlyForecastPoint] = Field(
        default_factory=list, description="Hourly series, empty when not requested."
    )
    daily: list[DailyForecastPoint] = Field(
        default_factory=list, description="Daily series, empty when not requested."
    )
    provenance: DataProvenance = Field(description="Where the data came from.")
