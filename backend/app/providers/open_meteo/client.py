"""Open-Meteo HTTP access and raw payload schema.

This module knows Open-Meteo's wire format and nothing about the canonical
domain model. It is the only place that names Open-Meteo variables such as
``temperature_2m`` or ``wind_gusts_10m``.

The payload is parsed into a typed model rather than passed around as a bare
dict, so a change in the upstream shape surfaces here as a validation error
instead of a ``KeyError`` three layers away.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.providers.http import UpstreamHttpClient

PROVIDER_ID = "open-meteo"

#: How far back the forecast endpoint will serve past hours. Open-Meteo's
#: documented ceiling; a larger value is rejected upstream.
MAX_PAST_DAYS = 92

#: Variables requested for current conditions.
CURRENT_VARIABLES: tuple[str, ...] = (
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "is_day",
    "precipitation",
    "weather_code",
    "cloud_cover",
    "pressure_msl",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
)

#: Variables requested for the hourly series.
HOURLY_VARIABLES: tuple[str, ...] = (
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation_probability",
    "precipitation",
    "weather_code",
    "pressure_msl",
    "cloud_cover",
    "visibility",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "uv_index",
    "is_day",
)

#: Variables requested for the daily series.
DAILY_VARIABLES: tuple[str, ...] = (
    "weather_code",
    "temperature_2m_max",
    "temperature_2m_min",
    "apparent_temperature_max",
    "apparent_temperature_min",
    "sunrise",
    "sunset",
    "uv_index_max",
    "precipitation_sum",
    "precipitation_hours",
    "precipitation_probability_max",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
)


class OpenMeteoPayload(BaseModel):
    """A forecast-endpoint response.

    ``hourly`` and ``daily`` arrive as parallel arrays keyed by variable name —
    a column layout the normaliser transposes into records. Units are declared
    in the matching ``*_units`` map and are read rather than assumed.
    """

    model_config = ConfigDict(extra="ignore")

    latitude: float
    longitude: float
    elevation: float | None = None
    timezone: str | None = None
    timezone_abbreviation: str | None = None
    utc_offset_seconds: int = 0

    current: dict[str, Any] | None = None
    current_units: dict[str, str] = Field(default_factory=dict)
    hourly: dict[str, list[Any]] | None = None
    hourly_units: dict[str, str] = Field(default_factory=dict)
    daily: dict[str, list[Any]] | None = None
    daily_units: dict[str, str] = Field(default_factory=dict)


class OpenMeteoClient:
    """Typed access to the Open-Meteo forecast endpoint.

    The shared connection pool is injected, which is also what lets tests drive
    the client through ``httpx.MockTransport`` with no network access.
    """

    def __init__(self, http: UpstreamHttpClient, *, forecast_url: str) -> None:
        self._http = http
        self._forecast_url = forecast_url

    @property
    def forecast_url(self) -> str:
        return self._forecast_url

    async def fetch(
        self,
        *,
        latitude: float,
        longitude: float,
        current: bool = False,
        hourly: bool = False,
        daily: bool = False,
        forecast_days: int | None = None,
        past_days: int | None = None,
    ) -> OpenMeteoPayload:
        """Request the given series for one point.

        Raises:
            WeatherProviderError: transport failure, error status, or a payload
                that does not match the expected shape.
        """
        params: dict[str, Any] = {
            "latitude": f"{latitude:.6f}",
            "longitude": f"{longitude:.6f}",
            # Local times plus an explicit offset; the normaliser makes them
            # timezone-aware rather than leaving naive strings in the model.
            "timezone": "auto",
        }
        if current:
            params["current"] = ",".join(CURRENT_VARIABLES)
        if hourly:
            params["hourly"] = ",".join(HOURLY_VARIABLES)
        if daily:
            params["daily"] = ",".join(DAILY_VARIABLES)
        if forecast_days is not None:
            params["forecast_days"] = forecast_days
        # Past hours arrive in the same hourly block as the forecast, from
        # the same models, so nothing downstream needs to know which is which.
        if past_days is not None:
            params["past_days"] = min(past_days, MAX_PAST_DAYS)

        body = await self._http.get_json(self._forecast_url, params=params)
        return OpenMeteoPayload.model_validate(body)
