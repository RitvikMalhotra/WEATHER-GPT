"""The canonical meteorological model.

This package is the contract between the providers below it and everything
above it. It is pure: Pydantic models, enums and arithmetic, with no I/O, no
HTTP and no knowledge of any specific data source. A new provider is correct
when it can produce these types; a consumer is safe when it reads only these
types.
"""

from app.domain.forecast import DailyForecastPoint, Forecast, HourlyForecastPoint
from app.domain.location import Coordinates, Location
from app.domain.provenance import DataProvenance
from app.domain.weather import (
    CurrentWeather,
    WeatherCondition,
    WeatherReport,
    interpret_wmo_code,
)

__all__ = [
    "Coordinates",
    "CurrentWeather",
    "DailyForecastPoint",
    "DataProvenance",
    "Forecast",
    "HourlyForecastPoint",
    "Location",
    "WeatherCondition",
    "WeatherReport",
    "interpret_wmo_code",
]
