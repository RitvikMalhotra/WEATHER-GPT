"""Units of measure and conversion into the canonical system.

WeatherGPT stores and reasons about exactly one unit per quantity:

    temperature    degrees Celsius       (°C)
    speed          metres per second     (m/s)
    pressure       hectopascals          (hPa)
    precipitation  millimetres           (mm)
    distance       metres                (m)
    direction      degrees true          (°)

Providers disagree about units — Open-Meteo answers in km/h by default, the US
models are imperial, and a provider may silently change its defaults. So the
normalisation layer never *assumes* a unit: it reads the unit the provider
declares alongside each value and converts through the functions below. An
unrecognised unit is an error, not a value we quietly pass through.

This module is pure arithmetic: no I/O, no provider knowledge, no dependency on
the rest of the application.
"""

from __future__ import annotations

from enum import Enum


class UnknownUnitError(ValueError):
    """A provider declared a unit the normalisation layer cannot interpret."""

    def __init__(self, quantity: str, raw_unit: str) -> None:
        super().__init__(f"Unsupported {quantity} unit: {raw_unit!r}")
        self.quantity = quantity
        self.raw_unit = raw_unit


class TemperatureUnit(str, Enum):
    CELSIUS = "°C"
    FAHRENHEIT = "°F"
    KELVIN = "K"


class SpeedUnit(str, Enum):
    METRES_PER_SECOND = "m/s"
    KILOMETRES_PER_HOUR = "km/h"
    MILES_PER_HOUR = "mph"
    KNOTS = "kn"


class PressureUnit(str, Enum):
    HECTOPASCAL = "hPa"
    KILOPASCAL = "kPa"
    INCHES_OF_MERCURY = "inHg"


class PrecipitationUnit(str, Enum):
    MILLIMETRE = "mm"
    CENTIMETRE = "cm"
    INCH = "inch"


class DistanceUnit(str, Enum):
    METRE = "m"
    KILOMETRE = "km"
    FOOT = "ft"
    MILE = "mi"


# Provider payloads spell units inconsistently ("°C", "C", "celsius"). Aliases
# are matched case-insensitively after stripping whitespace.
_TEMPERATURE_ALIASES: dict[str, TemperatureUnit] = {
    "°c": TemperatureUnit.CELSIUS,
    "c": TemperatureUnit.CELSIUS,
    "celsius": TemperatureUnit.CELSIUS,
    "degc": TemperatureUnit.CELSIUS,
    "°f": TemperatureUnit.FAHRENHEIT,
    "f": TemperatureUnit.FAHRENHEIT,
    "fahrenheit": TemperatureUnit.FAHRENHEIT,
    "degf": TemperatureUnit.FAHRENHEIT,
    "k": TemperatureUnit.KELVIN,
    "kelvin": TemperatureUnit.KELVIN,
}

_SPEED_ALIASES: dict[str, SpeedUnit] = {
    "m/s": SpeedUnit.METRES_PER_SECOND,
    "ms": SpeedUnit.METRES_PER_SECOND,
    "mps": SpeedUnit.METRES_PER_SECOND,
    "km/h": SpeedUnit.KILOMETRES_PER_HOUR,
    "kmh": SpeedUnit.KILOMETRES_PER_HOUR,
    "kph": SpeedUnit.KILOMETRES_PER_HOUR,
    "mph": SpeedUnit.MILES_PER_HOUR,
    "mi/h": SpeedUnit.MILES_PER_HOUR,
    "kn": SpeedUnit.KNOTS,
    "kt": SpeedUnit.KNOTS,
    "kts": SpeedUnit.KNOTS,
    "knots": SpeedUnit.KNOTS,
}

_PRESSURE_ALIASES: dict[str, PressureUnit] = {
    "hpa": PressureUnit.HECTOPASCAL,
    "mbar": PressureUnit.HECTOPASCAL,
    "mb": PressureUnit.HECTOPASCAL,
    "kpa": PressureUnit.KILOPASCAL,
    "inhg": PressureUnit.INCHES_OF_MERCURY,
    "in hg": PressureUnit.INCHES_OF_MERCURY,
}

_PRECIPITATION_ALIASES: dict[str, PrecipitationUnit] = {
    "mm": PrecipitationUnit.MILLIMETRE,
    "cm": PrecipitationUnit.CENTIMETRE,
    "inch": PrecipitationUnit.INCH,
    "in": PrecipitationUnit.INCH,
}

_DISTANCE_ALIASES: dict[str, DistanceUnit] = {
    "m": DistanceUnit.METRE,
    "metre": DistanceUnit.METRE,
    "meter": DistanceUnit.METRE,
    "km": DistanceUnit.KILOMETRE,
    "ft": DistanceUnit.FOOT,
    "feet": DistanceUnit.FOOT,
    "mi": DistanceUnit.MILE,
}

_TO_CELSIUS = {
    TemperatureUnit.CELSIUS: lambda v: v,
    TemperatureUnit.FAHRENHEIT: lambda v: (v - 32.0) * 5.0 / 9.0,
    TemperatureUnit.KELVIN: lambda v: v - 273.15,
}

_TO_METRES_PER_SECOND = {
    SpeedUnit.METRES_PER_SECOND: 1.0,
    SpeedUnit.KILOMETRES_PER_HOUR: 1000.0 / 3600.0,
    SpeedUnit.MILES_PER_HOUR: 0.44704,
    SpeedUnit.KNOTS: 0.514444,
}

_TO_HECTOPASCAL = {
    PressureUnit.HECTOPASCAL: 1.0,
    PressureUnit.KILOPASCAL: 10.0,
    PressureUnit.INCHES_OF_MERCURY: 33.8638866667,
}

_TO_MILLIMETRE = {
    PrecipitationUnit.MILLIMETRE: 1.0,
    PrecipitationUnit.CENTIMETRE: 10.0,
    PrecipitationUnit.INCH: 25.4,
}

_TO_METRE = {
    DistanceUnit.METRE: 1.0,
    DistanceUnit.KILOMETRE: 1000.0,
    DistanceUnit.FOOT: 0.3048,
    DistanceUnit.MILE: 1609.344,
}


def _resolve(quantity: str, raw_unit: str, aliases: dict[str, object]) -> object:
    key = raw_unit.strip().lower()
    try:
        return aliases[key]
    except KeyError:
        raise UnknownUnitError(quantity, raw_unit) from None


def to_celsius(value: float, raw_unit: str) -> float:
    """Convert a temperature reported in ``raw_unit`` to degrees Celsius."""
    unit = _resolve("temperature", raw_unit, _TEMPERATURE_ALIASES)
    return _TO_CELSIUS[unit](value)  # type: ignore[index]


def to_metres_per_second(value: float, raw_unit: str) -> float:
    """Convert a speed reported in ``raw_unit`` to metres per second."""
    unit = _resolve("speed", raw_unit, _SPEED_ALIASES)
    return value * _TO_METRES_PER_SECOND[unit]  # type: ignore[index]


def to_hectopascals(value: float, raw_unit: str) -> float:
    """Convert a pressure reported in ``raw_unit`` to hectopascals."""
    unit = _resolve("pressure", raw_unit, _PRESSURE_ALIASES)
    return value * _TO_HECTOPASCAL[unit]  # type: ignore[index]


def to_millimetres(value: float, raw_unit: str) -> float:
    """Convert a precipitation depth reported in ``raw_unit`` to millimetres."""
    unit = _resolve("precipitation", raw_unit, _PRECIPITATION_ALIASES)
    return value * _TO_MILLIMETRE[unit]  # type: ignore[index]


def to_metres(value: float, raw_unit: str) -> float:
    """Convert a distance reported in ``raw_unit`` to metres."""
    unit = _resolve("distance", raw_unit, _DISTANCE_ALIASES)
    return value * _TO_METRE[unit]  # type: ignore[index]
