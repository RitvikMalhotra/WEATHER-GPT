"""Unit conversion into the canonical system."""

from __future__ import annotations

import pytest

from app.domain.units import (
    UnknownUnitError,
    to_celsius,
    to_hectopascals,
    to_metres,
    to_metres_per_second,
    to_millimetres,
)


@pytest.mark.parametrize(
    ("value", "unit", "expected"),
    [
        (21.0, "°C", 21.0),
        (21.0, "celsius", 21.0),
        (69.8, "°F", 21.0),
        (294.15, "K", 21.0),
        (-40.0, "°F", -40.0),  # the crossover point
    ],
)
def test_temperatures_convert_to_celsius(value, unit, expected):
    assert to_celsius(value, unit) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize(
    ("value", "unit", "expected"),
    [
        (10.0, "m/s", 10.0),
        (36.0, "km/h", 10.0),
        (10.0, "mph", 4.4704),
        (10.0, "kn", 5.14444),
    ],
)
def test_speeds_convert_to_metres_per_second(value, unit, expected):
    assert to_metres_per_second(value, unit) == pytest.approx(expected, abs=1e-6)


def test_pressure_precipitation_and_distance_convert():
    assert to_hectopascals(1013.25, "hPa") == pytest.approx(1013.25)
    assert to_hectopascals(101.325, "kPa") == pytest.approx(1013.25)
    assert to_hectopascals(29.92, "inHg") == pytest.approx(1013.21, abs=0.01)

    assert to_millimetres(1.0, "inch") == pytest.approx(25.4)
    assert to_millimetres(2.5, "cm") == pytest.approx(25.0)

    assert to_metres(1.0, "km") == pytest.approx(1000.0)
    assert to_metres(1.0, "mi") == pytest.approx(1609.344)


def test_unit_matching_ignores_case_and_whitespace():
    assert to_celsius(69.8, "  Fahrenheit ") == pytest.approx(21.0)
    assert to_metres_per_second(36.0, "KM/H") == pytest.approx(10.0)


def test_an_unrecognised_unit_is_an_error_not_a_silent_passthrough():
    """Guessing here would corrupt every downstream value."""
    with pytest.raises(UnknownUnitError) as exc_info:
        to_metres_per_second(10.0, "furlongs/fortnight")

    assert exc_info.value.quantity == "speed"
    assert exc_info.value.raw_unit == "furlongs/fortnight"
