"""The meteorological validation engine.

Data that arrives from an upstream source is not trusted. Before anything is
served, cached or (later) persisted, it passes through the checks below:

* **Range checks** — is each value physically possible on Earth? Bounds are set
  from observed extremes with headroom, so they reject corruption and unit
  mistakes without rejecting record-breaking weather.
* **Consistency checks** — do the values agree with each other? Dew point above
  air temperature, a gust weaker than the sustained wind, or a daily minimum
  above its maximum all indicate a broken feed regardless of whether each
  number is individually plausible.
* **Temporal checks** — is the record fresh, and is its timestamp sane?

Two severities, with different consequences:

    ERROR    the record is rejected; the pipeline falls back to another source
    WARNING  the record is served, and the anomaly is logged

That distinction matters. A unit conversion that silently produces 25 m/s
instead of 25 km/h is the kind of failure that makes a weather system dangerous
rather than merely wrong, so it must fail loudly. A slightly stale observation
is worth flagging but still worth serving.

The engine is pure and synchronous: no I/O, no provider knowledge. It validates
the canonical model, so every source is held to identical standards.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum

from app.domain.forecast import Forecast
from app.domain.weather import CurrentWeather, WeatherReport


class ValidationSeverity(str, Enum):
    """How the pipeline should react to an issue."""

    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One failed check."""

    field: str
    message: str
    severity: ValidationSeverity
    value: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "field": self.field,
            "message": self.message,
            "severity": self.severity.value,
            "value": self.value,
        }


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Outcome of validating one record."""

    issues: tuple[ValidationIssue, ...] = ()

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(i for i in self.issues if i.severity is ValidationSeverity.ERROR)

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(i for i in self.issues if i.severity is ValidationSeverity.WARNING)

    @property
    def is_valid(self) -> bool:
        """True when nothing disqualifying was found. Warnings still pass."""
        return not self.errors


# Physically plausible bounds. Records: -89.2 °C (Vostok, 1983) and 56.7 °C
# (Furnace Creek, 1913); 113 m/s gust (Barrow Island, 1996); 870 hPa (Tip,
# 1979) and 1084 hPa (Agata, 1968). Each is widened for model output.
_RANGES: dict[str, tuple[float, float]] = {
    "temperature_c": (-95.0, 60.0),
    "apparent_temperature_c": (-110.0, 80.0),
    "dew_point_c": (-100.0, 45.0),
    "relative_humidity_pct": (0.0, 100.0),
    "pressure_msl_hpa": (850.0, 1100.0),
    "surface_pressure_hpa": (300.0, 1100.0),
    "wind_speed_ms": (0.0, 120.0),
    "wind_gust_ms": (0.0, 150.0),
    "wind_direction_deg": (0.0, 360.0),
    "precipitation_mm": (0.0, 2000.0),
    "precipitation_probability_pct": (0.0, 100.0),
    "cloud_cover_pct": (0.0, 100.0),
    "visibility_m": (0.0, 100_000.0),
    "uv_index": (0.0, 20.0),
    # Daily aggregates.
    "temperature_min_c": (-95.0, 60.0),
    "temperature_max_c": (-95.0, 60.0),
    "apparent_temperature_min_c": (-110.0, 80.0),
    "apparent_temperature_max_c": (-110.0, 80.0),
    "precipitation_sum_mm": (0.0, 2000.0),
    "precipitation_hours": (0.0, 24.0),
    "precipitation_probability_max_pct": (0.0, 100.0),
    "wind_speed_max_ms": (0.0, 120.0),
    "wind_gust_max_ms": (0.0, 150.0),
    "wind_direction_dominant_deg": (0.0, 360.0),
    "uv_index_max": (0.0, 20.0),
}

#: Dew point may exceed temperature by this much before it counts as an error.
#: Small crossovers are a rounding artefact; large ones are a broken feed.
_DEW_POINT_TOLERANCE_C = 0.5

#: Gusts below sustained wind by more than this are inconsistent.
_GUST_TOLERANCE_MS = 0.5


class WeatherValidator:
    """Validates canonical records against physical and temporal expectations.

    Args:
        max_age: an observation older than this is stale. ``None`` disables the
            staleness check.
        max_clock_skew: how far into the future an observation timestamp may sit
            before it is treated as an error.
    """

    def __init__(
        self,
        *,
        max_age: timedelta | None = timedelta(hours=3),
        max_clock_skew: timedelta = timedelta(minutes=90),
    ) -> None:
        self._max_age = max_age
        self._max_clock_skew = max_clock_skew

    # --- Public API ---------------------------------------------------------

    def validate_current(
        self, report: WeatherReport, *, now: datetime | None = None
    ) -> ValidationResult:
        """Validate a current-conditions report."""
        now = now or datetime.now(timezone.utc)
        issues: list[ValidationIssue] = []
        current = report.current

        issues.extend(self._check_ranges(current, prefix="current"))
        issues.extend(self._check_consistency(current, prefix="current"))
        issues.extend(self._check_timestamp(current.observed_at, now, prefix="current"))

        return ValidationResult(tuple(issues))

    def validate_forecast(
        self, forecast: Forecast, *, now: datetime | None = None
    ) -> ValidationResult:
        """Validate every point in a forecast.

        A forecast with no points at all is an error; individual bad points are
        reported against their own index so a log line identifies the culprit.
        """
        now = now or datetime.now(timezone.utc)
        issues: list[ValidationIssue] = []

        if not forecast.hourly and not forecast.daily:
            issues.append(
                ValidationIssue(
                    field="forecast",
                    message="Forecast contains neither hourly nor daily points.",
                    severity=ValidationSeverity.ERROR,
                )
            )

        for index, point in enumerate(forecast.hourly):
            prefix = f"hourly[{index}]"
            issues.extend(self._check_ranges(point, prefix=prefix))
            issues.extend(self._check_consistency(point, prefix=prefix))

        for index, point in enumerate(forecast.daily):
            prefix = f"daily[{index}]"
            issues.extend(self._check_ranges(point, prefix=prefix))
            issues.extend(self._check_daily_consistency(point, prefix=prefix))

        return ValidationResult(tuple(issues))

    # --- Checks -------------------------------------------------------------

    def _check_ranges(self, record: object, *, prefix: str) -> list[ValidationIssue]:
        """Every known numeric field must sit inside its physical bounds."""
        issues: list[ValidationIssue] = []
        for field, (low, high) in _RANGES.items():
            value = getattr(record, field, None)
            if value is None:
                continue
            if not low <= value <= high:
                issues.append(
                    ValidationIssue(
                        field=f"{prefix}.{field}",
                        message=(
                            f"{value:g} is outside the plausible range "
                            f"[{low:g}, {high:g}]."
                        ),
                        severity=ValidationSeverity.ERROR,
                        value=float(value),
                    )
                )
        return issues

    def _check_consistency(self, record: object, *, prefix: str) -> list[ValidationIssue]:
        """Cross-field checks for an instantaneous record."""
        issues: list[ValidationIssue] = []

        temperature = getattr(record, "temperature_c", None)
        dew_point = getattr(record, "dew_point_c", None)
        if temperature is not None and dew_point is not None:
            excess = dew_point - temperature
            if excess > _DEW_POINT_TOLERANCE_C:
                issues.append(
                    ValidationIssue(
                        field=f"{prefix}.dew_point_c",
                        message=(
                            f"Dew point {dew_point:g} °C exceeds air temperature "
                            f"{temperature:g} °C by {excess:g} °C."
                        ),
                        severity=ValidationSeverity.ERROR,
                        value=float(dew_point),
                    )
                )

        wind = getattr(record, "wind_speed_ms", None)
        gust = getattr(record, "wind_gust_ms", None)
        if wind is not None and gust is not None and gust < wind - _GUST_TOLERANCE_MS:
            issues.append(
                ValidationIssue(
                    field=f"{prefix}.wind_gust_ms",
                    message=(
                        f"Gust {gust:g} m/s is below sustained wind {wind:g} m/s."
                    ),
                    severity=ValidationSeverity.WARNING,
                    value=float(gust),
                )
            )

        humidity = getattr(record, "relative_humidity_pct", None)
        precipitation = getattr(record, "precipitation_mm", None)
        if (
            humidity is not None
            and precipitation is not None
            and precipitation > 0.2
            and humidity < 30.0
        ):
            issues.append(
                ValidationIssue(
                    field=f"{prefix}.relative_humidity_pct",
                    message=(
                        f"{precipitation:g} mm of precipitation reported at "
                        f"{humidity:g}% relative humidity."
                    ),
                    severity=ValidationSeverity.WARNING,
                    value=float(humidity),
                )
            )
        return issues

    def _check_daily_consistency(
        self, record: object, *, prefix: str
    ) -> list[ValidationIssue]:
        """Cross-field checks for a daily aggregate."""
        issues: list[ValidationIssue] = []

        minimum = getattr(record, "temperature_min_c", None)
        maximum = getattr(record, "temperature_max_c", None)
        if minimum is not None and maximum is not None and minimum > maximum:
            issues.append(
                ValidationIssue(
                    field=f"{prefix}.temperature_min_c",
                    message=(
                        f"Daily minimum {minimum:g} °C exceeds maximum {maximum:g} °C."
                    ),
                    severity=ValidationSeverity.ERROR,
                    value=float(minimum),
                )
            )

        hours = getattr(record, "precipitation_hours", None)
        total = getattr(record, "precipitation_sum_mm", None)
        if hours is not None and total is not None and hours > 0 and total == 0:
            issues.append(
                ValidationIssue(
                    field=f"{prefix}.precipitation_sum_mm",
                    message=f"{hours:g} precipitation hours but zero total accumulation.",
                    severity=ValidationSeverity.WARNING,
                    value=0.0,
                )
            )
        return issues

    def _check_timestamp(
        self, observed_at: datetime, now: datetime, *, prefix: str
    ) -> list[ValidationIssue]:
        """Reject future observations; flag stale ones."""
        issues: list[ValidationIssue] = []
        age = now - observed_at

        if age < -self._max_clock_skew:
            issues.append(
                ValidationIssue(
                    field=f"{prefix}.observed_at",
                    message=(
                        f"Observation is dated {abs(age)} in the future, beyond the "
                        f"{self._max_clock_skew} skew allowance."
                    ),
                    severity=ValidationSeverity.ERROR,
                )
            )
        elif self._max_age is not None and age > self._max_age:
            issues.append(
                ValidationIssue(
                    field=f"{prefix}.observed_at",
                    message=f"Observation is {age} old, beyond the {self._max_age} limit.",
                    severity=ValidationSeverity.WARNING,
                )
            )
        return issues
