"""Deterministic alert rules.

A rule is a small, inspectable object: read one canonical variable, compare it
to an ordered ladder of thresholds, report the highest band it reaches. No rule
consults a model, a language model, or anything but the number in front of it.

**Sample windows.** A threshold is meaningless without knowing what period the
value covers — 50 mm of rain in an hour and 50 mm across a day are different
events. Each sample therefore declares its window, and each rule declares which
windows it applies to. A rule is never evaluated against a window it was not
calibrated for.

**On the default thresholds.** These are engineering defaults for a prototype.
They are *not* official safety thresholds, and an alert produced by them is a
WeatherGPT rule result, never an official warning — see
:class:`~app.domain.alert.AlertSourceType`. The daily rainfall bands are set at
the boundaries of commonly published 24-hour rainfall categories so the numbers
are at least familiar to Indian users, but reusing a number does not borrow the
issuing authority that goes with it. Every threshold is configurable, and any
deployment making real decisions must set its own from a sourced specification.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Sequence

from app.alerts.severity import SeverityBand, build_ladder, classify
from app.config.settings import Settings
from app.domain.alert import AlertKind, AlertSeverity, AlertType
from app.domain.forecast import DailyForecastPoint, HourlyForecastPoint
from app.domain.weather import CurrentWeather


class SampleWindow(str, Enum):
    """The period a sample's values describe."""

    #: A current-conditions reading. Instantaneous variables are exact;
    #: accumulations cover a short, provider-defined reporting interval.
    OBSERVATION = "observation"
    #: A forecast hour. Accumulations cover that hour.
    HOUR = "hour"
    #: A forecast day. Aggregates are the day's min/max/total.
    DAY = "day"


@dataclass(frozen=True, slots=True)
class EvaluationSample:
    """One point in time and space, flattened to the vocabulary rules use.

    Observations and the two forecast resolutions name the same physical
    quantities differently — ``temperature_c`` versus ``temperature_max_c``,
    ``precipitation_mm`` versus ``precipitation_sum_mm``. Reconciling that here
    keeps every rule written against a single set of names, and keeps the
    daily/hourly distinction from leaking into rule bodies.
    """

    kind: AlertKind
    window: SampleWindow
    valid_from: datetime
    valid_until: datetime

    temperature_c: float | None = None
    apparent_temperature_c: float | None = None
    relative_humidity_pct: float | None = None
    wind_speed_ms: float | None = None
    wind_gust_ms: float | None = None
    precipitation_mm: float | None = None
    precipitation_probability_pct: float | None = None

    def read(self, variable: str) -> float | None:
        """Read a variable, treating absent and non-numeric alike as missing.

        A rule cannot fire on a variable the provider did not report, and must
        not fire on one that arrived as the wrong type.
        """
        value = getattr(self, variable, None)
        if value is None or isinstance(value, bool):
            return None
        return float(value) if isinstance(value, (int, float)) else None

    def context(self, *variables: str) -> dict[str, Any]:
        """Supporting values recorded alongside the triggering one."""
        return {
            name: self.read(name)
            for name in variables
            if self.read(name) is not None
        }


@dataclass(frozen=True, slots=True)
class ThresholdRule:
    """Fires when one variable meets a configured threshold.

    Attributes:
        rule_id: stable identifier, stored on every alert it produces.
        alert_type: the hazard.
        variable: canonical field read from the sample.
        unit: unit of the variable and of every threshold.
        bands: ascending severity ladder.
        windows: sample windows this rule is calibrated for.
        context_variables: other values recorded as supporting evidence.
    """

    rule_id: str
    alert_type: AlertType
    variable: str
    unit: str
    bands: tuple[SeverityBand, ...]
    windows: frozenset[SampleWindow]
    title: str
    explanation: str
    comparison: str = ">="
    context_variables: tuple[str, ...] = ()

    @property
    def entry_threshold(self) -> float:
        """Lowest threshold that produces an alert at all."""
        return self.bands[0].threshold

    def applies_to(self, sample: EvaluationSample) -> bool:
        return sample.window in self.windows

    def evaluate(self, sample: EvaluationSample) -> "RuleResult | None":
        """Compare the sample. Returns ``None`` when the rule does not fire.

        Not firing covers three distinct cases, none of which is an error: the
        rule does not apply to this window, the variable is missing, or the
        value is below every band.
        """
        if not self.applies_to(sample):
            return None

        value = sample.read(self.variable)
        if value is None:
            return None

        band = classify(value, self.bands)
        if band is None:
            return None

        return RuleResult(
            rule=self,
            sample=sample,
            observed_value=value,
            threshold=band.threshold,
            severity=band.severity,
        )


@dataclass(frozen=True, slots=True)
class RuleResult:
    """A rule that fired, with the numbers that made it fire."""

    rule: ThresholdRule
    sample: EvaluationSample
    observed_value: float
    threshold: float
    severity: AlertSeverity

    @property
    def context(self) -> dict[str, Any]:
        return self.sample.context(*self.rule.context_variables)

    def describe(self) -> str:
        """A factual sentence: what was measured, against what, and how certain.

        Deliberately flat. Interpreting what it means for a person is the AI
        layer's job, and it can only do that honestly if the alert itself does
        not already editorialise.
        """
        observed = self.sample.kind is AlertKind.OBSERVED
        subject = "Observed" if observed else "Forecast"
        window = {
            SampleWindow.OBSERVATION: "",
            SampleWindow.HOUR: " over one hour",
            SampleWindow.DAY: " over one day",
        }[self.sample.window]
        return (
            f"{subject} {self.rule.explanation}{window} of "
            f"{self.observed_value:g} {self.rule.unit} meets the WeatherGPT "
            f"{self.severity.value} threshold of {self.threshold:g} "
            f"{self.rule.unit}."
        )


# --- Sample construction ------------------------------------------------------


def sample_from_observation(
    current: CurrentWeather, *, validity: timedelta
) -> EvaluationSample:
    """Build a sample from validated current conditions.

    ``validity`` is how long the reading is treated as describing the present;
    it becomes the alert's window and, with it, when the alert expires if the
    condition is not seen again.
    """
    return EvaluationSample(
        kind=AlertKind.OBSERVED,
        window=SampleWindow.OBSERVATION,
        valid_from=current.observed_at,
        valid_until=current.observed_at + validity,
        temperature_c=current.temperature_c,
        apparent_temperature_c=current.apparent_temperature_c,
        relative_humidity_pct=current.relative_humidity_pct,
        wind_speed_ms=current.wind_speed_ms,
        wind_gust_ms=current.wind_gust_ms,
        precipitation_mm=current.precipitation_mm,
        # Current conditions carry no probability: it already happened.
        precipitation_probability_pct=None,
    )


def sample_from_hourly(point: HourlyForecastPoint) -> EvaluationSample:
    """Build a sample from one forecast hour."""
    return EvaluationSample(
        kind=AlertKind.FORECAST_RISK,
        window=SampleWindow.HOUR,
        valid_from=point.valid_at,
        valid_until=point.valid_at + timedelta(hours=1),
        temperature_c=point.temperature_c,
        apparent_temperature_c=point.apparent_temperature_c,
        relative_humidity_pct=point.relative_humidity_pct,
        wind_speed_ms=point.wind_speed_ms,
        wind_gust_ms=point.wind_gust_ms,
        precipitation_mm=point.precipitation_mm,
        precipitation_probability_pct=point.precipitation_probability_pct,
    )


def sample_from_daily(point: DailyForecastPoint) -> EvaluationSample:
    """Build a sample from one forecast day.

    Daily aggregates map onto the shared vocabulary by their worst case — the
    day's maximum temperature, maximum wind, total rainfall — because that is
    what a threshold rule is asking about.
    """
    start = datetime.combine(point.date, datetime.min.time(), tzinfo=timezone.utc)
    return EvaluationSample(
        kind=AlertKind.FORECAST_RISK,
        window=SampleWindow.DAY,
        valid_from=start,
        valid_until=start + timedelta(days=1),
        temperature_c=point.temperature_max_c,
        apparent_temperature_c=point.apparent_temperature_max_c,
        wind_speed_ms=point.wind_speed_max_ms,
        wind_gust_ms=point.wind_gust_max_ms,
        precipitation_mm=point.precipitation_sum_mm,
        precipitation_probability_pct=point.precipitation_probability_max_pct,
    )


# --- The default rule set -----------------------------------------------------


def build_rules(settings: Settings) -> tuple[ThresholdRule, ...]:
    """Construct the active rule set from configuration.

    Every threshold comes from :class:`~app.config.settings.Settings`; none is
    written in this function as a literal. Adding a rule means adding an entry
    here plus its thresholds in settings — nothing else in the system changes.
    """
    return (
        ThresholdRule(
            rule_id="HEAVY_RAINFALL_HOURLY_01",
            alert_type=AlertType.HEAVY_RAINFALL,
            variable="precipitation_mm",
            unit="mm",
            windows=frozenset({SampleWindow.HOUR}),
            bands=build_ladder(
                (settings.HEAVY_RAINFALL_THRESHOLD, AlertSeverity.WATCH),
                (settings.HEAVY_RAINFALL_WARNING_THRESHOLD, AlertSeverity.WARNING),
                (settings.HEAVY_RAINFALL_SEVERE_THRESHOLD, AlertSeverity.SEVERE),
            ),
            title="Heavy rainfall",
            explanation="rainfall",
            context_variables=("precipitation_probability_pct", "wind_speed_ms"),
        ),
        ThresholdRule(
            rule_id="HEAVY_RAINFALL_DAILY_01",
            alert_type=AlertType.HEAVY_RAINFALL,
            variable="precipitation_mm",
            unit="mm",
            windows=frozenset({SampleWindow.DAY}),
            bands=build_ladder(
                (settings.HEAVY_RAINFALL_DAILY_THRESHOLD, AlertSeverity.WATCH),
                (
                    settings.HEAVY_RAINFALL_DAILY_WARNING_THRESHOLD,
                    AlertSeverity.WARNING,
                ),
                (settings.HEAVY_RAINFALL_DAILY_SEVERE_THRESHOLD, AlertSeverity.SEVERE),
                (settings.HEAVY_RAINFALL_DAILY_EXTREME_THRESHOLD, AlertSeverity.EXTREME),
            ),
            title="Heavy rainfall",
            explanation="rainfall accumulation",
            context_variables=("precipitation_probability_pct",),
        ),
        ThresholdRule(
            rule_id="EXTREME_HEAT_01",
            alert_type=AlertType.EXTREME_HEAT,
            variable="temperature_c",
            unit="°C",
            windows=frozenset(
                {SampleWindow.OBSERVATION, SampleWindow.HOUR, SampleWindow.DAY}
            ),
            bands=build_ladder(
                (settings.EXTREME_HEAT_THRESHOLD, AlertSeverity.WATCH),
                (settings.EXTREME_HEAT_WARNING_THRESHOLD, AlertSeverity.WARNING),
                (settings.EXTREME_HEAT_SEVERE_THRESHOLD, AlertSeverity.SEVERE),
            ),
            title="Extreme heat",
            explanation="air temperature",
            # Dry-bulb temperature alone is not a heat-health model. Humidity
            # and the apparent temperature are recorded so a later rule can use
            # them, and so an explanation can qualify what this rule ignores.
            context_variables=("apparent_temperature_c", "relative_humidity_pct"),
        ),
        ThresholdRule(
            rule_id="HIGH_WIND_01",
            alert_type=AlertType.HIGH_WIND,
            variable="wind_speed_ms",
            unit="m/s",
            windows=frozenset(
                {SampleWindow.OBSERVATION, SampleWindow.HOUR, SampleWindow.DAY}
            ),
            bands=build_ladder(
                (settings.HIGH_WIND_THRESHOLD, AlertSeverity.WATCH),
                (settings.HIGH_WIND_WARNING_THRESHOLD, AlertSeverity.WARNING),
                (settings.HIGH_WIND_SEVERE_THRESHOLD, AlertSeverity.SEVERE),
            ),
            title="High wind",
            explanation="wind speed",
            context_variables=("wind_gust_ms",),
        ),
        ThresholdRule(
            rule_id="HIGH_WIND_GUST_01",
            alert_type=AlertType.HIGH_WIND,
            variable="wind_gust_ms",
            unit="m/s",
            windows=frozenset(
                {SampleWindow.OBSERVATION, SampleWindow.HOUR, SampleWindow.DAY}
            ),
            bands=build_ladder(
                (settings.HIGH_WIND_GUST_THRESHOLD, AlertSeverity.WATCH),
                (settings.HIGH_WIND_GUST_WARNING_THRESHOLD, AlertSeverity.WARNING),
                (settings.HIGH_WIND_GUST_SEVERE_THRESHOLD, AlertSeverity.SEVERE),
            ),
            title="Damaging wind gusts",
            explanation="wind gust",
            context_variables=("wind_speed_ms",),
        ),
        ThresholdRule(
            rule_id="SEVERE_PRECIPITATION_PROBABILITY_01",
            alert_type=AlertType.SEVERE_PRECIPITATION_PROBABILITY,
            variable="precipitation_probability_pct",
            unit="%",
            # Only forecasts carry a probability; an observation is certain.
            windows=frozenset({SampleWindow.HOUR, SampleWindow.DAY}),
            bands=build_ladder(
                (settings.SEVERE_PRECIPITATION_PROBABILITY, AlertSeverity.WATCH),
                (
                    settings.SEVERE_PRECIPITATION_PROBABILITY_WARNING,
                    AlertSeverity.WARNING,
                ),
            ),
            title="High likelihood of precipitation",
            explanation="probability of precipitation",
            context_variables=("precipitation_mm",),
        ),
    )


def rules_for_window(
    rules: Sequence[ThresholdRule], window: SampleWindow
) -> list[ThresholdRule]:
    """Rules calibrated for a given window."""
    return [rule for rule in rules if window in rule.windows]
