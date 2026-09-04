"""The deterministic rule engine.

Pure logic: no database, no HTTP, no clock beyond what the tests hand it. This
is the safety-critical decision — *did a threshold get crossed* — so it is
tested exhaustively, including the boundaries.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from app.alerts.engine import AlertEngine
from app.alerts.rules import (
    EvaluationSample,
    SampleWindow,
    ThresholdRule,
    build_rules,
    sample_from_daily,
    sample_from_hourly,
    sample_from_observation,
)
from app.alerts.severity import build_ladder, classify
from app.config.settings import Settings
from app.domain.alert import AlertKind, AlertSeverity, AlertType
from app.domain.forecast import DailyForecastPoint, Forecast, HourlyForecastPoint
from app.domain.location import Coordinates, Location
from app.domain.provenance import DataProvenance
from app.domain.weather import CurrentWeather, WeatherCondition, WeatherReport

NOW = datetime(2026, 9, 4, 7, 30, tzinfo=timezone.utc)
HYDERABAD = Coordinates(latitude=17.385, longitude=78.4867)


@pytest.fixture
def rules(settings: Settings):
    return build_rules(settings)


@pytest.fixture
def engine(rules):
    return AlertEngine(rules, observation_validity=timedelta(hours=1))


def _rule(rules, rule_id: str) -> ThresholdRule:
    return next(rule for rule in rules if rule.rule_id == rule_id)


def _sample(window: SampleWindow = SampleWindow.HOUR, **values) -> EvaluationSample:
    kind = (
        AlertKind.OBSERVED
        if window is SampleWindow.OBSERVATION
        else AlertKind.FORECAST_RISK
    )
    return EvaluationSample(
        kind=kind,
        window=window,
        valid_from=NOW,
        valid_until=NOW + timedelta(hours=1),
        **values,
    )


def _report(**current) -> WeatherReport:
    return WeatherReport(
        location=Location(coordinates=HYDERABAD, timezone="Asia/Kolkata"),
        current=CurrentWeather(
            **{
                "observed_at": NOW,
                "condition": WeatherCondition.CLEAR,
                **current,
            }
        ),
        provenance=DataProvenance(
            provider_id="open-meteo",
            provider_name="Open-Meteo",
            model="best_match",
            fetched_at=NOW,
        ),
    )


# --- Severity ladders ---------------------------------------------------------


def test_a_value_below_every_band_matches_nothing():
    ladder = build_ladder((10.0, AlertSeverity.WATCH), (20.0, AlertSeverity.WARNING))

    assert classify(5.0, ladder) is None


def test_the_highest_band_reached_wins():
    ladder = build_ladder(
        (10.0, AlertSeverity.WATCH),
        (20.0, AlertSeverity.WARNING),
        (30.0, AlertSeverity.SEVERE),
    )

    assert classify(25.0, ladder).severity is AlertSeverity.WARNING
    assert classify(35.0, ladder).severity is AlertSeverity.SEVERE


def test_a_value_exactly_on_a_threshold_is_inside_that_band():
    """Thresholds mean "this much or more"; excluding the boundary would make a
    rule configured at 50 silently ignore 50."""
    ladder = build_ladder((10.0, AlertSeverity.WATCH), (20.0, AlertSeverity.WARNING))

    assert classify(10.0, ladder).severity is AlertSeverity.WATCH
    assert classify(20.0, ladder).severity is AlertSeverity.WARNING


def test_a_ladder_out_of_threshold_order_is_rejected_at_construction():
    """An unordered ladder returns the wrong severity, silently."""
    with pytest.raises(ValueError, match="ascend by threshold"):
        build_ladder((20.0, AlertSeverity.WATCH), (10.0, AlertSeverity.WARNING))


def test_a_ladder_out_of_severity_order_is_rejected():
    with pytest.raises(ValueError, match="ascend by severity"):
        build_ladder((10.0, AlertSeverity.SEVERE), (20.0, AlertSeverity.WATCH))


def test_an_empty_ladder_is_rejected():
    with pytest.raises(ValueError):
        build_ladder()


def test_severity_ranks_are_ordered():
    ladder = [
        AlertSeverity.INFO,
        AlertSeverity.WATCH,
        AlertSeverity.WARNING,
        AlertSeverity.SEVERE,
        AlertSeverity.EXTREME,
    ]

    assert [severity.rank for severity in ladder] == sorted(
        severity.rank for severity in ladder
    )


# --- Threshold crossing -------------------------------------------------------


def test_below_the_threshold_no_rule_fires(rules, settings):
    rule = _rule(rules, "HIGH_WIND_01")
    sample = _sample(wind_speed_ms=settings.HIGH_WIND_THRESHOLD - 0.1)

    assert rule.evaluate(sample) is None


def test_exactly_at_the_threshold_the_rule_fires(rules, settings):
    rule = _rule(rules, "HIGH_WIND_01")
    sample = _sample(wind_speed_ms=settings.HIGH_WIND_THRESHOLD)

    result = rule.evaluate(sample)

    assert result is not None
    assert result.severity is AlertSeverity.WATCH
    assert result.threshold == pytest.approx(settings.HIGH_WIND_THRESHOLD)


def test_above_the_threshold_the_rule_fires(rules, settings):
    rule = _rule(rules, "HIGH_WIND_01")

    result = rule.evaluate(_sample(wind_speed_ms=settings.HIGH_WIND_THRESHOLD + 5))

    assert result is not None
    assert result.observed_value == pytest.approx(settings.HIGH_WIND_THRESHOLD + 5)


def test_severity_escalates_with_the_value(rules, settings):
    rule = _rule(rules, "HIGH_WIND_01")

    watch = rule.evaluate(_sample(wind_speed_ms=settings.HIGH_WIND_THRESHOLD))
    warning = rule.evaluate(_sample(wind_speed_ms=settings.HIGH_WIND_WARNING_THRESHOLD))
    severe = rule.evaluate(_sample(wind_speed_ms=settings.HIGH_WIND_SEVERE_THRESHOLD))

    assert watch.severity is AlertSeverity.WATCH
    assert warning.severity is AlertSeverity.WARNING
    assert severe.severity is AlertSeverity.SEVERE


def test_a_missing_variable_cannot_fire_a_rule(rules):
    """A value the provider did not report is not a value below the threshold."""
    rule = _rule(rules, "HIGH_WIND_01")

    assert rule.evaluate(_sample(wind_speed_ms=None)) is None
    assert rule.evaluate(_sample()) is None


def test_a_non_numeric_value_cannot_fire_a_rule():
    """Wrong type is missing, not truthy."""
    sample = _sample()
    object.__setattr__(sample, "wind_speed_ms", "very windy")

    assert sample.read("wind_speed_ms") is None


def test_a_boolean_is_not_read_as_a_number():
    sample = _sample()
    object.__setattr__(sample, "wind_speed_ms", True)

    assert sample.read("wind_speed_ms") is None


def test_an_unknown_variable_reads_as_missing():
    assert _sample().read("cosmic_ray_flux") is None


# --- Window applicability -----------------------------------------------------


def test_hourly_and_daily_rainfall_are_separate_rules(rules):
    """50 mm in an hour and 50 mm across a day are different events."""
    hourly = _rule(rules, "HEAVY_RAINFALL_HOURLY_01")
    daily = _rule(rules, "HEAVY_RAINFALL_DAILY_01")

    assert hourly.windows == frozenset({SampleWindow.HOUR})
    assert daily.windows == frozenset({SampleWindow.DAY})
    assert hourly.entry_threshold != daily.entry_threshold


def test_a_rule_never_fires_on_a_window_it_was_not_calibrated_for(rules, settings):
    hourly = _rule(rules, "HEAVY_RAINFALL_HOURLY_01")
    day_sample = _sample(
        SampleWindow.DAY, precipitation_mm=settings.HEAVY_RAINFALL_THRESHOLD * 2
    )

    assert hourly.evaluate(day_sample) is None


def test_probability_rules_do_not_apply_to_observations(rules):
    """An observation is certain; a probability of it is meaningless."""
    rule = _rule(rules, "SEVERE_PRECIPITATION_PROBABILITY_01")

    assert SampleWindow.OBSERVATION not in rule.windows


def test_heat_and_wind_apply_to_every_window(rules):
    for rule_id in ("EXTREME_HEAT_01", "HIGH_WIND_01"):
        rule = _rule(rules, rule_id)
        assert rule.windows == frozenset(
            {SampleWindow.OBSERVATION, SampleWindow.HOUR, SampleWindow.DAY}
        )


# --- Multiple rules -----------------------------------------------------------


def test_several_rules_can_fire_on_one_sample(engine, settings):
    sample = _sample(
        temperature_c=settings.EXTREME_HEAT_THRESHOLD + 1,
        wind_speed_ms=settings.HIGH_WIND_THRESHOLD + 1,
        precipitation_mm=settings.HEAVY_RAINFALL_THRESHOLD + 1,
    )

    results = engine.evaluate_sample(sample)

    assert {result.rule.alert_type for result in results} == {
        AlertType.EXTREME_HEAT,
        AlertType.HIGH_WIND,
        AlertType.HEAVY_RAINFALL,
    }


def test_wind_speed_and_gust_are_independent_rules(engine, settings):
    """A calm mean with damaging gusts is a real pattern and must be catchable."""
    sample = _sample(
        wind_speed_ms=settings.HIGH_WIND_THRESHOLD - 1,
        wind_gust_ms=settings.HIGH_WIND_GUST_THRESHOLD + 1,
    )

    fired = {result.rule.rule_id for result in engine.evaluate_sample(sample)}

    assert fired == {"HIGH_WIND_GUST_01"}


def test_a_quiet_sample_fires_nothing(engine):
    sample = _sample(
        temperature_c=24.0,
        wind_speed_ms=2.0,
        wind_gust_ms=4.0,
        precipitation_mm=0.0,
        precipitation_probability_pct=5.0,
    )

    assert engine.evaluate_sample(sample) == []


# --- Evidence -----------------------------------------------------------------


def test_a_result_carries_the_numbers_that_produced_it(rules, settings):
    rule = _rule(rules, "HIGH_WIND_01")

    result = rule.evaluate(_sample(wind_speed_ms=24.7, wind_gust_ms=31.0))

    assert result.rule.rule_id == "HIGH_WIND_01"
    assert result.rule.variable == "wind_speed_ms"
    assert result.rule.unit == "m/s"
    assert result.observed_value == pytest.approx(24.7)
    assert result.threshold == pytest.approx(settings.HIGH_WIND_WARNING_THRESHOLD)


def test_supporting_context_is_recorded(rules):
    rule = _rule(rules, "HIGH_WIND_01")

    result = rule.evaluate(_sample(wind_speed_ms=24.7, wind_gust_ms=31.0))

    assert result.context == {"wind_gust_ms": 31.0}


def test_heat_records_humidity_because_temperature_alone_is_not_a_health_model(rules):
    rule = _rule(rules, "EXTREME_HEAT_01")

    result = rule.evaluate(
        _sample(
            temperature_c=44.0,
            apparent_temperature_c=51.0,
            relative_humidity_pct=62.0,
        )
    )

    assert result.context == {
        "apparent_temperature_c": 51.0,
        "relative_humidity_pct": 62.0,
    }


def test_missing_context_variables_are_omitted_not_nulled(rules):
    rule = _rule(rules, "HIGH_WIND_01")

    result = rule.evaluate(_sample(wind_speed_ms=24.7))

    assert result.context == {}


def test_the_description_names_the_value_the_threshold_and_the_certainty(rules):
    rule = _rule(rules, "HIGH_WIND_01")

    observed = rule.evaluate(
        _sample(SampleWindow.OBSERVATION, wind_speed_ms=24.7)
    ).describe()
    forecast = rule.evaluate(_sample(SampleWindow.HOUR, wind_speed_ms=24.7)).describe()

    assert observed.startswith("Observed wind speed of 24.7 m/s")
    assert forecast.startswith("Forecast wind speed over one hour of 24.7 m/s")
    for text in (observed, forecast):
        assert "WeatherGPT" in text  # never implies an external authority
        assert "20.8 m/s" in text


# --- Sample construction ------------------------------------------------------


def test_an_observation_produces_an_observed_sample():
    sample = sample_from_observation(
        _report(temperature_c=41.0).current, validity=timedelta(hours=1)
    )

    assert sample.kind is AlertKind.OBSERVED
    assert sample.window is SampleWindow.OBSERVATION
    assert sample.valid_from == NOW
    assert sample.valid_until == NOW + timedelta(hours=1)


def test_an_observation_carries_no_probability():
    """It already happened; a probability would be a category error."""
    sample = sample_from_observation(
        _report(precipitation_mm=8.0).current, validity=timedelta(hours=1)
    )

    assert sample.precipitation_probability_pct is None


def test_a_forecast_hour_produces_a_forecast_risk_sample():
    sample = sample_from_hourly(
        HourlyForecastPoint(valid_at=NOW, temperature_c=41.0, wind_speed_ms=9.0)
    )

    assert sample.kind is AlertKind.FORECAST_RISK
    assert sample.window is SampleWindow.HOUR
    assert sample.valid_until == NOW + timedelta(hours=1)


def test_a_forecast_day_maps_aggregates_onto_the_shared_vocabulary():
    """A threshold rule asks about the day's worst case."""
    sample = sample_from_daily(
        DailyForecastPoint(
            date=date(2026, 9, 4),
            temperature_min_c=26.0,
            temperature_max_c=44.0,
            precipitation_sum_mm=120.0,
            wind_speed_max_ms=22.0,
            wind_gust_max_ms=30.0,
            precipitation_probability_max_pct=95.0,
        )
    )

    assert sample.kind is AlertKind.FORECAST_RISK
    assert sample.window is SampleWindow.DAY
    assert sample.temperature_c == pytest.approx(44.0)  # the maximum
    assert sample.precipitation_mm == pytest.approx(120.0)  # the total
    assert sample.wind_speed_ms == pytest.approx(22.0)  # the maximum
    assert sample.valid_from == datetime(2026, 9, 4, tzinfo=timezone.utc)
    assert sample.valid_until == datetime(2026, 9, 5, tzinfo=timezone.utc)


# --- Engine -------------------------------------------------------------------


def test_the_engine_evaluates_validated_current_conditions(engine, settings):
    report = _report(temperature_c=settings.EXTREME_HEAT_WARNING_THRESHOLD + 1)

    evaluation = engine.evaluate_observation(report, now=NOW)

    assert evaluation.triggered
    assert len(evaluation.results) == 1
    result = evaluation.results[0]
    assert result.rule.alert_type is AlertType.EXTREME_HEAT
    assert result.severity is AlertSeverity.WARNING
    assert result.sample.kind is AlertKind.OBSERVED


def test_calm_conditions_produce_no_alerts(engine):
    evaluation = engine.evaluate_observation(
        _report(temperature_c=24.0, wind_speed_ms=2.0), now=NOW
    )

    assert not evaluation.triggered
    assert evaluation.results == ()


def test_forecast_evaluation_produces_forecast_risk_only(engine, settings):
    forecast = Forecast(
        location=Location(coordinates=HYDERABAD),
        hourly=[
            HourlyForecastPoint(
                valid_at=NOW + timedelta(hours=2),
                precipitation_mm=settings.HEAVY_RAINFALL_SEVERE_THRESHOLD + 10,
            )
        ],
        daily=[],
        provenance=DataProvenance(
            provider_id="open-meteo", provider_name="Open-Meteo", fetched_at=NOW
        ),
    )

    evaluation = engine.evaluate_forecast(forecast, now=NOW)

    assert evaluation.triggered
    assert all(
        result.sample.kind is AlertKind.FORECAST_RISK for result in evaluation.results
    )
    assert evaluation.results[0].severity is AlertSeverity.SEVERE


def test_forecast_points_beyond_the_horizon_are_ignored(rules, settings):
    """A risk flagged ten days out is noise, not information."""
    engine = AlertEngine(rules, forecast_lookahead=timedelta(hours=6))
    forecast = Forecast(
        location=Location(coordinates=HYDERABAD),
        hourly=[
            HourlyForecastPoint(
                valid_at=NOW + timedelta(hours=hours),
                wind_speed_ms=settings.HIGH_WIND_SEVERE_THRESHOLD + 5,
            )
            for hours in (2, 48)
        ],
        provenance=DataProvenance(
            provider_id="open-meteo", provider_name="Open-Meteo", fetched_at=NOW
        ),
    )

    evaluation = engine.evaluate_forecast(forecast, now=NOW)

    assert evaluation.samples_examined == 1
    assert len(evaluation.results) == 1


def test_an_empty_forecast_evaluates_cleanly(engine):
    forecast = Forecast(
        location=Location(coordinates=HYDERABAD),
        provenance=DataProvenance(
            provider_id="open-meteo", provider_name="Open-Meteo", fetched_at=NOW
        ),
    )

    evaluation = engine.evaluate_forecast(forecast, now=NOW)

    assert not evaluation.triggered
    assert evaluation.samples_examined == 0


# --- Configurability ----------------------------------------------------------


def test_thresholds_come_from_configuration_not_from_code():
    """Every rule's entry threshold must trace back to a setting."""
    lenient = build_rules(
        Settings(
            HIGH_WIND_THRESHOLD=5.0,
            HIGH_WIND_WARNING_THRESHOLD=8.0,
            HIGH_WIND_SEVERE_THRESHOLD=12.0,
        )
    )
    strict = build_rules(
        Settings(
            HIGH_WIND_THRESHOLD=40.0,
            HIGH_WIND_WARNING_THRESHOLD=50.0,
            HIGH_WIND_SEVERE_THRESHOLD=60.0,
        )
    )

    breeze = _sample(wind_speed_ms=10.0)

    assert _rule(lenient, "HIGH_WIND_01").evaluate(breeze) is not None
    assert _rule(strict, "HIGH_WIND_01").evaluate(breeze) is None


def test_an_incoherent_threshold_configuration_fails_at_startup():
    """Raising the entry threshold above the warning band inverts the ladder.

    Failing while the rule set is being built means a misconfiguration is a
    crash on boot, not an alert that reports the wrong severity for months.
    """
    with pytest.raises(ValueError, match="ascend by threshold"):
        build_rules(Settings(HIGH_WIND_THRESHOLD=40.0))  # warning stays at 20.8


def test_every_rule_declares_an_id_type_variable_unit_and_ladder(rules):
    for rule in rules:
        assert rule.rule_id
        assert isinstance(rule.alert_type, AlertType)
        assert rule.variable
        assert rule.unit
        assert rule.bands
        assert rule.title and rule.explanation


def test_rule_ids_are_unique(rules):
    ids = [rule.rule_id for rule in rules]

    assert len(ids) == len(set(ids))
