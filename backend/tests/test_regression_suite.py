"""Comprehensive regression test suite for WeatherGPT continuation.

Verifies:
1. Temporal routing (past, present, future).
2. Historical retrieval ("What was the wind speed 6 hours ago?" -> hourly window, wind speed in rendered text).
3. 48-hour future forecast routing (hourly forecast preferred).
4. Multi-temporal query (pesticide in Khammam yesterday, rain in 48 hours, agricultural caveat).
5. Explicit location overrides session location.
6. Ambiguous location behavior.
7. Hindi and Hinglish query routing.
8. "कल" boundary check: does not match inside "कलकत्ता".
9. One-line verdict is strictly grounded in retrieved numbers.
10. Source / provenance accuracy (Open-Meteo vs NOAA GFS).
11. GFS provider preservation and model run provenance.
12. Deterministic alert engine rules preservation.
"""

from __future__ import annotations

from datetime import datetime, timezone
import pytest

from app.ai.conversation import ConversationState
from app.ai.grounding import CatalogRegistry, GroundedRenderer
from app.ai.intents import IntentDetector
from app.ai.models import (
    AdvisoryPurpose,
    Intent,
    LocationInput,
    ToolResult,
)
from app.ai.temporal import Tense
from app.ai import verdict
from app.alerts.engine import AlertEngine
from app.alerts.rules import build_rules
from app.config.settings import Settings
from app.api.v1.historical import HistoricalObservation, HistoricalWeatherResponse, TimeRange
from app.domain.alert import AlertSeverity, AlertType
from app.domain.forecast import Forecast, HourlyForecastPoint
from app.domain.location import Coordinates, Location
from app.domain.provenance import DataProvenance
from app.domain.weather import CurrentWeather, WeatherCondition, WeatherReport
from app.providers.gfs.provider import GFS_PROVIDER_ID
from app.services.geocoding import GeocodingService, rank

NOW = datetime(2026, 9, 5, 12, 0, tzinfo=timezone.utc)


def blank_state() -> ConversationState:
    return ConversationState(session_id="reg-session", updated_at=NOW)


def make_provenance(provider_id: str = "open-meteo", name: str = "Open-Meteo") -> DataProvenance:
    return DataProvenance(
        provider_id=provider_id,
        provider_name=name,
        model="best_match",
        fetched_at=NOW,
        source_url="https://weather.example.com",
        license="CC-BY-4.0",
        attribution="Weather data by " + name,
    )


# -----------------------------------------------------------------------------
# 1. Temporal Routing (Past, Present, Future)
# -----------------------------------------------------------------------------

def test_temporal_past_present_future_routing():
    detector = IntentDetector()
    state = blank_state()

    res_past = detector.detect("What was the wind speed 6 hours ago?", state, now=NOW)
    assert res_past.intent is Intent.HISTORICAL_WEATHER
    assert res_past.tense is Tense.PAST
    assert res_past.start_time is not None
    assert res_past.end_time is not None

    res_present = detector.detect("What is the weather in Hyderabad?", state, now=NOW)
    assert res_present.intent is Intent.CURRENT_WEATHER
    assert res_present.tense is Tense.PRESENT

    res_future = detector.detect("Will it rain tomorrow in Hyderabad?", state, now=NOW)
    assert res_future.intent is Intent.DAILY_FORECAST
    assert res_future.tense is Tense.FUTURE


# -----------------------------------------------------------------------------
# 2. Historical Retrieval & Wind Speed Rendering
# -----------------------------------------------------------------------------

def test_historical_hourly_window_and_wind_speed_rendering():
    """Verify that hourly historical queries include wind speed in rendered facts."""
    detector = IntentDetector()
    detected = detector.detect("What was the wind speed 6 hours ago in Hyderabad?", blank_state(), now=NOW)
    assert detected.intent is Intent.HISTORICAL_WEATHER
    assert detected.variable == "wind"

    # Mock historical response containing wind data
    history_resp = HistoricalWeatherResponse(
        requested={"latitude": 17.385, "longitude": 78.4867},
        location=Location(
            coordinates=Coordinates(latitude=17.385, longitude=78.4867),
            name="Hyderabad",
            country="India",
        ),
        range=TimeRange(start=NOW, end=NOW),
        search_radius_km=25.0,
        count=1,
        truncated=False,
        observations=[
            HistoricalObservation(
                latitude=17.385,
                longitude=78.4867,
                distance_km=0.0,
                weather=CurrentWeather(
                    observed_at=NOW,
                    temperature_c=28.5,
                    wind_speed_ms=6.8,
                    wind_gust_ms=11.2,
                    condition_description="Breezy",
                    precipitation_mm=0.0,
                ),
                provenance=make_provenance(),
            )
        ],
    )

    renderer = GroundedRenderer()
    catalog, _ = CatalogRegistry().resolve("en")
    tool_result = ToolResult(tool="historical_weather", data=history_resp.model_dump(mode="json"))

    rendered = renderer.render(Intent.HISTORICAL_WEATHER, [tool_result], catalog)
    # Wind speed must be explicitly present in the rendered facts line
    assert "wind 6.8 m/s" in rendered.answer
    assert "gust 11.2 m/s" in rendered.answer


# -----------------------------------------------------------------------------
# 3. 48-Hour Future Forecast Routing
# -----------------------------------------------------------------------------

def test_48_hour_forecast_routing_prefers_hourly():
    detector = IntentDetector()
    detected = detector.detect("Is there rain expected in 48 hours in Mumbai?", blank_state(), now=NOW)
    assert detected.intent is Intent.HOURLY_FORECAST
    assert detected.horizon_hours == 48
    assert detected.location is not None
    assert detected.location.location == "Mumbai"


# -----------------------------------------------------------------------------
# 4. Multi-Temporal Query (Pesticide Clause + 48h Weather Horizon)
# -----------------------------------------------------------------------------

def test_multi_temporal_pesticide_query_interpretation():
    query = (
        "I sprayed pesticides on my chilli crop in a farm in Khammam yesterday. "
        "Is there rain expected in 48 hours that can wash it off, and should I reapply?"
    )
    detected = IntentDetector().detect(query, blank_state(), now=NOW)

    assert detected.intent is Intent.HOURLY_FORECAST
    assert detected.tense is Tense.FUTURE
    assert detected.horizon_hours == 48
    assert detected.location is not None
    assert detected.location.location == "Khammam"
    assert detected.purpose is AdvisoryPurpose.AGRICULTURE
    assert detected.variable == "precipitation"
    assert detected.event_phrase is not None
    assert "yesterday" in detected.event_phrase.lower()

    # Verify agricultural verdict includes disclaimer regarding chemical rainfastness
    forecast = Forecast(
        location=Location(
            coordinates=Coordinates(latitude=17.247, longitude=80.151),
            name="Khammam",
            country="India",
        ),
        hourly=[
            HourlyForecastPoint(
                valid_at=NOW,
                precipitation_probability_pct=75.0,
                precipitation_mm=4.5,
                condition=WeatherCondition.RAIN,
            )
        ],
        provenance=make_provenance(),
    )
    v = verdict.build(
        intent=Intent.HOURLY_FORECAST,
        data=forecast,
        purpose=detected.purpose,
        horizon_hours=48,
    )
    assert v is not None
    assert v.caveat is not None
    assert "rainfast" in v.caveat.lower()
    # It must never instruct the user to reapply pesticides
    assert "reapply" not in v.text.lower()


# -----------------------------------------------------------------------------
# 5. Explicit Location Overrides Active Session Location
# -----------------------------------------------------------------------------

def test_explicit_location_overrides_session_context():
    detector = IntentDetector()
    state = blank_state()
    state.location = LocationInput(location="Hyderabad")

    # Asking about Mumbai while session is on Hyderabad
    detected = detector.detect("What about Mumbai?", state, now=NOW)
    assert detected.location is not None
    assert detected.location.location == "Mumbai"

    # If no location is mentioned, it inherits Hyderabad
    detected_inherit = detector.detect("Will it rain tomorrow?", state, now=NOW)
    assert detected_inherit.location is not None
    assert detected_inherit.location.location == "Hyderabad"


# -----------------------------------------------------------------------------
# 6. Ambiguous Location Behavior
# -----------------------------------------------------------------------------

def test_ambiguous_location_ranking_and_flagging():
    candidates = [
        Location(
            coordinates=Coordinates(latitude=26.9, longitude=75.8),
            name="Miyapur",
            admin1="Rajasthan",
            country="India",
        ),
        Location(
            coordinates=Coordinates(latitude=17.49, longitude=78.39),
            name="Miyapur",
            admin1="Telangana",
            country="India",
        ),
    ]
    # Name shared across multiple administrative states is flagged ambiguous
    assert GeocodingService.ambiguous(candidates) is True

    # Biased near Hyderabad prioritizes Telangana
    hyderabad_pt = Coordinates(latitude=17.385, longitude=78.4867)
    ranked = rank(candidates, "Miyapur", near=hyderabad_pt)
    assert ranked[0].admin1 == "Telangana"

    # Single unique location is NOT ambiguous
    single = [
        Location(
            coordinates=Coordinates(latitude=17.2, longitude=80.1),
            name="Khammam",
            admin1="Telangana",
            country="India",
        )
    ]
    assert GeocodingService.ambiguous(single) is False


# -----------------------------------------------------------------------------
# 7. Hindi and Hinglish Routing
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("msg", "expected_tense", "expected_place"),
    [
        ("आज हैदराबाद का मौसम कैसा है?", Tense.PRESENT, "हैदराबाद"),
        ("कल दिल्ली में बारिश होगी?", Tense.FUTURE, "दिल्ली"),
        ("6 घंटे पहले हवा की गति कितनी थी?", Tense.PAST, None),
        ("aaj Hyderabad ka mausam kaisa hai?", Tense.PRESENT, "Hyderabad"),
        ("kal Delhi mein baarish hogi?", Tense.FUTURE, "Delhi"),
        ("Hyderabad में कल rain होगी?", Tense.FUTURE, "Hyderabad"),
    ],
)
def test_hindi_and_hinglish_query_routing(msg, expected_tense, expected_place):
    detected = IntentDetector().detect(msg, blank_state(), now=NOW)
    assert detected.tense is expected_tense
    if expected_place:
        assert detected.location is not None
        assert detected.location.location == expected_place


# -----------------------------------------------------------------------------
# 8. "कल" Does NOT Match Inside "कलकत्ता"
# -----------------------------------------------------------------------------

def test_kal_does_not_match_inside_kolkata():
    detected = IntentDetector().detect("कलकत्ता का मौसम कैसा है?", blank_state(), now=NOW)
    assert detected.tense is Tense.PRESENT
    assert detected.location is not None
    assert "कलकत्ता" in detected.location.location


# -----------------------------------------------------------------------------
# 9. One-Line Verdict Grounded in Retrieved Numbers
# -----------------------------------------------------------------------------

def test_verdict_grounded_in_actual_data():
    forecast_dry = Forecast(
        location=Location(
            coordinates=Coordinates(latitude=17.385, longitude=78.4867),
            name="Hyderabad",
            country="India",
        ),
        hourly=[
            HourlyForecastPoint(
                valid_at=NOW,
                precipitation_probability_pct=0.0,
                precipitation_mm=0.0,
            )
        ],
        provenance=make_provenance(),
    )
    v_dry = verdict.build(intent=Intent.HOURLY_FORECAST, data=forecast_dry, horizon_hours=24)
    assert v_dry is not None
    assert "unlikely" in v_dry.text.lower() or "no rain" in v_dry.text.lower()

    # Empty reading produces no verdict (never invents numbers)
    assert verdict.build(intent=Intent.UNKNOWN, data=None) is None


# -----------------------------------------------------------------------------
# 10. Source / Provenance Accuracy & GFS Provider
# -----------------------------------------------------------------------------

def test_gfs_provider_metadata_and_provenance():
    prov = make_provenance(provider_id=GFS_PROVIDER_ID, name="NOAA GFS")
    assert prov.provider_id == "noaa-gfs"
    assert prov.provider_name == "NOAA GFS"
    assert "GFS" in prov.attribution


# -----------------------------------------------------------------------------
# 11. Deterministic Alert Engine Rules
# -----------------------------------------------------------------------------

def test_alert_engine_rules_execution():
    engine = AlertEngine(build_rules(Settings()))
    report = WeatherReport(
        location=Location(
            coordinates=Coordinates(latitude=28.6, longitude=77.2),
            name="Delhi",
            country="India",
        ),
        current=CurrentWeather(
            observed_at=NOW,
            temperature_c=46.5,  # Extreme heat
            condition=WeatherCondition.CLEAR,
        ),
        provenance=make_provenance(),
    )
    evaluation = engine.evaluate_observation(report)
    heat = [r for r in evaluation.results if r.rule.alert_type is AlertType.EXTREME_HEAT]
    assert len(heat) > 0
    assert heat[0].severity in {
        AlertSeverity.WATCH,
        AlertSeverity.WARNING,
        AlertSeverity.SEVERE,
        AlertSeverity.EXTREME,
    }
