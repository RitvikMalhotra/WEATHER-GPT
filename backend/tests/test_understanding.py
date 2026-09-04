"""What the router understands, and what the verdict is allowed to say.

Three things are pinned here, because each of them is a way to be confidently
wrong rather than merely unhelpful:

* a question about the past must not be answered with the present;
* a place named in the message must beat the place the session is on;
* a verdict must never contain a number nobody measured.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.ai import verdict as verdicts
from app.ai.conversation import ConversationState
from app.ai.intents import IntentDetector
from app.ai.models import AdvisoryPurpose, Intent, LocationInput
from app.domain.forecast import DailyForecastPoint, Forecast, HourlyForecastPoint
from app.domain.location import Coordinates, Location
from app.domain.provenance import DataProvenance
from app.domain.weather import CurrentWeather, WeatherCondition, WeatherReport
from app.services.geocoding import GeocodingService, rank

NOW = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)


def blank_state() -> ConversationState:
    return ConversationState(session_id="s", updated_at=NOW)


def detect(message: str, state: ConversationState | None = None):
    return IntentDetector().detect(message, state or blank_state(), now=NOW)


# ------------------------------------------------------------------ routing


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        # The hard assertions: the same variable, three different questions.
        ("What was the wind speed 6 hours ago?", Intent.HISTORICAL_WEATHER),
        ("What is the wind speed now?", Intent.CURRENT_WEATHER),
        ("What will the wind speed be in 6 hours?", Intent.HOURLY_FORECAST),
        ("What was the rainfall yesterday?", Intent.HISTORICAL_WEATHER),
        ("What is the weather now?", Intent.CURRENT_WEATHER),
        ("Is there rain expected in 48 hours?", Intent.HOURLY_FORECAST),
        ("Will it rain tomorrow?", Intent.DAILY_FORECAST),
        ("7 day forecast", Intent.DAILY_FORECAST),
        # A past event in a sentence whose question is about the future.
        ("I sprayed pesticides yesterday. Will it rain tomorrow?", Intent.DAILY_FORECAST),
        ("I sprayed pesticides yesterday. Is rain expected in 48 hours?", Intent.HOURLY_FORECAST),
    ],
)
def test_the_tense_of_the_question_chooses_the_data(message: str, expected: Intent) -> None:
    assert detect(message).intent is expected


def test_a_planning_question_with_a_horizon_uses_the_forecast_for_that_horizon() -> None:
    """The pesticide question, end to end through the router.

    It mentions a crop, a farm, a place and two different times. The horizon it
    asks about is 48 hours ahead, so that is the data it gets — and the fact
    that it is an agricultural question changes the advice, not the window.
    """
    found = detect(
        "I sprayed pesticides on my chilli crop in a farm in Khammam yesterday. "
        "Is there rain expected in 48 hours that can wash it off, and should I reapply?"
    )

    assert found.intent is Intent.HOURLY_FORECAST
    assert found.horizon_hours == 48
    assert found.location is not None and found.location.location == "Khammam"
    assert found.purpose is AdvisoryPurpose.AGRICULTURE
    assert found.variable == "precipitation"
    assert found.event_phrase is not None  # "yesterday" is kept, as context


def test_a_planning_question_with_no_clock_in_it_still_composes_a_risk_view() -> None:
    assert detect("Is it safe to travel to Pune?").intent is Intent.LOCATION_RISK


def test_an_hours_ago_question_carries_instants_and_a_calendar_one_carries_dates() -> None:
    hourly = detect("What was the wind speed 6 hours ago?")
    assert hourly.start_time is not None and hourly.end_time is not None

    calendar = detect("How much rain fell yesterday?")
    assert calendar.start is not None and calendar.end is not None
    assert calendar.start_time is None


# ----------------------------------------------------------------- location


def test_a_place_in_the_message_beats_the_place_the_session_is_on() -> None:
    state = blank_state()
    state.location = LocationInput(location="Hyderabad")

    assert detect("What is the weather in Mumbai?", state).location.location == "Mumbai"


def test_a_question_with_no_place_inherits_the_conversation() -> None:
    state = blank_state()
    state.location = LocationInput(location="Delhi")

    assert detect("Will it rain tomorrow?", state).location.location == "Delhi"


@pytest.mark.parametrize(
    ("message", "place"),
    [
        ("What is the weather in Hyderabad?", "Hyderabad"),
        ("What about Miyapur?", "Miyapur"),
        ("Now tell me about Delhi", "Delhi"),
        ("weather in my farm in Nashik", "Nashik"),
        ("Is it safe to travel to Pune?", "Pune"),
        # Hindi marks its places from behind.
        ("हैदराबाद का मौसम कैसा है?", "हैदराबाद"),
        ("दिल्ली में कल बारिश होगी?", "दिल्ली"),
        ("Mumbai में बारिश का chance कितना है?", "Mumbai"),
        ("aaj Hyderabad ka mausam kaisa hai?", "Hyderabad"),
        ("kal Delhi mein baarish hogi?", "Delhi"),
    ],
)
def test_the_place_is_read_out_of_the_sentence(message: str, place: str) -> None:
    found = detect(message)
    assert found.location is not None
    assert found.location.location == place


@pytest.mark.parametrize(
    "message",
    [
        "Is it going to rain?",
        "What is the temperature?",
        "6 ghante pehle wind speed kitni thi?",
        "agale 3 din ka forecast batao",
    ],
)
def test_a_measurement_word_never_becomes_a_place(message: str) -> None:
    """Sending "rain" to a geocoder answers about the wrong point on Earth."""
    assert detect(message).location is None


def test_a_bare_place_name_is_a_question_about_which_place() -> None:
    found = detect("Miyapur")
    assert found.intent is Intent.LOCATION_SEARCH
    assert found.location is not None and found.location.location == "Miyapur"


# ------------------------------------------------------------------ ranking


def at(name: str, latitude: float, longitude: float, admin1: str, population: int | None = None):
    return Location(
        coordinates=Coordinates(latitude=latitude, longitude=longitude),
        name=name,
        admin1=admin1,
        country="India",
        population=population,
    )


def test_the_conversation_decides_which_miyapur() -> None:
    """Ranking, not a rule about Miyapur. Any repeated name behaves this way."""
    candidates = [
        at("Miyapur", 26.9, 75.8, "Rajasthan"),
        at("Miyapur", 17.49, 78.39, "Telangana"),
    ]
    hyderabad = Coordinates(latitude=17.385, longitude=78.4867)

    assert rank(candidates, "Miyapur", hyderabad)[0].admin1 == "Telangana"
    # With no conversation to go on, the order is left to the gazetteer.
    assert rank(candidates, "Miyapur")[0].admin1 == "Rajasthan"


def test_an_exact_name_outranks_a_longer_one_that_merely_contains_it() -> None:
    candidates = [at("Miyapurwa", 27.8, 81.9, "Lumbini"), at("Miyapur", 17.4, 78.3, "Telangana")]
    assert rank(candidates, "Miyapur")[0].name == "Miyapur"


def test_an_accent_does_not_hide_a_repeated_name() -> None:
    """A gazetteer returns "Miyāpur" or "Miyapur"; both are the same question."""
    candidates = [at("Miyāpur", 16.3, 76.9, "Karnataka"), at("Miyapur", 17.4, 78.3, "Telangana")]
    assert GeocodingService.ambiguous(candidates)


def test_one_place_of_that_name_is_not_ambiguous() -> None:
    assert not GeocodingService.ambiguous([at("Khammam", 17.2, 80.1, "Telangana")])


# ------------------------------------------------------------------ verdict


def provenance() -> DataProvenance:
    return DataProvenance(
        provider_id="p", provider_name="P", fetched_at=NOW, source_url="https://p.test"
    )


def report(**current) -> WeatherReport:
    return WeatherReport(
        location=at("Hyderabad", 17.4, 78.5, "Telangana"),
        current=CurrentWeather(observed_at=NOW, **current),
        provenance=provenance(),
    )


def test_a_verdict_reads_the_value_it_was_given() -> None:
    hot = verdicts.build(
        intent=Intent.CURRENT_WEATHER,
        data=report(temperature_c=38.0, apparent_temperature_c=41.0),
    )
    mild = verdicts.build(
        intent=Intent.CURRENT_WEATHER,
        data=report(temperature_c=22.0, apparent_temperature_c=22.0),
    )

    assert hot is not None and mild is not None
    assert hot.text != mild.text


def test_no_measurement_means_no_verdict() -> None:
    """Silence is the honest output when nothing was reported."""
    assert verdicts.build(intent=Intent.CURRENT_WEATHER, data=report()) is None


def test_a_verdict_is_never_offered_for_a_result_that_carries_no_reading() -> None:
    assert verdicts.build(intent=Intent.LOCATION_SEARCH, data={"results": []}) is None
    assert verdicts.build(intent=Intent.ALERTS, data={"alerts": []}) is None
    assert verdicts.build(intent=Intent.UNKNOWN, data=None) is None


def test_a_forecast_verdict_only_looks_as_far_as_the_question_did() -> None:
    """Rain on day six does not make it rain in the next six hours."""
    dry_then_wet = Forecast(
        location=at("Hyderabad", 17.4, 78.5, "Telangana"),
        hourly=[
            HourlyForecastPoint(
                valid_at=NOW.replace(hour=hour), precipitation_probability_pct=0.0
            )
            for hour in range(6)
        ]
        + [
            HourlyForecastPoint(
                valid_at=NOW.replace(hour=12), precipitation_probability_pct=95.0,
                precipitation_mm=8.0,
            )
        ],
        provenance=provenance(),
    )

    near = verdicts.build(intent=Intent.HOURLY_FORECAST, data=dry_then_wet, horizon_hours=6)
    far = verdicts.build(intent=Intent.HOURLY_FORECAST, data=dry_then_wet, horizon_hours=24)

    assert near is not None and far is not None
    assert near.text != far.text


def test_a_rainfall_verdict_reports_the_total_that_was_measured() -> None:
    class Record:
        def __init__(self, mm: float) -> None:
            self.weather = CurrentWeather(observed_at=NOW, precipitation_mm=mm)

    reading = verdicts.build(
        intent=Intent.HISTORICAL_WEATHER,
        data=[Record(4.0), Record(6.5)],
        variable="precipitation",
    )

    assert reading is not None
    assert "10.5" in reading.text


def test_an_agricultural_question_is_answered_without_advising_on_the_chemical() -> None:
    """Whether to re-apply depends on the label, and this layer says so."""
    forecast = Forecast(
        location=at("Khammam", 17.2, 80.1, "Telangana"),
        daily=[
            DailyForecastPoint(
                date=NOW.date(),
                precipitation_probability_max_pct=80.0,
                condition=WeatherCondition.RAIN,
            )
        ],
        provenance=provenance(),
    )

    reading = verdicts.build(
        intent=Intent.DAILY_FORECAST,
        data=forecast,
        purpose=AdvisoryPurpose.AGRICULTURE,
        horizon_hours=48,
    )

    assert reading is not None
    # The weather half of the question is what gets answered: a recommendation
    # about the window, built from the figures that came back.
    assert "80" in reading.text
    said = f"{reading.text} {reading.caveat or ''}".lower()
    # And only the weather half. Rainfast periods, labels and product chemistry
    # are questions about the product, and raising them unprompted answers a
    # question nobody asked — which is what this test now guards against.
    for product_talk in ("rainfast", "label", "product", "reapply", "chemical"):
        assert product_talk not in said


def test_a_hindi_question_gets_a_hindi_verdict() -> None:
    reading = verdicts.build(
        intent=Intent.CURRENT_WEATHER,
        data=report(temperature_c=38.0, apparent_temperature_c=41.0),
        language="hi",
    )
    assert reading is not None
    assert any("ऀ" <= ch <= "ॿ" for ch in reading.text)


# ------------------------------------------------------- carrying a question


def test_a_new_place_repeats_the_question_the_conversation_was_asking() -> None:
    """"What about Miyapur?" is not a new subject, it is a new place."""
    state = blank_state()
    state.last_intent = Intent.CURRENT_WEATHER
    state.location = LocationInput(location="Hyderabad")

    found = detect("What about Miyapur?", state)

    assert found.intent is Intent.CURRENT_WEATHER
    assert found.location is not None and found.location.location == "Miyapur"


def test_a_place_with_no_question_behind_it_is_a_question_about_the_place() -> None:
    assert detect("Miyapur").intent is Intent.LOCATION_SEARCH


def test_an_off_topic_message_carries_nothing_forward() -> None:
    """A conversation about the weather does not make every message one."""
    state = blank_state()
    state.last_intent = Intent.CURRENT_WEATHER
    state.location = LocationInput(location="Hyderabad")

    assert detect("Tell me a joke.", state).intent is Intent.UNKNOWN


def test_a_trace_of_rain_is_not_reported_as_no_rain() -> None:
    """The verdict must not contradict the rows printed above it."""

    class Record:
        def __init__(self, mm: float) -> None:
            self.weather = CurrentWeather(observed_at=NOW, precipitation_mm=mm)

    trace = verdicts.build(
        intent=Intent.HISTORICAL_WEATHER,
        data=[Record(0.1), Record(0.2)],
        variable="precipitation",
    )
    dry = verdicts.build(
        intent=Intent.HISTORICAL_WEATHER,
        data=[Record(0.0), Record(0.0)],
        variable="precipitation",
    )

    assert trace is not None and "0.3" in trace.text
    assert dry is not None and "0.3" not in dry.text
    assert trace.text != dry.text
