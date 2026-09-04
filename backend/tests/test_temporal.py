"""Temporal routing: past, present, future, and the difference between them.

These are the hard assertions. A question about six hours ago that returns the
current wind is not a rough answer, it is a wrong one delivered with the same
confidence as a right one — and the same goes for reading "yesterday" out of a
sentence whose actual question is about tomorrow. Every case below is a
sentence a person would really type.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.ai.language import detect_language, is_hinglish
from app.ai.temporal import Tense, plan

NOW = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)


def tense(message: str) -> Tense:
    return plan(message, now=NOW).tense


# --------------------------------------------------------------- the basics


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("What is the weather now?", Tense.PRESENT),
        ("What is the weather in Hyderabad?", Tense.PRESENT),
        ("What is the temperature right now?", Tense.PRESENT),
        ("What was the wind speed 6 hours ago?", Tense.PAST),
        ("What was the rainfall yesterday?", Tense.PAST),
        ("What was the temperature in Delhi yesterday?", Tense.PAST),
        ("How much rain did Mumbai receive last week?", Tense.PAST),
        ("30 minutes ago, what was the visibility?", Tense.PAST),
        ("What will the wind speed be in 6 hours?", Tense.FUTURE),
        ("Will it rain tomorrow?", Tense.FUTURE),
        ("Is there rain expected in 48 hours?", Tense.FUTURE),
        ("What will the weather be over the next 48 hours?", Tense.FUTURE),
        ("In 30 minutes will it rain?", Tense.FUTURE),
        ("7 day forecast in Hyderabad", Tense.FUTURE),
    ],
)
def test_past_present_and_future_are_told_apart(message: str, expected: Tense) -> None:
    assert tense(message) is expected


# ------------------------------------------------------------ multi-temporal


@pytest.mark.parametrize(
    "message",
    [
        "I planted rice yesterday. Will it rain tomorrow?",
        "I sprayed pesticides two days ago. Is there rain expected over the next 48 hours?",
        "It rained yesterday. Will it rain again tomorrow?",
        "The temperature was very high this morning. Will it cool down tonight?",
        "I applied fertilizer yesterday. What's the weather expected to be over the next three days?",
        "It flooded yesterday. Is heavy rain expected tonight?",
        "My flight landed this morning. Will there be thunderstorms tonight?",
        "We left the port 3 hours ago. What will the wind be like over the next 12 hours?",
    ],
)
def test_a_past_event_does_not_drag_a_future_question_into_the_past(message: str) -> None:
    """The clause that says when something *happened* is not the question."""
    assert tense(message) is Tense.FUTURE


def test_the_event_is_kept_apart_from_the_weather_target() -> None:
    resolved = plan(
        "I sprayed pesticides on my chilli crop in a farm in Khammam yesterday. "
        "Is there rain expected in 48 hours that can wash it off, and should I reapply?",
        now=NOW,
    )
    assert resolved.tense is Tense.FUTURE
    assert resolved.horizon_hours == 48
    assert resolved.hourly, "a 48-hour question is answered hour by hour"
    assert resolved.event is not None and "yesterday" in resolved.event.phrase.lower()


def test_the_furthest_stated_horizon_wins() -> None:
    """Answering the next 6 hours does not answer a question about 48."""
    resolved = plan("Will it rain in the next 6 hours, or in the next 48 hours?", now=NOW)
    assert resolved.horizon_hours == 48


# ------------------------------------------------------------------- windows


def test_an_hours_ago_question_gets_an_hour_precision_window() -> None:
    resolved = plan("What was the wind speed 6 hours ago?", now=NOW)
    assert resolved.hourly
    assert resolved.target is not None
    assert resolved.target.start is not None and resolved.target.end is not None
    # Wide enough to contain the hourly reading either side of the instant.
    assert resolved.target.start < NOW.replace(hour=6) < resolved.target.end


def test_yesterday_stays_a_calendar_date_for_the_location_to_resolve() -> None:
    resolved = plan("What was the rainfall yesterday?", now=NOW)
    assert resolved.target is not None
    assert resolved.target.start_date == NOW.date().replace(day=3)
    assert resolved.target.granularity == "day"


def test_a_daypart_narrows_a_calendar_reference() -> None:
    resolved = plan("How hot was it yesterday evening?", now=NOW)
    assert resolved.target is not None
    assert resolved.target.local_hours == (17, 21)


def test_a_48_hour_question_asks_for_enough_forecast_days() -> None:
    assert plan("Is there rain expected in 48 hours?", now=NOW).days >= 3


# ------------------------------------------------------------------- Hindi


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("आज हैदराबाद का मौसम कैसा है?", Tense.PRESENT),
        ("कल दिल्ली में बारिश होगी?", Tense.FUTURE),
        ("कल हैदराबाद में कितनी बारिश हुई?", Tense.PAST),
        ("6 घंटे पहले हवा की गति कितनी थी?", Tense.PAST),
        ("अगले 3 दिनों का मौसम बताओ", Tense.FUTURE),
        ("मुंबई में तापमान कितना है?", Tense.PRESENT),
        ("परसों बारिश होगी?", Tense.FUTURE),
    ],
)
def test_devanagari_questions_are_routed_by_tense(message: str, expected: Tense) -> None:
    assert tense(message) is expected


def test_kolkata_is_not_read_as_yesterday() -> None:
    """"कल" is a substring of "कलकत्ता"; a word boundary that ignores that
    turns a question about a city into a question about a date."""
    assert tense("कलकत्ता का मौसम कैसा है?") is Tense.PRESENT


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("aaj Hyderabad ka mausam kaisa hai?", Tense.PRESENT),
        ("kal Delhi mein baarish hogi?", Tense.FUTURE),
        ("kal Hyderabad mein kitni baarish hui?", Tense.PAST),
        ("6 ghante pehle wind speed kitni thi?", Tense.PAST),
        ("agale 3 din ka forecast batao", Tense.FUTURE),
    ],
)
def test_hinglish_questions_are_routed_by_tense(message: str, expected: Tense) -> None:
    assert tense(message) is expected


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Hyderabad में कल rain होगी?", Tense.FUTURE),
        ("दिल्ली का weather कैसा है?", Tense.PRESENT),
        ("Mumbai में बारिश का chance कितना है?", Tense.PRESENT),
    ],
)
def test_a_sentence_in_two_scripts_is_still_one_question(
    message: str, expected: Tense
) -> None:
    assert tense(message) is expected


# --------------------------------------------------------------- language


def test_the_message_outranks_the_language_selector() -> None:
    """A Hindi sentence typed with English selected was still asked in Hindi."""
    assert detect_language("कल बारिश होगी क्या?", hint="en", previous="en") == "hi"
    assert detect_language("kal baarish hogi kya?", hint="en", previous="en") == "hi"


def test_the_selector_decides_what_the_text_does_not() -> None:
    assert detect_language("Hyderabad", hint="hi", previous="en") == "hi"
    assert detect_language("Hyderabad", hint=None, previous="hi") == "hi"


def test_an_english_question_is_not_mistaken_for_hinglish() -> None:
    for message in (
        "What is the weather in Kailashahar?",
        "How much rain fell in Mumbai last week?",
        "Is it safe to drive to Pune this evening?",
    ):
        assert not is_hinglish(message), message
        assert detect_language(message, hint="en", previous="en") == "en"
