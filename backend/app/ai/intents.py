"""Deterministic intent and argument extraction used as an LLM safety guardrail.

Everything a request needs is derived here, from the person's own words:

    what     the weather variable they asked about
    where    the place named in this message, or the one the conversation is on
    when     the tense and window, from :mod:`app.ai.temporal`
    why      the context that changes what a useful answer looks like

A model may help choose a tool. It never supplies one of these. That boundary
is what stops a fluent guess from becoming a weather fact, and it is why this
module reads three registers of Hindi and English rather than deferring the
hard sentences to a model that would answer them confidently either way.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timezone

from app.ai.conversation import ConversationState
from app.ai.language import (
    ALERT_SUBJECT,
    DEVANAGARI,
    PURPOSE_TERMS,
    RISK_SUBJECT,
    VARIABLES,
    WEATHER_SUBJECT,
    detect_language,
    is_hinglish,
)
from app.ai.models import AdvisoryPurpose, Intent, LocationInput
from app.ai.temporal import TemporalPlan, Tense, plan as temporal_plan

_COORDINATES = re.compile(
    r"(?P<latitude>[+-]?\d{1,2}(?:\.\d+)?)\s*[,/]\s*"
    r"(?P<longitude>[+-]?\d{1,3}(?:\.\d+)?)"
)

# --- English and Latin-script place anchors --------------------------------

#: "about" is here for the follow-up that carries a conversation forward:
#: "What about Miyapur?" names a place and nothing else.
_PLACED = re.compile(
    r"\b(?:in|at|for|near|around|of|about|over)\s+(?P<place>[^?!,;.]+)", re.IGNORECASE
)
# "travel to Pune" carries a place; "going to rain" does not. Only verbs that
# genuinely take a destination are listed, so a plain infinitive never becomes
# a location.
_DESTINATION = re.compile(
    r"\b(?:travel(?:ling|ing)?|driv(?:e|ing)|fly(?:ing)?|flight|trip|"
    r"head(?:ing)?|commut(?:e|ing)|visit(?:ing)?)\s+to\s+(?P<place>[^?!,;.]+)",
    re.IGNORECASE,
)
# Verb-led search phrasing, which carries no preposition to anchor on.
_SEARCH_TARGET = re.compile(
    r"^(?:where\s+is|find|search(?:\s+for)?|look\s+up)\s+(?P<place>[^?!,;.]+)",
    re.IGNORECASE,
)

# --- Hindi and Hinglish place anchors --------------------------------------
#
# Hindi marks a place with a postposition *after* the noun — "हैदराबाद का मौसम",
# "Delhi mein baarish" — so the anchor that works for English ("in Delhi") finds
# nothing here. The place is the one or two words immediately before the
# postposition, and the blocklist below removes the ones that are not places.

_POSTPOSITION_DEV = re.compile(
    r"(?P<place>(?:[\wऀ-ॿ]+\s+)?[\wऀ-ॿ]+)\s*"
    r"(?:का|के|की|में|मे|पर|पे)(?![\wऀ-ॿ])"
)
_POSTPOSITION_ROMAN = re.compile(
    r"\b(?P<place>(?:[\w]+\s+)?[\w]+)\s+(?:ka|ke|ki|mein|me|main|par|pe)\b",
    re.IGNORECASE,
)

#: Words a capture can land on that are never a place. Guards every anchor
#: above against turning a verb phrase, a measurement or a date into a
#: geocoder query.
_NON_PLACE = frozenset(
    {
        # English structure
        "a", "an", "the", "be", "it", "this", "that", "there", "here",
        "go", "get", "do", "stay", "work", "bed", "sleep", "me", "you", "us",
        "them", "it all", "course", "tell", "show", "give", "please", "what",
        "how", "when", "which", "is", "was", "will", "chance", "probability",
        "expected", "next", "last", "my", "our", "farm", "crop", "field",
        # measurements and subjects, in all three registers
        "rain", "snow", "wind", "weather", "sun", "storm", "temperature",
        "humidity", "forecast", "precipitation", "visibility", "pressure",
        "मौसम", "बारिश", "वर्षा", "तापमान", "हवा", "बादल", "धूप", "गर्मी", "ठंड",
        "नमी", "दृश्यता", "संभावना", "गति", "मात्रा", "पूर्वानुमान", "चेतावनी",
        "mausam", "mosam", "baarish", "barish", "varsha", "tapman", "taapman",
        "hawa", "hava", "garmi", "sardi", "nami", "badal", "dhoop",
        # time, in all three registers
        "today", "tomorrow", "yesterday", "tonight", "now", "week", "month",
        "day", "days", "hour", "hours", "minute", "minutes", "morning",
        "evening", "afternoon", "night",
        "आज", "कल", "परसों", "अभी", "दिन", "दिनों", "घंटे", "घंटा", "मिनट",
        "सुबह", "शाम", "रात", "दोपहर", "हफ्ते", "सप्ताह", "अगले", "पिछले",
        "aaj", "kal", "parso", "parson", "abhi", "din", "ghante", "ghanta",
        "subah", "sham", "shaam", "raat", "dopahar", "hafte", "agle", "agale",
        "pichle", "pichhle", "baad", "pehle", "pahle",
        # question words
        "कितनी", "कितना", "कितने", "कैसा", "कैसी", "कैसे", "क्या", "कौन",
        "kitna", "kitni", "kitne", "kaisa", "kaisi", "kaise", "kya",
    }
)

#: Words that name a *feature at* a place rather than a place. "Vizag harbour"
#: and "Visakhapatnam port" are questions about Visakhapatnam, but a gazetteer
#: asked for the whole phrase returns nothing at all — and an empty result is
#: how a question about the right city silently becomes a question about
#: whatever the session was already on.
#:
#: Stripped only from the end, and never down to nothing, so a name that opens
#: with one of these survives intact: "Port Blair" keeps its Port, and so does
#: "Lake Placid".
_PLACE_FEATURE = frozenset(
    {
        "harbour", "harbor", "port", "docks", "dock", "jetty", "quay", "pier",
        "marina", "beach", "coast", "shore", "shoreline",
        "airport", "aerodrome", "station", "terminal", "junction",
        "city", "town", "village", "district", "taluk", "tehsil", "mandal",
        "region", "area", "outskirts",
    }
)

_ISO_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")

#: An explicit request for the numerical model, read from the user's own words.
#: A provider is never taken from a model's tool arguments: naming the source is
#: a claim about where a number came from, and only the user may make it.
_NWP_REQUESTED = re.compile(r"\b(?:gfs|nwp|noaa)\b", re.IGNORECASE)
GFS_PROVIDER = "noaa-gfs"

#: Phrasing that asks for the *hour by hour* shape of a period rather than its
#: daily summary.
_HOURLY_CUE = re.compile(
    r"\b(?:hourly|each\s+hour|per\s+hour|hour\s+by\s+hour|प्रति\s*घंटा|ghante\s*ghante)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DetectedRequest:
    """Everything the orchestrator needs, all of it read from the message."""

    intent: Intent
    location: LocationInput | None
    days: int = 7
    start: date | None = None
    end: date | None = None
    #: Hour-precision window, when the question was about a specific moment.
    start_time: datetime | None = None
    end_time: datetime | None = None
    #: Local hours of day the answer should be narrowed to, e.g. "yesterday
    #: evening" -> (17, 21).
    local_hours: tuple[int, int] | None = None
    purpose: AdvisoryPurpose = AdvisoryPurpose.GENERAL
    language: str = "en"
    provider: str | None = None
    #: The measurement the question is about, when it named one.
    variable: str | None = None
    tense: Tense = Tense.PRESENT
    #: Forward horizon in hours, for a forecast request.
    horizon_hours: int = 0
    #: A past time the person mentioned as context rather than as the target.
    event_phrase: str | None = None


class IntentDetector:
    """Portable, inspectable routing for the baseline and LLM fallback path.

    Arguments are derived from the user message and the session context instead
    of from generated function arguments. A model may assist with function
    selection, but cannot turn a vague question into a silently guessed place,
    date, warning, or weather value.
    """

    def detect(
        self,
        message: str,
        state: ConversationState,
        *,
        language_hint: str | None = None,
        today: date | None = None,
        now: datetime | None = None,
    ) -> DetectedRequest:
        normalized = " ".join(message.split())
        lower = normalized.casefold()
        moment = now or (
            datetime.combine(today, datetime.min.time(), tzinfo=timezone.utc)
            if today
            else datetime.now(timezone.utc)
        )

        language = detect_language(normalized, hint=language_hint, previous=state.language)
        purpose = _purpose(normalized, state.purpose)
        location = _location(normalized, state)
        provider = GFS_PROVIDER if _NWP_REQUESTED.search(normalized) else None
        variable = _variable(normalized)
        when = temporal_plan(normalized, now=moment)

        common = {
            "purpose": purpose,
            "language": language,
            "provider": provider,
            "variable": variable,
            "tense": when.tense,
            "horizon_hours": when.horizon_hours,
            "event_phrase": when.event.phrase if when.event else None,
        }

        # Existing WeatherGPT rule alerts. Left exactly as they were.
        if ALERT_SUBJECT.search(normalized):
            return DetectedRequest(Intent.ALERTS, location, days=when.days, **common)

        # Tense decides before anything else, because it decides which *data*
        # can answer the question at all. A planning question that names a time
        # is a question about that time; only a planning question with no clock
        # in it falls through to the composed risk view.
        if when.tense is Tense.PAST:
            return DetectedRequest(
                Intent.HISTORICAL_WEATHER,
                location,
                days=1,
                **_past_window(when),
                **common,
            )

        if when.tense is Tense.FUTURE:
            # A counted window ("the next 48 hours") is answered hour by hour;
            # a named day ("tomorrow") is answered as a day, because an hourly
            # answer to it leads with the rest of today.
            hourly = when.hourly or bool(_HOURLY_CUE.search(normalized))
            return DetectedRequest(
                Intent.HOURLY_FORECAST if hourly else Intent.DAILY_FORECAST,
                location,
                days=when.days,
                local_hours=when.target.local_hours if when.target else None,
                **common,
            )

        if RISK_SUBJECT.search(normalized) or (
            purpose is not AdvisoryPurpose.GENERAL and WEATHER_SUBJECT.search(normalized)
        ):
            return DetectedRequest(Intent.LOCATION_RISK, location, days=when.days, **common)

        if WEATHER_SUBJECT.search(normalized) or variable is not None:
            return DetectedRequest(Intent.CURRENT_WEATHER, location, days=when.days, **common)

        if _SEARCH_TARGET.search(normalized):
            return DetectedRequest(Intent.LOCATION_SEARCH, location, **common)

        # A message whose only new information is a place changes the place,
        # not the subject: after "what is the weather in Hyderabad?", "what
        # about Miyapur?" is still that question. Identity is the test —
        # `_location` hands back the session's own object when it inherits, so
        # a different object means this message named somewhere itself.
        named_here = location is not None and location is not state.location
        if named_here:
            carried = _carry_forward(state.last_intent)
            if carried is not None:
                return DetectedRequest(carried, location, days=when.days, **common)
            # Nothing to carry forward, so the place itself is the question.
            if _is_place_only(normalized):
                return DetectedRequest(Intent.LOCATION_SEARCH, location, **common)

        return DetectedRequest(Intent.UNKNOWN, location, days=when.days, **common)


#: Questions a bare place name can inherit. A search or an unknown carries
#: nothing forward, because there is nothing to ask again about the new place.
_CARRIED_FORWARD = frozenset(
    {
        Intent.CURRENT_WEATHER,
        Intent.HOURLY_FORECAST,
        Intent.DAILY_FORECAST,
        Intent.HISTORICAL_WEATHER,
        Intent.LOCATION_RISK,
        Intent.ALERTS,
    }
)


def _carry_forward(last: Intent) -> Intent | None:
    """The question to re-ask about a newly named place, if there is one."""
    return last if last in _CARRIED_FORWARD else None


def _past_window(when: TemporalPlan) -> dict[str, object]:
    """Turn a resolved past target into the arguments the history tool takes.

    An hour-precision question carries timestamps; a calendar one carries dates
    and is resolved against the location's own local day further down, because
    a UTC midnight is the wrong midnight for the person asking.
    """
    target = when.target
    if target is None:
        return {}
    return {
        "start": target.start_date,
        "end": target.end_date,
        "start_time": target.start,
        "end_time": target.end,
        "local_hours": target.local_hours,
    }


# ------------------------------------------------------------------ location


def _location(message: str, state: ConversationState) -> LocationInput | None:
    """The place this message is about, or the one the conversation is on.

    A place named in the message always wins: someone on Hyderabad who asks
    about Mumbai has asked about Mumbai.
    """
    coordinates = _COORDINATES.search(message)
    if coordinates:
        latitude = float(coordinates.group("latitude"))
        longitude = float(coordinates.group("longitude"))
        if -90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0:
            return LocationInput(latitude=latitude, longitude=longitude)

    # Hindi marks its places from behind. Only consulted for a message that is
    # actually in Hindi: "ka", "me" and "par" are all English words too, and
    # reading them as postpositions in an English sentence invents places.
    if DEVANAGARI.search(message) or is_hinglish(message):
        for pattern in (_POSTPOSITION_DEV, _POSTPOSITION_ROMAN):
            for match in pattern.finditer(message):
                candidate = _clean_place(match.group("place"))
                if candidate:
                    return LocationInput(location=candidate)

    for pattern in (_PLACED, _DESTINATION, _SEARCH_TARGET):
        match = pattern.search(message)
        if match:
            candidate = _clean_place(match.group("place"))
            if candidate:
                return LocationInput(location=candidate)

    if _is_place_only(message):
        candidate = _clean_place(message)
        if candidate:
            return LocationInput(location=candidate)

    # No place in this message, so the question is about the place the
    # conversation is already on. That is the whole of the priority order: a
    # named place wins, and everything else inherits.
    return state.location


def _clean_place(value: str) -> str:
    """Reduce a captured phrase to the place inside it, or to nothing."""
    candidate = re.sub(
        r"\b(?:today|tomorrow|tonight|this afternoon|this evening|next week|"
        r"this week|(?:this |next )?weekend|hourly|forecast|weather|alerts?|warnings?|risk|"
        # "Hyderabad right now" is a question about Hyderabad. These say *when*
        # and never *where*, so they are cut with everything after them rather
        # than left to be popped a word at a time — "now" would go and "right"
        # would stay, leaving a name no gazetteer has.
        r"right now|just now|at the moment|at present|currently|presently|"
        # A duration is not a place: "New Delhi for 3 days" must not resolve to
        # "3 days". The optional preposition is consumed with it. Every unit is
        # listed, because "Chennai next 12 hours" fails the same way "next 12
        # days" would, and the bare number left behind is not popped: a place
        # may legitimately end in one ("Sector 15").
        r"for travel|for farming|"
        r"(?:for\s+|over\s+|next\s+|the\s+next\s+|coming\s+|upcoming\s+)?"
        r"\d{1,3}\s+(?:days?|hours?|hrs?|minutes?|mins?|weeks?|months?))\b.*$",
        "",
        value,
        flags=re.IGNORECASE,
    ).strip(" \t-'\"?!.,")
    # A dangling preposition left by the cut above ("New Delhi for").
    candidate = re.sub(
        r"\s+(?:for|in|at|near|around|on|over|next)$", "", candidate, flags=re.IGNORECASE
    ).strip()
    candidate = _innermost(candidate)

    # Drop leading words that are not part of the name: "kal Delhi" -> "Delhi".
    tokens = candidate.split()
    while len(tokens) > 1 and tokens[0].casefold() in _NON_PLACE:
        tokens.pop(0)
    while tokens and tokens[-1].casefold() in _NON_PLACE:
        tokens.pop()
    # The feature the place carries is context, not part of its name. Dropped
    # last, and only while a name remains, so the gazetteer is asked about the
    # place rather than about the dock in it.
    while len(tokens) > 1 and tokens[-1].casefold().strip("'s") in _PLACE_FEATURE:
        tokens.pop()
    candidate = " ".join(tokens)

    if not candidate or candidate.casefold() in _NON_PLACE:
        return ""
    # A quantity is never a place.
    if candidate[0].isdigit():
        return ""
    return candidate[:200]


def _innermost(place: str) -> str:
    """Prefer the place itself over the thing sitting in it.

    "my farm in Nashik" is a geocoder miss; "Nashik" is a hit. A nested
    preposition means the outer words describe a feature at the location, not
    the location, so the innermost phrase is the one to send.
    """
    while True:
        nested = _PLACED.search(place)
        if nested is None:
            return place
        inner = nested.group("place").strip(" \t-'\"")
        # Only descend into something that could be a place. A quantity or a
        # filler word means the outer phrase was the real answer.
        if not inner or inner[0].isdigit() or inner.casefold() in _NON_PLACE:
            return place
        place = inner


def _is_place_only(message: str) -> bool:
    """True when the whole message is a bare place name.

    "Miyapur" is a complete question in a conversation about the weather, and
    answering it with "which location did you mean?" is the right response.
    """
    words = message.strip(" ?!.,").split()
    if not (1 <= len(words) <= 4):
        return False
    if WEATHER_SUBJECT.search(message) or ALERT_SUBJECT.search(message):
        return False
    if any(term.search(message) for term in VARIABLES.values()):
        return False
    return not any(word.casefold() in _NON_PLACE for word in words)


# ------------------------------------------------------------------ context


def _variable(message: str) -> str | None:
    """The measurement the question named, if it named one."""
    for name, term in VARIABLES.items():
        if term.search(message):
            return name
    return None


def _purpose(message: str, previous: AdvisoryPurpose) -> AdvisoryPurpose:
    """The context that changes what a useful answer looks like."""
    for name, term in PURPOSE_TERMS.items():
        if term.search(message):
            return AdvisoryPurpose(name)
    lowered = message.casefold()
    if lowered in {"what about it", "and tomorrow", "how about tomorrow"}:
        return previous
    return AdvisoryPurpose.GENERAL
