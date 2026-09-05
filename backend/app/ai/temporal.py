"""When a question is about.

The hard part is not recognising "yesterday". It is knowing that in

    "I sprayed pesticides yesterday. Is rain expected in the next 48 hours?"

"yesterday" is when *the person did something* and "the next 48 hours" is when
*the weather matters*. Reading the first as the weather target answers a
question nobody asked, with real data, confidently. So every time expression in
a message is extracted with its tense, and then one of them is chosen as the
**target** while the rest become **event context**.

The choice is one rule, and it is the whole of the multi-temporal behaviour:

    a stated future beats a stated past, which beats the present.

A person who mentions a future time is asking about the future — the past
clause is background they are giving you. Only when nothing points forward does
a past reference become the thing being asked about.

Everything here is deterministic and locale-aware but timezone-honest: an
hour-precision window is computed in UTC, while a calendar reference such as
"yesterday" stays a *date* and is resolved against the location's own local day
further down, because a UTC day boundary is the wrong midnight for the person
asking.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from enum import Enum

from app.ai.language import devanagari_word


class Tense(str, Enum):
    PAST = "past"
    PRESENT = "present"
    FUTURE = "future"


@dataclass(frozen=True, slots=True)
class TimeReference:
    """One time expression found in a message."""

    tense: Tense
    phrase: str
    position: int
    #: False for a bare verb cue ("will", "was"), which sets a direction but
    #: names no time. An explicit reference always outranks a cue.
    explicit: bool
    granularity: str = "day"  # "hour" | "day"
    #: Hour-precision past window, in UTC.
    start: datetime | None = None
    end: datetime | None = None
    #: Calendar past window, resolved against the location's local day.
    start_date: date | None = None
    end_date: date | None = None
    #: Local hours of day this reference restricts to, e.g. morning = (5, 12).
    local_hours: tuple[int, int] | None = None
    #: Forward horizon, in hours.
    horizon_hours: int | None = None


@dataclass(frozen=True, slots=True)
class TemporalPlan:
    """The resolved answer to "when is this question about?"."""

    tense: Tense
    target: TimeReference | None
    #: A past time the person mentioned as context rather than as the target.
    event: TimeReference | None
    #: True when the answer needs hour-by-hour resolution rather than daily.
    hourly: bool
    #: Forecast horizon in hours, for a future plan.
    horizon_hours: int
    #: Calendar days of forecast to request, inclusive of today.
    days: int

    @property
    def is_past(self) -> bool:
        return self.tense is Tense.PAST

    @property
    def is_future(self) -> bool:
        return self.tense is Tense.FUTURE


# ------------------------------------------------------------------- units

_HOURS_PER = {
    "minute": 1 / 60, "minutes": 1 / 60, "min": 1 / 60, "mins": 1 / 60,
    "hour": 1, "hours": 1, "hr": 1, "hrs": 1,
    "day": 24, "days": 24,
    "week": 168, "weeks": 168,
    # Devanagari
    "मिनट": 1 / 60, "घंटे": 1, "घंटा": 1, "घंटों": 1, "दिन": 24, "दिनों": 24,
    "हफ्ते": 168, "हफ़्ते": 168, "सप्ताह": 168,
    # Romanised
    "ghante": 1, "ghanta": 1, "ghanto": 1, "ghanton": 1,
    "din": 24, "dino": 24, "dinon": 24, "hafte": 168, "haftey": 168, "saptah": 168,
}

_UNIT_ALTERNATION = "|".join(sorted(_HOURS_PER, key=len, reverse=True))

#: People write "the next three days" as readily as "the next 3 days", and in
#: three languages. A counted expression that only reads digits silently drops
#: half of them.
_NUMBER_WORDS: dict[str, int] = {
    "a": 1, "an": 1, "one": 1, "couple": 2, "few": 3, "several": 3,
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "twenty-four": 24, "twenty four": 24, "forty-eight": 48, "forty eight": 48,
    # Devanagari
    "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पांच": 5, "पाँच": 5, "छह": 6,
    "सात": 7, "आठ": 8, "नौ": 9, "दस": 10, "कुछ": 3,
    # Romanised
    "ek": 1, "do": 2, "teen": 3, "char": 4, "paanch": 5, "panch": 5,
    "chhah": 6, "chhe": 6, "saat": 7, "aath": 8, "nau": 9, "das": 10,
    "kuch": 3,
}

_COUNT = r"\d{1,3}|" + "|".join(
    re.escape(word) for word in sorted(_NUMBER_WORDS, key=len, reverse=True)
)


def _count(text: str) -> int:
    """Read a count written as digits or as a word, in any of the three registers."""
    token = text.strip().casefold()
    if token.isdigit():
        return int(token)
    return _NUMBER_WORDS.get(token, _NUMBER_WORDS.get(text.strip(), 1))

#: Parts of the day, as local hours. Used to narrow a calendar reference.
_DAYPARTS: dict[str, tuple[int, int]] = {
    "morning": (5, 12), "सुबह": (5, 12), "subah": (5, 12),
    "afternoon": (12, 17), "दोपहर": (12, 17), "dopahar": (12, 17),
    "evening": (17, 21), "शाम": (17, 21), "sham": (17, 21), "shaam": (17, 21),
    "night": (21, 24), "रात": (21, 24), "raat": (21, 24),
}
_DAYPART_ALTERNATION = "|".join(sorted(_DAYPARTS, key=len, reverse=True))

_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}

# ---------------------------------------------------------------- patterns
#
# Latin-script patterns cover English and romanised Hindi together: they share
# an alphabet and, in a mixed sentence, a clause.

_ISO_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")

# --- past ---
_AGO = re.compile(
    rf"\b({_COUNT})\s*({_UNIT_ALTERNATION})\s*(?:ago|back|earlier|pehle|pahle|before)\b",
    re.IGNORECASE,
)
_AGO_DEV = re.compile(rf"({_COUNT})\s*({_UNIT_ALTERNATION})\s*पहले")
_LAST_N = re.compile(
    rf"\b(?:last|past|previous|pichh?le|beete?)\s+({_COUNT})\s*({_UNIT_ALTERNATION})\b",
    re.IGNORECASE,
)
_LAST_N_DEV = re.compile(rf"(?:पिछले|बीते|गत)\s*({_COUNT})\s*({_UNIT_ALTERNATION})")
_LAST_PERIOD = re.compile(
    r"\b(?:last|past|previous|pichh?le)\s+(week|month|night)\b", re.IGNORECASE
)
_LAST_PERIOD_DEV = re.compile(r"(?:पिछले|पिछली|बीते)\s*(हफ्ते|सप्ताह|महीने|रात)")
_LAST_WEEKDAY = re.compile(
    rf"\b(?:last|past)\s+({'|'.join(_WEEKDAYS)})\b", re.IGNORECASE
)
_FUTURE_WEEKDAY = re.compile(
    rf"\b(?:(?:this|next|coming|following)\s+)?({'|'.join(_WEEKDAYS)})\b",
    re.IGNORECASE,
)
_CLOCK = re.compile(
    r"\b(?:around|at|by)?\s*(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<meridiem>a\.?m\.?|p\.?m\.?)\b",
    re.IGNORECASE,
)
_EARLIER_TODAY = re.compile(
    r"\b(?:earlier\s+today|this\s+morning|earlier\s+this\s+morning|aaj\s+subah)\b",
    re.IGNORECASE,
)
_EARLIER_TODAY_DEV = re.compile(r"आज\s*सुबह")

# --- future ---
_IN_N = re.compile(
    rf"\b(?:in|within|after|over\s+the\s+next|in\s+the\s+next|for\s+the\s+next|next|"
    rf"agle|agale|agli|agali|aane\s+wale)\s+({_COUNT})\s*({_UNIT_ALTERNATION})\b",
    re.IGNORECASE,
)
_IN_N_DEV = re.compile(
    rf"(?:अगले|अगली|आने\s*वाले|आने\s*वाली)\s*({_COUNT})\s*({_UNIT_ALTERNATION})"
)
_N_LATER = re.compile(
    rf"\b({_COUNT})\s*({_UNIT_ALTERNATION})\s*(?:from\s+now|later|baad|ke\s+baad)\b",
    re.IGNORECASE,
)
_N_LATER_DEV = re.compile(rf"({_COUNT})\s*({_UNIT_ALTERNATION})\s*(?:बाद|में)")
_NEXT_PERIOD = re.compile(
    r"\b(?:next|coming|upcoming|agle|agale)\s+(week|weekend|month|few\s+days)\b",
    re.IGNORECASE,
)
_NEXT_PERIOD_DEV = re.compile(r"(?:अगले|आने\s*वाले)\s*(हफ्ते|सप्ताह|सप्ताहांत|महीने|कुछ\s*दिन)")

_TOMORROW = re.compile(
    rf"\btomorrow(?:\s+({_DAYPART_ALTERNATION}))?\b", re.IGNORECASE
)
_DAY_AFTER = re.compile(
    r"\b(?:day\s+after\s+tomorrow|parso|parson|parsoo)\b", re.IGNORECASE
)
_DAY_AFTER_DEV = devanagari_word("परसों")
_BARE_DAYS = re.compile(
    r"\b(\d{1,2})[-\s]?(?:day|days|din|dino|dinon|दिन|दिनों)\b"
    r"(?!\s*(?:ago|back|earlier|pehle|pahle|पहले))",
    re.IGNORECASE,
)
_TONIGHT = re.compile(
    r"\b(?:tonight|later\s+tonight|this\s+evening|later\s+today|this\s+afternoon)\b",
    re.IGNORECASE,
)
_YESTERDAY = re.compile(
    rf"\byesterday(?:\s+({_DAYPART_ALTERNATION}))?\b", re.IGNORECASE
)

# --- present ---
_NOW = re.compile(
    r"\b(?:right\s+now|now|currently|at\s+the\s+moment|at\s+present|presently|"
    r"outside|abhi|is\s+waqt)\b",
    re.IGNORECASE,
)
_NOW_DEV = re.compile(r"अभी|इस\s*समय|वर्तमान")
_TODAY = re.compile(r"\b(?:today|aaj)\b", re.IGNORECASE)
_TODAY_DEV = devanagari_word("आज")

# --- verb cues: a direction with no named time ---
_WILL = re.compile(
    r"\b(?:will|won't|wont|shall|going\s+to|gonna|expected|expecting|expect|"
    r"forecast|forecasted|upcoming|ahead|hogi|hoga|honge|rahegi|rahega|"
    r"chalegi|chalega|barsegi|milega)\b",
    re.IGNORECASE,
)
_WILL_DEV = re.compile(r"होगी|होगा|होंगे|रहेगी|रहेगा|बरसेगी|बरसेगा|मिलेगा|आएगी|आएगा")
#: Romanised "the" (they were) is deliberately absent: it collides with the
#: English article, which appears in nearly every question, and a false past
#: cue silently turns a question about now into a question about the past.
_WAS = re.compile(
    r"\b(?:was|were|did|had|has\s+been|have\s+been|recorded|fell|rained|"
    r"hui|huyi|hua|thi|tha)\b",
    re.IGNORECASE,
)
_WAS_DEV = re.compile(r"हुई|हुआ|हुए|थी|था|थे|रही\s*थी|दर्ज")

#: "कल" and "kal" mean both yesterday and tomorrow; only the verb says which.
_KAL_DEV = devanagari_word("कल")
_KAL_ROMAN = re.compile(r"\bkal\b", re.IGNORECASE)


# ---------------------------------------------------------------- helpers


def _hours(count: str, unit: str) -> float:
    return float(_count(count)) * _HOURS_PER[unit.lower()]


def _past_window(now: datetime, hours: float) -> TimeReference:
    """A past instant, widened enough to catch the nearest hourly record.

    An hourly series has a value at the top of each hour, so a question about
    "6 hours ago" has to accept the reading either side of it. Ninety minutes
    is the smallest window that always contains one.
    """
    instant = now - timedelta(hours=hours)
    day_scale = hours >= 48
    return TimeReference(
        tense=Tense.PAST,
        phrase=f"{hours:g}h ago",
        position=0,
        explicit=True,
        granularity="day" if day_scale else "hour",
        start=instant - timedelta(minutes=90),
        end=instant + timedelta(minutes=90),
        start_date=(now - timedelta(hours=hours)).date() if day_scale else None,
        end_date=now.date() if day_scale else None,
    )


def _calendar(
    tense: Tense,
    phrase: str,
    position: int,
    start_date: date,
    end_date: date,
    daypart: str | None = None,
) -> TimeReference:
    return TimeReference(
        tense=tense,
        phrase=phrase,
        position=position,
        explicit=True,
        granularity="hour" if daypart else "day",
        start_date=start_date,
        end_date=end_date,
        local_hours=_DAYPARTS.get((daypart or "").lower()),
    )


def _forward(
    phrase: str, position: int, hours: float, *, granularity: str | None = None
) -> TimeReference:
    """A forward-looking reference.

    Granularity is what a counted window implies, not what its length implies:
    "in the next 12 hours" wants the hours, "tomorrow" wants the day. Answering
    "tomorrow" hour by hour shows the rest of *today* first, which is a precise
    answer to a question nobody asked.
    """
    horizon = max(1, min(int(math.ceil(hours)), 16 * 24))
    return TimeReference(
        tense=Tense.FUTURE,
        phrase=phrase,
        position=position,
        explicit=True,
        granularity=granularity or ("hour" if horizon <= HOURLY_HORIZON_HOURS else "day"),
        horizon_hours=horizon,
    )


def _clock_window(message: str) -> tuple[int, int] | None:
    match = _CLOCK.search(message)
    if match is None:
        return None
    hour = int(match.group("hour"))
    meridiem = match.group("meridiem").replace(".", "").lower()
    if not 1 <= hour <= 12:
        return None
    if meridiem == "pm" and hour != 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0
    return hour, min(hour + 1, 24)


# ---------------------------------------------------------------- extraction


def references(message: str, *, now: datetime | None = None) -> list[TimeReference]:
    """Every time expression in a message, in the order they appear."""
    moment = now or datetime.now(timezone.utc)
    today = moment.date()
    found: list[TimeReference] = []

    def add(reference: TimeReference, at: int) -> None:
        found.append(
            TimeReference(
                tense=reference.tense,
                phrase=reference.phrase,
                position=at,
                explicit=reference.explicit,
                granularity=reference.granularity,
                start=reference.start,
                end=reference.end,
                start_date=reference.start_date,
                end_date=reference.end_date,
                local_hours=reference.local_hours,
                horizon_hours=reference.horizon_hours,
            )
        )

    # --- explicit past ---------------------------------------------------
    for pattern in (_AGO, _AGO_DEV):
        for match in pattern.finditer(message):
            add(_past_window(moment, _hours(match.group(1), match.group(2))), match.start())

    for pattern in (_LAST_N, _LAST_N_DEV):
        for match in pattern.finditer(message):
            hours = _hours(match.group(1), match.group(2))
            start = (moment - timedelta(hours=hours)).date()
            add(_calendar(Tense.PAST, match.group(0), 0, start, today), match.start())

    for match in _LAST_PERIOD.finditer(message):
        word = match.group(1).lower()
        span = {"week": 7, "month": 30, "night": 1}[word]
        add(
            _calendar(
                Tense.PAST, match.group(0), 0,
                today - timedelta(days=span), today - timedelta(days=1) if word == "night" else today,
                daypart="night" if word == "night" else None,
            ),
            match.start(),
        )

    for match in _LAST_PERIOD_DEV.finditer(message):
        span = 30 if "महीने" in match.group(1) else (1 if "रात" in match.group(1) else 7)
        add(
            _calendar(Tense.PAST, match.group(0), 0, today - timedelta(days=span), today),
            match.start(),
        )

    for match in _LAST_WEEKDAY.finditer(message):
        target = _WEEKDAYS[match.group(1).lower()]
        delta = (today.weekday() - target) % 7 or 7
        day = today - timedelta(days=delta)
        add(_calendar(Tense.PAST, match.group(0), 0, day, day), match.start())

    for pattern in (_EARLIER_TODAY, _EARLIER_TODAY_DEV):
        for match in pattern.finditer(message):
            add(_calendar(Tense.PAST, match.group(0), 0, today, today, daypart="morning"), match.start())

    for match in _YESTERDAY.finditer(message):
        day = today - timedelta(days=1)
        part = match.group(1) if match.lastindex else None
        add(_calendar(Tense.PAST, match.group(0), 0, day, day, daypart=part), match.start())

    for match in _ISO_DATE.finditer(message):
        try:
            day = date.fromisoformat(match.group(1))
        except ValueError:
            continue
        tense = Tense.PAST if day <= today else Tense.FUTURE
        if tense is Tense.PAST:
            add(_calendar(Tense.PAST, match.group(1), 0, day, day), match.start())
        else:
            hours = (day - today).days * 24 + 24
            add(_forward(match.group(1), 0, hours), match.start())

    # --- explicit future -------------------------------------------------
    for pattern in (_IN_N, _IN_N_DEV, _N_LATER, _N_LATER_DEV):
        for match in pattern.finditer(message):
            add(_forward(match.group(0), 0, _hours(match.group(1), match.group(2))), match.start())

    for match in _NEXT_PERIOD.finditer(message):
        word = match.group(1).lower()
        add(_forward(match.group(0), 0, 24 * (30 if "month" in word else 7)), match.start())

    for match in _NEXT_PERIOD_DEV.finditer(message):
        add(_forward(match.group(0), 0, 24 * (30 if "महीने" in match.group(1) else 7)), match.start())

    for match in _FUTURE_WEEKDAY.finditer(message):
        if match.group(1).lower() in {"last", "past"}:
            continue
        delta = (_WEEKDAYS[match.group(1).lower()] - today.weekday()) % 7 or 7
        clock = _clock_window(message)
        add(
            TimeReference(
                tense=Tense.FUTURE,
                phrase=match.group(0),
                position=match.start(),
                explicit=True,
                granularity="hour" if clock else "day",
                local_hours=clock,
                horizon_hours=delta * 24 + 24,
                start_date=today + timedelta(days=delta),
                end_date=today + timedelta(days=delta),
            ),
            match.start(),
        )

    for match in _TOMORROW.finditer(message):
        part = match.group(1) if match.lastindex else None
        reference = _forward(match.group(0), 0, 48, granularity="day")
        if part:
            reference = TimeReference(
                tense=Tense.FUTURE, phrase=match.group(0), position=0, explicit=True,
                granularity="hour", horizon_hours=48,
                local_hours=_DAYPARTS.get(part.lower()),
                start_date=today + timedelta(days=1), end_date=today + timedelta(days=1),
            )
        add(reference, match.start())

    for pattern in (_DAY_AFTER, _DAY_AFTER_DEV):
        for match in pattern.finditer(message):
            add(_forward(match.group(0), 0, 72, granularity="day"), match.start())

    for match in _TONIGHT.finditer(message):
        add(_forward(match.group(0), 0, 12), match.start())

    # "7 day forecast", "3 din ka forecast" — a horizon with no preposition to
    # anchor it. Only read when nothing in the sentence already pointed
    # backwards, because "over the last 3 days" contains the same two words.
    if not any(r.tense is Tense.PAST and r.explicit for r in found):
        for match in _BARE_DAYS.finditer(message):
            add(_forward(match.group(0), 0, int(match.group(1)) * 24), match.start())

    # "कल"/"kal" is yesterday or tomorrow; the verb in the sentence decides.
    for pattern in (_KAL_DEV, _KAL_ROMAN):
        for match in pattern.finditer(message):
            if _past_voice(message):
                day = today - timedelta(days=1)
                add(_calendar(Tense.PAST, match.group(0), 0, day, day), match.start())
            else:
                add(_forward(match.group(0), 0, 48, granularity="day"), match.start())

    # --- present ---------------------------------------------------------
    for pattern in (_NOW, _NOW_DEV):
        for match in pattern.finditer(message):
            add(TimeReference(Tense.PRESENT, match.group(0), 0, True), match.start())

    for pattern in (_TODAY, _TODAY_DEV):
        for match in pattern.finditer(message):
            add(TimeReference(Tense.PRESENT, match.group(0), 0, True), match.start())

    # --- verb cues -------------------------------------------------------
    for pattern in (_WILL, _WILL_DEV):
        match = pattern.search(message)
        if match:
            add(TimeReference(Tense.FUTURE, match.group(0), 0, False, horizon_hours=24), match.start())
            break

    for pattern in (_WAS, _WAS_DEV):
        match = pattern.search(message)
        if match:
            add(TimeReference(Tense.PAST, match.group(0), 0, False), match.start())
            break

    found.sort(key=lambda reference: reference.position)
    return found


def _past_voice(message: str) -> bool:
    """True when the sentence is told in the past tense."""
    return bool(_WAS_DEV.search(message) or _WAS.search(message))


# ----------------------------------------------------------------- planning

#: Below this a forecast is answered hour by hour rather than day by day.
HOURLY_HORIZON_HOURS = 48


def plan(message: str, *, now: datetime | None = None) -> TemporalPlan:
    """Decide what time the *weather* question is about.

    A stated future beats a stated past, which beats the present. Everything
    the target does not claim, and that points backwards, becomes event
    context — the thing that happened, not the thing being asked about.
    """
    moment = now or datetime.now(timezone.utc)
    found = references(message, now=moment)

    explicit = [reference for reference in found if reference.explicit]
    futures = [r for r in explicit if r.tense is Tense.FUTURE]
    pasts = [r for r in explicit if r.tense is Tense.PAST]
    presents = [r for r in explicit if r.tense is Tense.PRESENT]
    cues = [reference for reference in found if not reference.explicit]

    target: TimeReference | None = None
    event: TimeReference | None = None

    if futures:
        # The furthest stated horizon is the one that has to be covered: a
        # question about the next 48 hours is not answered by the next 6.
        target = max(futures, key=lambda r: r.horizon_hours or 0)
        event = pasts[0] if pasts else None
    elif pasts:
        target = pasts[0]
    elif presents:
        target = presents[0]
    else:
        forward = next((c for c in cues if c.tense is Tense.FUTURE), None)
        backward = next((c for c in cues if c.tense is Tense.PAST), None)
        target = forward or backward

    if target is None:
        return TemporalPlan(Tense.PRESENT, None, None, hourly=False, horizon_hours=0, days=7)

    # A present reference alongside a forward-looking verb ("will it rain
    # today?") is a question about the rest of the day, not about right now.
    if target.tense is Tense.PRESENT and any(c.tense is Tense.FUTURE for c in cues):
        target = _forward(target.phrase, target.position, 24)

    if target.tense is Tense.FUTURE:
        horizon = target.horizon_hours or 24
        return TemporalPlan(
            tense=Tense.FUTURE,
            target=target,
            event=event,
            hourly=target.granularity == "hour",
            horizon_hours=horizon,
            days=max(1, min(math.ceil(horizon / 24) + 1, 16)),
        )

    if target.tense is Tense.PAST:
        return TemporalPlan(
            tense=Tense.PAST,
            target=target,
            event=None,
            hourly=target.granularity == "hour",
            horizon_hours=0,
            days=1,
        )

    return TemporalPlan(Tense.PRESENT, target, None, hourly=False, horizon_hours=0, days=7)
