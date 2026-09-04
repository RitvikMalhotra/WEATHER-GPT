"""The facts a recommendation is allowed to be built from.

A verdict has to answer the question that was asked — "should I spray today?",
not "here is a reading of the rainfall probability". Answering it takes two
things that used to be missing here: the question itself, and the retrieved
weather reduced to the handful of figures a decision actually turns on.

This module produces the second. It reads a backend response and emits a
:class:`Brief`: peak rainfall probability over the window that was asked about,
when rain first appears, how much, peak wind, the temperature range, the
conditions reported, and — the part that matters most — which variables
relevant to *this* question were **not** returned.

Two properties are load-bearing:

* **Every number in a Brief was returned by the backend in this turn.** Nothing
  is derived beyond summing hourly accumulations, which is arithmetic on
  retrieved values, and taking maxima and minima of them.
* **A Brief carries its own numeric vocabulary.** ``numbers`` holds every
  figure that may legitimately appear in an answer built from it, which is what
  lets a generated sentence be checked for invented values rather than trusted.

Nothing here decides anything. It is the evidence; the recommendation is made
by :mod:`app.ai.recommend` and the safety-critical thresholds stay in the
deterministic alert engine, which this module never touches.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Sequence

from app.ai.models import AdvisoryPurpose, Intent
from app.domain.forecast import Forecast
from app.domain.weather import CurrentWeather, WeatherCondition, WeatherReport

#: Variables a question in each domain turns on. Used only to report what was
#: missing — never to invent a value or to block an answer.
_RELEVANT: dict[AdvisoryPurpose, tuple[str, ...]] = {
    AdvisoryPurpose.MARINE: ("wind", "precipitation", "wave height", "sea state"),
    AdvisoryPurpose.AGRICULTURE: ("precipitation", "wind", "temperature"),
    AdvisoryPurpose.TRAVEL: ("precipitation", "wind", "visibility"),
    AdvisoryPurpose.GENERAL: ("precipitation", "temperature"),
}

#: Marine variables no configured provider returns. Named so the absence can be
#: stated as a fact about this deployment rather than guessed at.
_MARINE_FIELDS = ("wave_height_m", "swell_height_m", "sea_state", "wave_period_s")

_STORMY = {WeatherCondition.THUNDERSTORM, WeatherCondition.THUNDERSTORM_WITH_HAIL}


@dataclass(frozen=True)
class Brief:
    """Retrieved weather, reduced to what a decision turns on."""

    question: str
    domain: AdvisoryPurpose
    intent: Intent
    place: str
    window_hours: int
    #: True when the window is in the past, so a recommendation becomes a reading.
    past: bool = False

    rain_peak_pct: float | None = None
    rain_first_at: datetime | None = None
    rain_total_mm: float | None = None
    wind_peak_ms: float | None = None
    gust_peak_ms: float | None = None
    temp_max_c: float | None = None
    temp_min_c: float | None = None
    feels_max_c: float | None = None
    humidity_pct: float | None = None
    visibility_m: float | None = None
    storm: bool = False
    fog: bool = False
    conditions: tuple[str, ...] = ()
    active_alerts: int | None = None
    #: Variables this question turns on that the response did not carry.
    missing: tuple[str, ...] = ()
    #: Points actually read, so an empty window is distinguishable from a dry one.
    samples: int = 0

    numbers: frozenset[str] = field(default_factory=frozenset)

    @property
    def has_any_data(self) -> bool:
        return self.samples > 0 and any(
            value is not None
            for value in (
                self.rain_peak_pct,
                self.rain_total_mm,
                self.wind_peak_ms,
                self.gust_peak_ms,
                self.temp_max_c,
                self.feels_max_c,
            )
        )


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 1)


def _first(*values: float | None) -> float | None:
    """The first value that was actually reported.

    Written out rather than using ``a or b`` because a reported 0.0 — no rain,
    no wind, freezing point — is falsy, and coalescing it away would turn "the
    source said zero" into "the source said nothing". Those are different
    answers, and only one of them is true.
    """
    for value in values:
        if value is not None:
            return value
    return None


def _field(point: object, *names: str) -> float | None:
    """One reading from a point, whichever name this point type gives it."""
    return _first(*(getattr(point, name, None) for name in names))


def _peak(values: Iterable[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return max(present) if present else None


def _trough(values: Iterable[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return min(present) if present else None


def _numbers_in(*values: object) -> frozenset[str]:
    """Every figure an answer built from this brief may legitimately contain."""
    tokens: set[str] = set()
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            tokens.add(f"{round(float(value), 1):g}")
            tokens.add(f"{int(round(float(value)))}")
        elif isinstance(value, datetime):
            tokens.update({f"{value.hour}", f"{value.hour:02d}", f"{value.day}"})
            hour12 = value.hour % 12 or 12
            tokens.add(f"{hour12}")
    return frozenset(tokens)


def _missing_for(domain: AdvisoryPurpose, present: set[str], marine_data: bool) -> tuple[str, ...]:
    absent = [name for name in _RELEVANT.get(domain, ()) if name not in present]
    if domain is AdvisoryPurpose.MARINE and not marine_data:
        for name in ("wave height", "sea state"):
            if name not in absent:
                absent.append(name)
    return tuple(absent)


def _from_points(points: Sequence[object]) -> dict[str, object]:
    """Reduce forecast points, hourly or daily, to the figures a decision uses."""
    probabilities = [
        _field(p, "precipitation_probability_pct", "precipitation_probability_max_pct")
        for p in points
    ]
    amounts = [_field(p, "precipitation_mm", "precipitation_sum_mm") for p in points]
    winds = [_field(p, "wind_speed_ms", "wind_speed_max_ms") for p in points]
    gusts = [_field(p, "wind_gust_ms", "wind_gust_max_ms") for p in points]
    highs = [_field(p, "temperature_c", "temperature_max_c") for p in points]
    lows = [_field(p, "temperature_c", "temperature_min_c") for p in points]

    wet = [
        p
        for p in points
        if (_field(p, "precipitation_mm", "precipitation_sum_mm") or 0.0) > 0.0
        or (
            _field(
                p,
                "precipitation_probability_pct",
                "precipitation_probability_max_pct",
            )
            or 0.0
        )
        >= 50.0
    ]
    first_wet = wet[0] if wet else None
    present = {
        name
        for name, series in (
            ("precipitation", probabilities + amounts),
            ("wind", winds + gusts),
            ("temperature", highs + lows),
        )
        if any(v is not None for v in series)
    }
    totals = [a for a in amounts if a is not None]
    return {
        "rain_peak_pct": _round(_peak(probabilities)),
        "rain_total_mm": _round(sum(totals)) if totals else None,
        "rain_first_at": _first_moment(first_wet),
        "wind_peak_ms": _round(_peak(winds)),
        "gust_peak_ms": _round(_peak(gusts)),
        "temp_max_c": _round(_peak(highs)),
        "temp_min_c": _round(_trough(lows)),
        "storm": any(getattr(p, "condition", None) in _STORMY for p in points),
        "fog": any(getattr(p, "condition", None) is WeatherCondition.FOG for p in points),
        "conditions": tuple(
            dict.fromkeys(
                d for d in (getattr(p, "condition_description", None) for p in points) if d
            )
        )[:3],
        "present": present,
        "marine": any(
            getattr(p, f, None) is not None for p in points for f in _MARINE_FIELDS
        ),
    }


def _first_moment(point: object):
    """When a point applies — hourly points carry an instant, daily ones a date."""
    if point is None:
        return None
    return _first_present(getattr(point, "valid_at", None), getattr(point, "date", None))


def _first_present(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _current_facts(current: CurrentWeather) -> dict[str, object]:
    facts = _from_points([current])
    facts["feels_max_c"] = _round(
        _first(current.apparent_temperature_c, current.temperature_c)
    )
    facts["humidity_pct"] = _round(current.relative_humidity_pct)
    facts["visibility_m"] = _round(current.visibility_m)
    if current.visibility_m is not None:
        facts["present"] = set(facts["present"]) | {"visibility"}  # type: ignore[operator]
    return facts


def build(
    *,
    question: str,
    intent: Intent,
    domain: AdvisoryPurpose,
    data: object,
    place: str,
    window_hours: int,
    active_alerts: int | None = None,
) -> Brief | None:
    """Reduce one backend result to the evidence a recommendation may use."""
    facts: dict[str, object]
    samples = 0
    past = False

    if isinstance(data, WeatherReport):
        facts = _current_facts(data.current)
        samples = 1
    elif isinstance(data, Forecast):
        if data.hourly:
            points: Sequence[object] = data.hourly[: max(1, window_hours)]
        else:
            points = data.daily[: max(1, (window_hours + 23) // 24)]
        if not points:
            return None
        facts = _from_points(points)
        samples = len(points)
    elif isinstance(data, list) and data:
        # Historical observations, each wrapping a CurrentWeather.
        observations = [record.weather for record in data if getattr(record, "weather", None)]
        if not observations:
            return None
        facts = _from_points(observations)
        # Accumulations over a past window sum; a forecast's do not mean the same.
        facts["feels_max_c"] = _round(
            _peak([o.apparent_temperature_c or o.temperature_c for o in observations])
        )
        samples = len(observations)
        past = True
    else:
        return None

    present = set(facts.pop("present"))  # type: ignore[arg-type]
    marine = bool(facts.pop("marine"))

    brief = Brief(
        question=question.strip()[:500],
        domain=domain,
        intent=intent,
        place=place,
        window_hours=window_hours,
        past=past,
        samples=samples,
        active_alerts=active_alerts,
        missing=_missing_for(domain, present, marine),
        **facts,  # type: ignore[arg-type]
    )
    return _with_numbers(brief)


def _with_numbers(brief: Brief) -> Brief:
    tokens = _numbers_in(
        brief.rain_peak_pct,
        brief.rain_total_mm,
        brief.wind_peak_ms,
        brief.gust_peak_ms,
        brief.temp_max_c,
        brief.temp_min_c,
        brief.feels_max_c,
        brief.humidity_pct,
        brief.visibility_m,
        brief.rain_first_at,
        brief.active_alerts,
        brief.window_hours,
        brief.samples,
    )
    return Brief(**{**brief.__dict__, "numbers": tokens})


_NUMERIC = re.compile(r"\d+(?:\.\d+)?")


def unsupported_numbers(text: str, brief: Brief) -> list[str]:
    """Figures in ``text`` that the brief does not account for.

    The check that makes a generated sentence safe to show: a recommendation may
    reason about the data, but every number it states must be one the backend
    returned. Anything else is invented, whatever it is about.
    """
    allowed = set(brief.numbers)
    # Small integers appear as ordinary prose ("the next 3 hours" restating the
    # window, "a couple") and as clock parts already covered above; a figure is
    # only interesting here if it is not in the brief at all.
    return [
        token
        for token in _NUMERIC.findall(text)
        if token not in allowed and f"{float(token):g}" not in allowed
    ]


def as_facts(brief: Brief) -> str:
    """The brief as plain lines, for a model that may only reword it.

    Every line is a figure the backend returned. Missing variables are listed
    explicitly, because a model that is not told what is absent will reach for
    the nearest available number instead and present it as an answer.
    """
    lines: list[str] = [
        f"- location: {brief.place or 'the requested location'}",
        f"- window: {'the past period asked about' if brief.past else f'the next {brief.window_hours} hours'}",
        f"- data points read: {brief.samples}",
    ]
    if brief.rain_peak_pct is not None:
        lines.append(f"- peak precipitation probability: {brief.rain_peak_pct:g}%")
    if brief.rain_total_mm is not None:
        lines.append(f"- total precipitation: {brief.rain_total_mm:g} mm")
    if brief.rain_first_at is not None:
        lines.append(f"- precipitation first appears at: {brief.rain_first_at.isoformat()}")
    if brief.wind_peak_ms is not None:
        lines.append(f"- peak wind speed: {brief.wind_peak_ms:g} m/s")
    if brief.gust_peak_ms is not None:
        lines.append(f"- peak wind gust: {brief.gust_peak_ms:g} m/s")
    if brief.temp_max_c is not None:
        lines.append(f"- maximum temperature: {brief.temp_max_c:g} °C")
    if brief.temp_min_c is not None:
        lines.append(f"- minimum temperature: {brief.temp_min_c:g} °C")
    if brief.feels_max_c is not None:
        lines.append(f"- peak apparent temperature: {brief.feels_max_c:g} °C")
    if brief.humidity_pct is not None:
        lines.append(f"- relative humidity: {brief.humidity_pct:g}%")
    if brief.storm:
        lines.append("- thunderstorm conditions appear in this window")
    if brief.fog:
        lines.append("- fog appears in this window")
    if brief.conditions:
        lines.append(f"- reported conditions: {', '.join(brief.conditions)}")
    if brief.active_alerts is not None:
        lines.append(f"- existing WeatherGPT alert records: {brief.active_alerts}")
    if brief.missing:
        lines.append(f"- MISSING (not returned, do not guess): {', '.join(brief.missing)}")
    return "\n".join(lines)
