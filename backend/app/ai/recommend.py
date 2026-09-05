"""The recommendation a person actually asked for.

The old verdict described the weather: "rain is possible, keep a cover handy".
That is a reading, and it is the same reading whether the question was about
spraying a field, sailing out of a harbour or walking to the shops. A person
asking "should I spray today?" wants an answer to *that*, reasoned from the
forecast — which is what this module composes.

Three rules shape every sentence it produces:

* **The question leads.** What the recommendation is *about* comes from the
  question's domain and the window it named. Rain is mentioned when rain is
  what the data says matters, not because rain is the default subject.
* **Only retrieved figures appear.** Every number comes from a
  :class:`~app.ai.brief.Brief`, and a variable the response did not carry is
  named as missing rather than filled in.
* **It recommends on weather, and stops there.** The weather part of "should I
  spray" is answerable from a forecast. Whether the product tolerates the rain
  that follows is a question about the product, and this module does not raise
  it unless the person did — a verdict that opens with rainfast periods has
  answered a question nobody asked.

Nothing here is a threshold in the safety sense. The deterministic alert engine
owns those and is untouched by this module; these bands only decide how a
sentence is phrased.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from app.ai.brief import Brief
from app.ai.models import AdvisoryPurpose, Intent

#: Phrasing bands. Not safety thresholds — see the module docstring.
RAIN_LIKELY_PCT = 60.0
RAIN_POSSIBLE_PCT = 30.0
WIND_STRONG_MS = 10.8   # Beaufort 6
WIND_MODERATE_MS = 5.5  # Beaufort 4
HOT_C = 35.0
WARM_C = 30.0
COLD_C = 12.0
HEAVY_RAIN_MM = 25.0
SOME_RAIN_MM = 2.5


@dataclass(frozen=True)
class Recommendation:
    text: str
    icon: str
    caveat: str | None = None


def _hindi(language: str) -> bool:
    return language.split("-")[0].casefold() == "hi"


def _when(moment, formatter: Callable | None) -> str | None:
    if moment is None:
        return None
    if formatter is not None:
        try:
            return formatter(moment)
        except Exception:  # noqa: BLE001 - a formatter must never break an answer
            return None
    return getattr(moment, "isoformat", lambda: str(moment))()


# ------------------------------------------------------------------ pieces
#
# Each returns a clause about one variable, or None when the data does not
# support one. The domain writers below choose which to use and in what order,
# which is what makes the sentence depend on the question.


def _rain_clause(brief: Brief, language: str, formatter) -> str | None:
    hi = _hindi(language)
    peak, total, when = brief.rain_peak_pct, brief.rain_total_mm, _when(brief.rain_first_at, formatter)

    if brief.past:
        if total is None:
            return None
        shown = f"{total:g}"
        if total >= HEAVY_RAIN_MM:
            return f"{shown} मिमी बारिश दर्ज हुई — भारी दौर" if hi else f"{shown} mm of rain was recorded — a heavy spell"
        if total >= SOME_RAIN_MM:
            return f"{shown} मिमी बारिश दर्ज हुई" if hi else f"{shown} mm of rain was recorded"
        if total > 0:
            return f"केवल {shown} मिमी नाममात्र बारिश" if hi else f"only {shown} mm fell, a trace"
        return "कोई बारिश दर्ज नहीं हुई" if hi else "no rain was recorded"

    if peak is None and total is None:
        return None
    if peak is not None and peak >= RAIN_LIKELY_PCT:
        base = f"बारिश की संभावना {peak:g}% तक" if hi else f"rain reaches {peak:g}% probability"
    elif peak is not None and peak >= RAIN_POSSIBLE_PCT:
        base = f"बारिश की कुछ संभावना ({peak:g}%)" if hi else f"rain is possible at {peak:g}%"
    elif peak is not None:
        return (
            f"बारिश की संभावना कम है ({peak:g}%)" if hi else f"rain stays unlikely at {peak:g}%"
        )
    else:
        base = "बारिश दर्ज है" if hi else "rain appears"

    if when:
        base += f", लगभग {when} से" if hi else f", first around {when}"
    if total:
        base += f" ({total:g} मिमी)" if hi else f" ({total:g} mm)"
    return base


def _wind_clause(brief: Brief, language: str) -> str | None:
    hi = _hindi(language)
    peak = brief.gust_peak_ms if brief.gust_peak_ms is not None else brief.wind_peak_ms
    if peak is None:
        return None
    shown = f"{peak:g}"
    if peak >= WIND_STRONG_MS:
        return f"हवा तेज़ है, झोंके {shown} m/s तक" if hi else f"wind is strong, gusting to {shown} m/s"
    if peak >= WIND_MODERATE_MS:
        return f"हवा मध्यम, झोंके {shown} m/s तक" if hi else f"wind is moderate at up to {shown} m/s"
    return f"हवा हल्की, {shown} m/s तक" if hi else f"wind stays light at up to {shown} m/s"


def _heat_clause(brief: Brief, language: str) -> str | None:
    hi = _hindi(language)
    value = brief.feels_max_c if brief.feels_max_c is not None else brief.temp_max_c
    if value is None:
        return None
    shown = f"{value:g}"
    if value >= HOT_C:
        return f"तापमान {shown} °C तक — काफ़ी गर्म" if hi else f"it reaches {shown} °C, which is hot"
    if value >= WARM_C:
        return f"तापमान {shown} °C तक — गर्म" if hi else f"it is warm at up to {shown} °C"
    if brief.temp_min_c is not None and brief.temp_min_c <= COLD_C:
        low = f"{brief.temp_min_c:g}"
        return f"तापमान {low} °C तक गिरता है — ठंडा" if hi else f"it drops to {low} °C, which is cold"
    return f"तापमान {shown} °C के आसपास — सहज" if hi else f"temperature sits around {shown} °C, comfortable"


def _missing_clause(brief: Brief, language: str) -> str | None:
    """What the answer could not weigh, said plainly rather than skipped."""
    if not brief.missing:
        return None
    hi = _hindi(language)
    names = ", ".join(brief.missing)
    if hi:
        return (
            f"इस पूर्वानुमान में {names} का डेटा नहीं है, इसलिए इतने भर से यह तय नहीं किया "
            f"जा सकता कि यह सुरक्षित है।"
        )
    return (
        f"This forecast carries no {names} data, so it cannot settle whether this is safe "
        f"on its own."
    )


# ----------------------------------------------------------------- writers


def _marine(brief: Brief, language: str, formatter) -> Recommendation:
    hi = _hindi(language)
    parts = [c for c in (_wind_clause(brief, language), _rain_clause(brief, language, formatter)) if c]
    icon = "💨" if (brief.gust_peak_ms or brief.wind_peak_ms or 0) >= WIND_STRONG_MS else "🌊"

    if brief.storm:
        lead = (
            "मैं इस अवधि में समुद्र में जाने से बचूँगा — गरज के साथ तूफ़ानी स्थिति दिख रही है"
            if hi
            else "I would not head out in this window — thunderstorms show up in the forecast"
        )
        icon = "⛈️"
    elif (brief.gust_peak_ms or brief.wind_peak_ms or 0) >= WIND_STRONG_MS:
        lead = (
            "मैं इस अवधि में छोटी नाव लेकर जाने से बचूँगा"
            if hi
            else "I would hold off on a small boat over this window"
        )
    elif parts:
        lead = (
            "मौसम के लिहाज़ से हालात ठीक दिखते हैं"
            if hi
            else "On the weather alone, conditions look workable"
        )
    else:
        lead = (
            "इस अवधि के लिए पर्याप्त मौसम डेटा नहीं मिला"
            if hi
            else "The forecast returned too little for this window to judge on"
        )
    body = f"{lead}: {'; '.join(parts)}." if parts else f"{lead}."
    return Recommendation(body, icon, _missing_clause(brief, language))


def _agriculture(brief: Brief, language: str, formatter) -> Recommendation:
    """Answers the weather half of a field-work question, and only that half."""
    hi = _hindi(language)
    rain = _rain_clause(brief, language, formatter)
    wind = _wind_clause(brief, language)
    peak = brief.rain_peak_pct or 0.0
    gust = brief.gust_peak_ms if brief.gust_peak_ms is not None else (brief.wind_peak_ms or 0.0)
    when = _when(brief.rain_first_at, formatter)

    if brief.storm or peak >= RAIN_LIKELY_PCT:
        lead = (
            "मैं आज छिड़काव टालूँगा"
            if hi
            else "I would hold off on spraying in this window"
        )
        if when:
            tail = (
                f"सुरक्षित समय {when} से पहले का सूखा अंतराल है"
                if hi
                else f"the drier stretch before {when} is the better window"
            )
        else:
            tail = "बाद में हालात बेहतर हो सकते हैं" if hi else "conditions look better later"
        icon = "🌧️"
    elif peak >= RAIN_POSSIBLE_PCT:
        lead = (
            "छिड़काव संभव है, पर मौसम पर नज़र रखें"
            if hi
            else "Spraying looks possible, but keep an eye on the sky"
        )
        tail = (
            "जल्दी करना बेहतर है" if hi else "earlier in the window is the safer bet"
        )
        icon = "🌦️"
    elif gust >= WIND_STRONG_MS:
        lead = (
            "बारिश की चिंता नहीं, पर हवा तेज़ है — बहाव का ख़तरा"
            if hi
            else "Rain is not the problem here, but the wind is — spray drift is the risk"
        )
        tail = "हवा शांत होने पर करें" if hi else "wait for a calmer stretch"
        icon = "💨"
    else:
        lead = (
            "मौसम के लिहाज़ से छिड़काव के लिए हालात ठीक दिखते हैं"
            if hi
            else "On the weather, this window looks suitable for spraying"
        )
        tail = ""
        icon = "🌤️"

    facts = "; ".join(c for c in (rain, wind) if c)
    pieces = [p for p in (facts, tail) if p]
    return Recommendation(
        f"{lead}. {'. '.join(pieces)}." if pieces else f"{lead}.",
        icon,
    )


def _travel(brief: Brief, language: str, formatter) -> Recommendation:
    hi = _hindi(language)
    parts = [
        c
        for c in (
            _rain_clause(brief, language, formatter),
            _wind_clause(brief, language),
        )
        if c
    ]
    if brief.storm:
        lead = "इस अवधि में यात्रा टालना बेहतर" if hi else "I would rather not travel in this window"
        icon = "⛈️"
    elif brief.fog:
        lead = "दृश्यता कम है — अतिरिक्त समय रखें" if hi else "Visibility is the problem here — allow extra time"
        icon = "🌫️"
    elif (brief.rain_peak_pct or 0) >= RAIN_LIKELY_PCT:
        lead = (
            "यात्रा हो सकती है, पर गीली सड़कों की तैयारी रखें"
            if hi
            else "Travel looks fine, but plan for wet roads"
        )
        icon = "🌧️"
    else:
        lead = "यात्रा के लिए हालात ठीक दिखते हैं" if hi else "This looks like a good window to travel"
        icon = "🚗"
    return Recommendation(f"{lead}: {'; '.join(parts)}." if parts else f"{lead}.", icon)


def _outdoor_event(brief: Brief, language: str, formatter) -> Recommendation:
    hi = _hindi(language)
    rain = _rain_clause(brief, language, formatter)
    wind = _wind_clause(brief, language)
    heat = _heat_clause(brief, language)
    peak = brief.rain_peak_pct or 0.0
    gust = brief.gust_peak_ms if brief.gust_peak_ms is not None else (brief.wind_peak_ms or 0.0)
    feels = brief.feels_max_c if brief.feels_max_c is not None else brief.temp_max_c

    if brief.storm or peak >= RAIN_LIKELY_PCT:
        lead = "इस समय कार्यक्रम टालना बेहतर है" if hi else "I would postpone the outdoor event"
        icon = "⛈️" if brief.storm else "🌧️"
    elif gust >= WIND_STRONG_MS:
        lead = "तेज़ हवा के कारण कार्यक्रम के लिए मौसम अनुकूल नहीं है" if hi else "The weather is not suitable for the event because of strong wind"
        icon = "💨"
    elif feels is not None and feels >= HOT_C:
        lead = "गर्मी के कारण कार्यक्रम में सावधानी रखें" if hi else "The event is possible, but plan for the heat"
        icon = "🥵"
    else:
        lead = "कार्यक्रम के लिए मौसम अनुकूल दिखता है" if hi else "The weather looks suitable for the outdoor event"
        icon = "🌤️"

    facts = "; ".join(clause for clause in (rain, wind, heat) if clause)
    return Recommendation(f"{lead}: {facts}." if facts else f"{lead}.", icon, _missing_clause(brief, language))


def _general(brief: Brief, language: str, formatter) -> Recommendation:
    """Leads with whatever the data says is the notable thing, not with rain."""
    hi = _hindi(language)
    rain = _rain_clause(brief, language, formatter)
    wind = _wind_clause(brief, language)
    heat = _heat_clause(brief, language)
    peak = brief.rain_peak_pct or 0.0
    gust = brief.gust_peak_ms if brief.gust_peak_ms is not None else (brief.wind_peak_ms or 0.0)
    feels = brief.feels_max_c if brief.feels_max_c is not None else brief.temp_max_c

    if brief.storm:
        return Recommendation(
            ("गरज के साथ तूफ़ानी स्थिति दर्ज है — बाहर का काम टालें।" if hi
             else "Thunderstorms show up in this window — worth putting outdoor plans off."),
            "⛈️",
        )
    if brief.past:
        parts = [c for c in (rain, heat, wind) if c]
        return Recommendation(
            ("रिकॉर्ड के अनुसार: " if hi else "Over that period: ") + "; ".join(parts) + ".",
            "📊",
        ) if parts else Recommendation(
            ("उस अवधि के लिए पर्याप्त रिकॉर्ड नहीं मिला।" if hi
             else "Too little was recorded for that period to read."), "📊"
        )

    # The lead is whichever variable is actually doing something.
    if peak >= RAIN_LIKELY_PCT:
        lead, icon = (rain or ""), "🌧️"
    elif gust >= WIND_STRONG_MS:
        lead, icon = (wind or ""), "💨"
    elif feels is not None and (feels >= HOT_C or (brief.temp_min_c or 99) <= COLD_C):
        lead, icon = (heat or ""), ("🥵" if feels >= HOT_C else "🧥")
    elif peak >= RAIN_POSSIBLE_PCT:
        lead, icon = (rain or ""), "🌦️"
    else:
        lead, icon = (heat or rain or wind or ""), "🌤️"

    supporting = [c for c in (rain, heat, wind) if c and c != lead][:1]
    if not lead:
        return Recommendation(
            ("इस अवधि के लिए पर्याप्त डेटा नहीं मिला।" if hi
             else "Too little was returned for this window to read."), "🌤️"
        )
    sentence = lead[0].upper() + lead[1:]
    if supporting:
        sentence += ("; " if hi else "; ") + supporting[0]
    return Recommendation(sentence + ".", icon)


_WRITERS = {
    AdvisoryPurpose.MARINE: _marine,
    AdvisoryPurpose.AGRICULTURE: _agriculture,
    AdvisoryPurpose.TRAVEL: _travel,
    AdvisoryPurpose.OUTDOOR_EVENT: _outdoor_event,
    AdvisoryPurpose.GENERAL: _general,
}


def compose(brief: Brief, *, language: str = "en", format_time=None) -> Recommendation | None:
    """The deterministic recommendation for this question and this data."""
    if not brief.has_any_data:
        return None
    if brief.past:
        # A past window is read, never recommended on, whatever the domain.
        return _general(brief, language, format_time)
    writer = _WRITERS.get(brief.domain, _general)
    return writer(brief, language, format_time)
