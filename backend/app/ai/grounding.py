"""Deterministic rendering of validated tool results.

No model-generated prose reaches a user.  Every weather value in an answer is
read from a typed response returned by the WeatherGPT backend in this turn.

Translation works the same way: a catalog supplies the *labels*, and the
numbers are copied from the backend response untouched.  A model never sees a
weather value, so it cannot alter one on the way through.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

from app.ai.models import ChatResponse, Intent, LocationRiskResult, SourceReference, ToolResult
from app.api.v1.alerts import AlertListResponse
from app.api.v1.historical import HistoricalWeatherResponse
from app.api.v1.locations import LocationSearchResponse
from app.domain.forecast import Forecast
from app.domain.weather import WeatherReport


@dataclass(frozen=True)
class RenderedAnswer:
    answer: str
    safety_note: str | None = None


#: Every label the renderer can emit. A catalog overrides what it has and
#: inherits the rest, so a partial translation degrades to English per string
#: rather than failing or blocking the answer.
_ENGLISH: dict[str, str] = {
    "unavailable": (
        "Weather backend data is temporarily unavailable, so I cannot provide "
        "weather values right now."
    ),
    "need_location": (
        "Which location should I use? Please provide a place name or latitude "
        "and longitude."
    ),
    "need_history_dates": (
        "Please provide the historical date or inclusive date range to look up."
    ),
    "unknown": (
        "I can help with current weather, hourly or daily forecasts, stored "
        "historical observations, existing WeatherGPT alerts, and travel or "
        "agriculture planning."
    ),
    "need_search_place": "Which place should I search for?",
    "no_data": "No backend weather data was returned.",
    "unrenderable": "The requested backend result cannot be rendered safely.",
    "the_location": "the requested location",
    "no_values": "no reported values",
    # Current conditions
    "current_for": "Current conditions for {place}:",
    "condition": "Condition",
    "temperature": "Temperature",
    "feels_like": "Feels-like temperature",
    "humidity": "Relative humidity",
    "wind_speed": "Wind speed",
    "wind_gust": "Wind gust",
    "precipitation_interval": "Precipitation in the reporting interval",
    "visibility": "Visibility",
    "observed_at": "Observed at",
    # Forecast
    "hourly_for": "Hourly forecast for {place} (returned points):",
    "no_hourly": "No hourly forecast points were returned for {place}.",
    "hourly_truncated": (
        "The backend returned {total} hourly points; this answer shows the first 24."
    ),
    "daily_for": "Daily forecast for {place}:",
    "no_daily": "No daily forecast points were returned for {place}.",
    "min": "min",
    "max": "max",
    "precipitation": "precipitation",
    "max_precip_probability": "max precipitation probability",
    "max_wind": "max wind",
    "temperature_short": "temperature",
    "precipitation_probability": "precipitation probability",
    "wind": "wind",
    "gust": "gust",
    # Historical
    "historical_for": (
        "Weather observations for {place}, {start} through {end}: {count} returned."
    ),
    "historical_truncated": "The backend marked this result as truncated.",
    "historical_showing": "This answer shows the first 20 of {count} observations.",
    # Alerts and risk
    "alerts_count": "Existing WeatherGPT alert records returned: {count}.",
    "risk_for": (
        "{purpose} planning considerations for {place}, based only on retrieved "
        "backend data:"
    ),
    "risk_none": "The retrieved fields did not contain a specific planning consideration.",
    "risk_active": "Existing active WeatherGPT alert records returned: {count}.",
    # Said only when the alert store could not be read. It reports the gap
    # rather than filling it: an unread store is not an all-clear.
    "risk_alerts_unavailable": (
        "The WeatherGPT alert store could not be read for this request, so no "
        "statement is made about existing alerts. The weather values above were "
        "retrieved normally."
    ),
    "purpose_general": "General",
    "purpose_agriculture": "Agriculture",
    "purpose_travel": "Travel",
    "purpose_marine": "Marine",
    # Locations
    "candidates_for": "Location candidates for '{query}':",
    "no_candidates": "No location candidates were returned for '{query}'.",
    # Provenance
    "sources": "Sources and freshness:",
    "fetched": "fetched {at}",
    "cached": "served from backend cache",
    "not_cached": "not served from backend cache",
    "model": "model {name}",
    "model_run": "model run {at}",
    "licence": "licence {name}",
}

#: Hindi labels. Numerals and units stay Western/SI: they are copied from the
#: backend response, and rewriting them would be editing data, not translating.
_HINDI: dict[str, str] = {
    "unavailable": (
        "मौसम बैकएंड डेटा अभी उपलब्ध नहीं है, इसलिए मैं अभी मौसम के आंकड़े नहीं दे सकता।"
    ),
    "need_location": "कृपया स्थान बताएं — जगह का नाम या अक्षांश और देशांतर।",
    "need_history_dates": "कृपया पिछली तारीख या तारीखों की सीमा बताएं।",
    "unknown": (
        "मैं वर्तमान मौसम, प्रति घंटा या दैनिक पूर्वानुमान, संग्रहीत ऐतिहासिक "
        "प्रेक्षण, मौजूदा WeatherGPT अलर्ट, और यात्रा या कृषि नियोजन में मदद कर सकता हूँ।"
    ),
    "need_search_place": "मैं किस जगह को खोजूँ?",
    "no_data": "बैकएंड से कोई मौसम डेटा नहीं मिला।",
    "unrenderable": "अनुरोधित बैकएंड परिणाम को सुरक्षित रूप से प्रस्तुत नहीं किया जा सकता।",
    "the_location": "अनुरोधित स्थान",
    "no_values": "कोई मान उपलब्ध नहीं",
    "current_for": "{place} के लिए वर्तमान मौसम:",
    "condition": "स्थिति",
    "temperature": "तापमान",
    "feels_like": "अनुभव किया गया तापमान",
    "humidity": "सापेक्ष आर्द्रता",
    "wind_speed": "हवा की गति",
    "wind_gust": "हवा का झोंका",
    "precipitation_interval": "रिपोर्टिंग अवधि में वर्षा",
    "visibility": "दृश्यता",
    "observed_at": "प्रेक्षण समय",
    "hourly_for": "{place} के लिए प्रति घंटा पूर्वानुमान (प्राप्त बिंदु):",
    "no_hourly": "{place} के लिए कोई प्रति घंटा पूर्वानुमान बिंदु नहीं मिला।",
    "hourly_truncated": (
        "बैकएंड ने {total} प्रति घंटा बिंदु लौटाए; यह उत्तर पहले 24 दिखाता है।"
    ),
    "daily_for": "{place} के लिए दैनिक पूर्वानुमान:",
    "no_daily": "{place} के लिए कोई दैनिक पूर्वानुमान बिंदु नहीं मिला।",
    "min": "न्यूनतम",
    "max": "अधिकतम",
    "precipitation": "वर्षा",
    "max_precip_probability": "अधिकतम वर्षा संभावना",
    "max_wind": "अधिकतम हवा",
    "temperature_short": "तापमान",
    "precipitation_probability": "वर्षा संभावना",
    "wind": "हवा",
    "gust": "झोंका",
    "historical_for": "{place} के लिए {start} से {end} तक मौसम प्रेक्षण: {count} प्राप्त।",
    "historical_truncated": "बैकएंड ने इस परिणाम को अपूर्ण चिह्नित किया।",
    "historical_showing": "यह उत्तर {count} प्रेक्षणों में से पहले 20 दिखाता है।",
    "alerts_count": "मौजूदा WeatherGPT अलर्ट रिकॉर्ड: {count}।",
    "risk_for": (
        "{place} के लिए {purpose} नियोजन हेतु विचार, केवल प्राप्त बैकएंड डेटा पर आधारित:"
    ),
    "risk_none": "प्राप्त आंकड़ों में कोई विशेष नियोजन बिंदु नहीं मिला।",
    "risk_active": "मौजूदा सक्रिय WeatherGPT अलर्ट रिकॉर्ड: {count}।",
    "risk_alerts_unavailable": (
        "इस अनुरोध के लिए WeatherGPT अलर्ट रिकॉर्ड नहीं पढ़े जा सके, इसलिए मौजूदा अलर्ट के "
        "बारे में कोई कथन नहीं दिया गया है। ऊपर दिए गए मौसम मान सामान्य रूप से प्राप्त हुए हैं।"
    ),
    "purpose_general": "सामान्य",
    "purpose_agriculture": "कृषि",
    "purpose_travel": "यात्रा",
    "purpose_marine": "समुद्री",
    "candidates_for": "'{query}' के लिए संभावित स्थान:",
    "no_candidates": "'{query}' के लिए कोई स्थान नहीं मिला।",
    "sources": "स्रोत और नवीनता:",
    "fetched": "प्राप्त {at}",
    "cached": "बैकएंड कैश से",
    "not_cached": "बैकएंड कैश से नहीं",
    "model": "मॉडल {name}",
    "model_run": "मॉडल रन {at}",
    "licence": "लाइसेंस {name}",
}


class MessageCatalog:
    """Labels for one language, falling back to English string by string."""

    language: str = "en"
    labels: dict[str, str] = _ENGLISH

    def label(self, key: str, **values: object) -> str:
        template = self.labels.get(key) or _ENGLISH[key]
        return template.format(**values) if values else template

    # Named accessors keep call sites readable for the common clarifications.
    def unavailable(self) -> str:
        return self.label("unavailable")

    def need_location(self) -> str:
        return self.label("need_location")

    def need_history_dates(self) -> str:
        return self.label("need_history_dates")

    def unknown(self) -> str:
        return self.label("unknown")


class HindiCatalog(MessageCatalog):
    language = "hi"
    labels = _HINDI


class CatalogRegistry:
    """Locale selection with explicit English fallback until a catalog is installed."""

    def __init__(self) -> None:
        self._catalogs: dict[str, MessageCatalog] = {
            "en": MessageCatalog(),
            "hi": HindiCatalog(),
        }

    def resolve(self, language: str) -> tuple[MessageCatalog, bool]:
        normalized = language.casefold().replace("_", "-")
        catalog = self._catalogs.get(normalized) or self._catalogs.get(normalized.split("-", 1)[0])
        return (catalog, False) if catalog else (self._catalogs["en"], True)


class GroundedRenderer:
    """Formats only exact backend facts and bounded, data-linked considerations."""

    def render(
        self, intent: Intent, results: list[ToolResult], catalog: MessageCatalog | None = None
    ) -> RenderedAnswer:
        catalog = catalog or MessageCatalog()
        if not results:
            return RenderedAnswer(catalog.label("no_data"))
        result = results[0]
        renderers: dict[str, Callable[[ToolResult, MessageCatalog], RenderedAnswer]] = {
            "current_weather": self._current,
            "hourly_forecast": self._hourly,
            "daily_forecast": self._daily,
            "historical_weather": self._historical,
            "alerts": self._alerts,
            "location_risk": self._risk,
            "location_search": self._locations,
        }
        renderer = renderers.get(result.tool)
        if renderer is None:
            return RenderedAnswer(catalog.label("unrenderable"))
        return renderer(result, catalog)

    def _current(self, result: ToolResult, catalog: MessageCatalog) -> RenderedAnswer:
        report = WeatherReport.model_validate(result.data)
        current = report.current
        lines = [catalog.label("current_for", place=_place(report.location.display_name, catalog))]
        if current.condition_description:
            lines.append(f"- {catalog.label('condition')}: {current.condition_description}.")
        elif current.condition.value != "unknown":
            lines.append(
                f"- {catalog.label('condition')}: {current.condition.value.replace('_', ' ')}."
            )
        _append(lines, catalog.label("temperature"), current.temperature_c, "°C")
        _append(lines, catalog.label("feels_like"), current.apparent_temperature_c, "°C")
        _append(lines, catalog.label("humidity"), current.relative_humidity_pct, "%")
        _append(lines, catalog.label("wind_speed"), current.wind_speed_ms, "m/s")
        _append(lines, catalog.label("wind_gust"), current.wind_gust_ms, "m/s")
        _append(
            lines, catalog.label("precipitation_interval"), current.precipitation_mm, "mm"
        )
        _append(lines, catalog.label("visibility"), current.visibility_m, "m")
        lines.append(f"- {catalog.label('observed_at')}: {current.observed_at.isoformat()}.")
        return RenderedAnswer("\n".join(lines))

    def _hourly(self, result: ToolResult, catalog: MessageCatalog) -> RenderedAnswer:
        forecast = Forecast.model_validate(result.data)
        place = _place(forecast.location.display_name, catalog)
        if not forecast.hourly:
            return RenderedAnswer(catalog.label("no_hourly", place=place))
        lines = [catalog.label("hourly_for", place=place)]
        for point in forecast.hourly[:24]:
            facts = _hour_facts(point, catalog)
            lines.append(
                f"- {point.valid_at.isoformat()}: "
                f"{', '.join(facts) if facts else catalog.label('no_values')}."
            )
        if len(forecast.hourly) > 24:
            lines.append(f"- {catalog.label('hourly_truncated', total=len(forecast.hourly))}")
        return RenderedAnswer("\n".join(lines))

    def _daily(self, result: ToolResult, catalog: MessageCatalog) -> RenderedAnswer:
        forecast = Forecast.model_validate(result.data)
        place = _place(forecast.location.display_name, catalog)
        if not forecast.daily:
            return RenderedAnswer(catalog.label("no_daily", place=place))
        lines = [catalog.label("daily_for", place=place)]
        for point in forecast.daily:
            facts: list[str] = []
            if point.condition_description:
                facts.append(point.condition_description)
            _fact(facts, catalog.label("min"), point.temperature_min_c, "°C")
            _fact(facts, catalog.label("max"), point.temperature_max_c, "°C")
            _fact(facts, catalog.label("precipitation"), point.precipitation_sum_mm, "mm")
            _fact(
                facts,
                catalog.label("max_precip_probability"),
                point.precipitation_probability_max_pct,
                "%",
            )
            _fact(facts, catalog.label("max_wind"), point.wind_speed_max_ms, "m/s")
            lines.append(
                f"- {point.date.isoformat()}: "
                f"{', '.join(facts) if facts else catalog.label('no_values')}."
            )
        return RenderedAnswer("\n".join(lines))

    def _historical(self, result: ToolResult, catalog: MessageCatalog) -> RenderedAnswer:
        history = HistoricalWeatherResponse.model_validate(result.data)
        where = (
            history.location.display_name
            if history.location is not None
            else f"{history.requested['latitude']:g}, {history.requested['longitude']:g}"
        )
        lines = [
            catalog.label(
                "historical_for",
                place=where,
                latitude=f"{history.requested['latitude']:g}",
                longitude=f"{history.requested['longitude']:g}",
                start=_window_bound(history.range.start),
                end=_window_bound(history.range.end),
                count=history.count,
            )
        ]
        if history.truncated:
            lines.append(f"- {catalog.label('historical_truncated')}")
        for observation in history.observations[:20]:
            facts: list[str] = []
            _fact(
                facts, catalog.label("temperature_short"), observation.weather.temperature_c, "°C"
            )
            if observation.weather.condition_description:
                facts.append(observation.weather.condition_description)
            _fact(
                facts, catalog.label("precipitation"), observation.weather.precipitation_mm, "mm"
            )
            _fact(facts, catalog.label("wind"), observation.weather.wind_speed_ms, "m/s")
            _fact(facts, catalog.label("gust"), observation.weather.wind_gust_ms, "m/s")
            lines.append(
                f"- {observation.weather.observed_at.isoformat()} at "
                f"{observation.distance_km:g} km: "
                f"{', '.join(facts) if facts else catalog.label('no_values')}."
            )
        if len(history.observations) > 20:
            lines.append(f"- {catalog.label('historical_showing', count=history.count)}")
        return RenderedAnswer("\n".join(lines))

    def _alerts(self, result: ToolResult, catalog: MessageCatalog) -> RenderedAnswer:
        response = AlertListResponse.model_validate(result.data)
        if not response.alerts:
            return RenderedAnswer("No active alerts were found for this location.")

        groups = defaultdict(list)
        for alert in response.alerts:
            groups[(alert.title, alert.severity.value)].append(alert)

        lines = [f"I found {response.count} active weather alerts."]
        for (title, severity), alerts in groups.items():
            start = min(alert.valid_from for alert in alerts)
            end = max(alert.valid_until for alert in alerts)
            values = [alert.evidence.observed_value for alert in alerts]
            thresholds = {alert.evidence.threshold for alert in alerts}
            summary = f"- {title}: {len(alerts)} {severity} alert(s), from "
            summary += (
                f"{start.strftime('%d-%m-%Y %I:%M %p')} to "
                f"{end.strftime('%d-%m-%Y %I:%M %p')}"
            )
            if len(thresholds) == 1:
                unit = alerts[0].evidence.unit
                summary += f"; {unit} ranged from {min(values):g} to {max(values):g}"
                summary += f" (threshold {next(iter(thresholds)):g} {unit})"
            lines.append(summary + ".")

        return RenderedAnswer("\n".join(lines))

    def _risk(self, result: ToolResult, catalog: MessageCatalog) -> RenderedAnswer:
        risk = LocationRiskResult.model_validate(result.data)
        place = _place(risk.location.display_name, catalog)
        lines = [f"{place}: here's what the available weather data shows."]
        if risk.considerations:
            for item in risk.considerations:
                lines.append(item.statement.rstrip("."))
        else:
            lines.append(catalog.label("risk_none"))
        if risk.active_alert_count:
            lines.append(
                f"I also found {risk.active_alert_count} active WeatherGPT alert "
                "record(s) for this location."
            )
        return RenderedAnswer(" ".join(lines))

    def _locations(self, result: ToolResult, catalog: MessageCatalog) -> RenderedAnswer:
        locations = LocationSearchResponse.model_validate(result.data)
        if not locations.results:
            return RenderedAnswer(catalog.label("no_candidates", query=locations.query))
        lines = [catalog.label("candidates_for", query=locations.query)]
        for location in locations.results:
            coordinates = location.coordinates
            lines.append(
                f"- {_place(location.display_name, catalog)} "
                f"({coordinates.latitude:g}, {coordinates.longitude:g})"
            )
        return RenderedAnswer("\n".join(lines))


def _window_bound(moment) -> str:
    """A window boundary, stated at the precision it was asked at.

    A question about yesterday produced a boundary of 23:59:59.999999. Reciting
    that back is precise about the wrong thing; the day is what was asked for.
    """
    whole_day = (moment.hour, moment.minute) in {(0, 0), (23, 59)}
    return moment.date().isoformat() if whole_day else moment.isoformat()


def _place(value: str, catalog: MessageCatalog) -> str:
    return value or catalog.label("the_location")


def _append(lines: list[str], label: str, value: float | None, unit: str) -> None:
    if value is not None:
        lines.append(f"- {label}: {value:g} {unit}.")


def _fact(target: list[str], label: str, value: float | None, unit: str) -> None:
    if value is not None:
        target.append(f"{label} {value:g} {unit}")


def _hour_facts(point, catalog: MessageCatalog) -> list[str]:
    facts: list[str] = []
    if point.condition_description:
        facts.append(point.condition_description)
    _fact(facts, catalog.label("temperature_short"), point.temperature_c, "°C")
    _fact(facts, catalog.label("precipitation"), point.precipitation_mm, "mm")
    _fact(
        facts,
        catalog.label("precipitation_probability"),
        point.precipitation_probability_pct,
        "%",
    )
    _fact(facts, catalog.label("wind"), point.wind_speed_ms, "m/s")
    _fact(facts, catalog.label("gust"), point.wind_gust_ms, "m/s")
    return facts
