"""Read-only AI tools backed exclusively by WeatherGPT's FastAPI API."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel

from app.ai.backend import BackendAPIClient
from app.ai.models import (
    AdvisoryPurpose,
    AlertsArguments,
    CurrentWeatherArguments,
    ForecastArguments,
    HistoricalWeatherArguments,
    LocationInput,
    LocationRiskArguments,
    LocationRiskResult,
    LocationSearchArguments,
    RiskConsideration,
    SourceReference,
    ToolResult,
)
from app.config.logging import get_logger
from app.domain.forecast import Forecast
from app.domain.location import Coordinates, Location
from app.domain.provenance import DataProvenance
from app.domain.weather import WeatherReport

logger = get_logger(__name__)

ArgumentsModel = TypeVar("ArgumentsModel", bound=BaseModel)
#: A backend result that carries the location it applies to.
ModelWithLocation = TypeVar("ModelWithLocation", WeatherReport, Forecast)


class ToolInputError(ValueError):
    """The user needs to clarify a tool argument before a backend request."""


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    arguments_model: type[BaseModel]

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.arguments_model.model_json_schema(),
            },
        }


TOOL_DEFINITIONS: dict[str, ToolDefinition] = {
    "current_weather": ToolDefinition(
        "current_weather", "Get validated current conditions for one location.", CurrentWeatherArguments
    ),
    "hourly_forecast": ToolDefinition(
        "hourly_forecast", "Get validated hourly forecast points for one location.", ForecastArguments
    ),
    "daily_forecast": ToolDefinition(
        "daily_forecast", "Get validated daily forecast points for one location.", ForecastArguments
    ),
    "historical_weather": ToolDefinition(
        "historical_weather",
        "Get weather observations previously stored by WeatherGPT for a date range.",
        HistoricalWeatherArguments,
    ),
    "alerts": ToolDefinition(
        "alerts",
        "Get existing WeatherGPT deterministic rule alerts. This cannot create or modify alerts.",
        AlertsArguments,
    ),
    "location_risk": ToolDefinition(
        "location_risk",
        "Read current weather, forecast, and existing alerts to provide a bounded travel or agriculture consideration.",
        LocationRiskArguments,
    ),
    "location_search": ToolDefinition(
        "location_search", "Search for location candidates without choosing one.", LocationSearchArguments
    ),
}


class BackendWeatherTools:
    """Tool registry that delegates every factual lookup to the backend API."""

    def __init__(self, backend: BackendAPIClient) -> None:
        self._backend = backend

    @property
    def definitions(self) -> list[dict[str, Any]]:
        return [definition.openai_schema() for definition in TOOL_DEFINITIONS.values()]

    async def aclose(self) -> None:
        await self._backend.aclose()

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        near: Coordinates | None = None,
    ) -> ToolResult:
        """Run one read-only tool.

        ``near`` is the place the conversation is already about. It only
        reorders gazetteer candidates, which is what makes "what about
        Miyapur?" after a question about Hyderabad find the right Miyapur. It
        never reaches a weather call and never changes a value.
        """
        definition = TOOL_DEFINITIONS.get(name)
        if definition is None:
            raise ToolInputError(f"Unsupported read-only tool: {name}.")
        parsed = definition.arguments_model.model_validate(arguments)
        method = getattr(self, f"_{name}")
        return await method(parsed, near)

    async def _current_weather(
        self, args: CurrentWeatherArguments, near: Coordinates | None = None
    ) -> ToolResult:
        place = await self._resolve(args, near)
        report = await self._backend.current(
            latitude=place.coordinates.latitude,
            longitude=place.coordinates.longitude,
            provider=args.provider,
        )
        return self._result("current_weather", _named(report, place), report.provenance)

    async def _hourly_forecast(
        self, args: ForecastArguments, near: Coordinates | None = None
    ) -> ToolResult:
        place = await self._resolve(args, near)
        forecast = await self._backend.forecast(
            latitude=place.coordinates.latitude,
            longitude=place.coordinates.longitude,
            days=args.days,
            hourly=True,
            daily=False,
            provider=args.provider,
        )
        return self._result("hourly_forecast", _named(forecast, place), forecast.provenance)

    async def _daily_forecast(
        self, args: ForecastArguments, near: Coordinates | None = None
    ) -> ToolResult:
        place = await self._resolve(args, near)
        forecast = await self._backend.forecast(
            latitude=place.coordinates.latitude,
            longitude=place.coordinates.longitude,
            days=args.days,
            hourly=False,
            daily=True,
            provider=args.provider,
        )
        return self._result("daily_forecast", _named(forecast, place), forecast.provenance)

    async def _historical_weather(
        self, args: HistoricalWeatherArguments, near: Coordinates | None = None
    ) -> ToolResult:
        place = await self._resolve(args, near)
        coordinates = place.coordinates
        # An hour-precision question sends instants; a calendar one sends dates
        # and is resolved against the location's own day downstream.
        if args.start_time is not None and args.end_time is not None:
            start, end = args.start_time.isoformat(), args.end_time.isoformat()
        else:
            assert args.start is not None and args.end is not None
            start, end = args.start.isoformat(), args.end.isoformat()
        history = await self._backend.historical(
            latitude=coordinates.latitude,
            longitude=coordinates.longitude,
            start=start,
            end=end,
            radius_km=args.radius_km,
            hour_from=args.hour_from,
            hour_to=args.hour_to,
            provider=args.provider,
        )
        sources = _unique_sources(
            "historical_weather", [observation.provenance for observation in history.observations]
        )
        # The place was resolved before the lookup; naming it is what turns
        # "observations near 17.3606, 78.4741" into a sentence about Hyderabad.
        named = history.model_copy(update={"location": place if place.name else None})
        return ToolResult(
            tool="historical_weather", data=named.model_dump(mode="json"), sources=sources
        )

    async def _alerts(
        self, args: AlertsArguments, near: Coordinates | None = None
    ) -> ToolResult:
        coordinates = (await self._resolve(args, near)).coordinates
        alerts = await self._backend.alerts(
            latitude=coordinates.latitude,
            longitude=coordinates.longitude,
            radius_km=args.radius_km,
            provider=args.provider,
        )
        sources = _unique_sources("alerts", [alert.provenance for alert in alerts.alerts])
        return ToolResult(tool="alerts", data=alerts.model_dump(mode="json"), sources=sources)

    async def _location_search(
        self, args: LocationSearchArguments, near: Coordinates | None = None
    ) -> ToolResult:
        locations = await self._backend.locations(
            query=args.query, limit=args.limit, near=near
        )
        return ToolResult(tool="location_search", data=locations.model_dump(mode="json"))

    async def _location_risk(
        self, args: LocationRiskArguments, near: Coordinates | None = None
    ) -> ToolResult:
        """Compose backend facts into considerations; this never evaluates alert rules."""
        place = await self._resolve(args, near)
        coordinates = place.coordinates
        location_params = {"latitude": coordinates.latitude, "longitude": coordinates.longitude}
        # The alert store is the only leg of this that can be absent while the
        # question still has an answer: measurements and a forecast are what a
        # planning consideration is built from, and the alert record is
        # additional evidence beside them. So its failure is captured rather
        # than raised, and reported as "not available" — never as "none found",
        # which is a claim about the weather that nothing here checked.
        current, forecast, alerts = await asyncio.gather(
            self._backend.current(**location_params, provider=args.provider),
            self._backend.forecast(
                **location_params,
                days=args.days,
                hourly=True,
                daily=True,
                provider=args.provider,
            ),
            self._backend.alerts(
                **location_params, provider=args.provider
            ),
            return_exceptions=True,
        )
        # A failure of either of these is a failure of the request.
        for outcome in (current, forecast):
            if isinstance(outcome, BaseException):
                raise outcome
        if isinstance(alerts, BaseException):
            logger.warning(
                "ai.location_risk.alerts_unavailable",
                extra={"reason": type(alerts).__name__},
            )
            alerts = None

        # Both legs were fetched by point, so both come back unlabelled. The
        # gazetteer already named these coordinates before the lookup, and
        # putting that name back is what turns "23.2337,77.3585: here's what
        # the data shows" into a sentence about Bhopal. Every other tool does
        # this; this one was the omission.
        current = _named(current, place)
        forecast = _named(forecast, place)

        risk = LocationRiskResult(
            purpose=args.purpose,
            location=current.location,
            considerations=_considerations(
                args.purpose, current, forecast, args.window_hours
            ),
            active_alert_count=alerts.count if alerts is not None else None,
            alert_disclaimer=alerts.disclaimer if alerts is not None else None,
            current=current.model_dump(mode="json"),
            forecast=forecast.model_dump(mode="json"),
            alerts=alerts.model_dump(mode="json") if alerts is not None else None,
        )
        sources = _unique_sources(
            "location_risk",
            [current.provenance, forecast.provenance]
            + ([alert.provenance for alert in alerts.alerts] if alerts is not None else []),
        )
        return ToolResult(tool="location_risk", data=risk.model_dump(mode="json"), sources=sources)

    async def _resolve(
        self, args: LocationInput, near: Coordinates | None = None
    ) -> Location:
        """Turn whatever the question named into one place on the map.

        Every tool resolves through here rather than handing a name to the
        weather routes, for three reasons: the ranking that knows about the
        conversation lives in one place, the resolved place is what the answer
        names, and a question that could not be resolved fails before any
        weather is fetched instead of after.
        """
        if args.coordinates is not None:
            return Location(coordinates=args.coordinates)
        assert args.location is not None
        matches = await self._backend.locations(query=args.location, limit=5, near=near)
        if not matches.results:
            raise ToolInputError(f"I could not find a location matching '{args.location}'.")
        return matches.results[0]

    @staticmethod
    def _result(tool: str, model: BaseModel, provenance: DataProvenance) -> ToolResult:
        return ToolResult(
            tool=tool,
            data=model.model_dump(mode="json"),
            sources=[SourceReference.from_provenance(tool, provenance)],
        )


def _named(model: ModelWithLocation, place: Location) -> ModelWithLocation:
    """Put the resolved place name back on a result fetched by coordinates.

    Asking the weather routes for a point gets a point back: correct, and
    unreadable. This restores the labels the gazetteer already gave us for
    exactly those coordinates while leaving the provider's own grid point and
    every measured value untouched. No name is invented and no number moves.
    """
    if place.name is None or model.location.name is not None:
        return model
    return model.model_copy(
        update={
            "location": model.location.model_copy(
                update={
                    "name": place.name,
                    "admin1": place.admin1 or model.location.admin1,
                    "country": place.country or model.location.country,
                    "country_code": place.country_code or model.location.country_code,
                }
            )
        }
    )


def _unique_sources(tool: str, provenances: Iterable[DataProvenance]) -> list[SourceReference]:
    sources: list[SourceReference] = []
    seen: set[tuple[str, str | None, str]] = set()
    for provenance in provenances:
        key = (provenance.provider_id, provenance.model, provenance.fetched_at.isoformat())
        if key not in seen:
            seen.add(key)
            sources.append(SourceReference.from_provenance(tool, provenance))
    return sources


def _considerations(
    purpose: AdvisoryPurpose,
    current: WeatherReport,
    forecast: Forecast,
    window_hours: int = 24,
) -> list[RiskConsideration]:
    """Make only conditional, factual planning observations from returned fields.

    No severity bands, thresholds, warning states, or safety claims are created
    here.  Those belong to the backend Alert Engine and its persisted results.

    Each consideration is one short line about one variable, read over the same
    window the verdict reads. Two rules keep it legible:

    * **Aggregate, do not sample.** Reporting the first wet hour beside a
      verdict that reports the whole window puts "83%, 0.7 mm" under "96%,
      13.1 mm" and reads as a contradiction. Both were true; only one was the
      answer to the question.
    * **State the figure, not advice about the figure.** "consider that
      reported precipitation in your plans" told the reader nothing they did
      not already know from the number beside it. The recommendation is the
      verdict's job, and it does it in one sentence that names the activity.
    """
    hours = forecast.hourly[: max(1, window_hours)]
    items: list[RiskConsideration] = []

    now = current.current
    if now.condition_description or now.temperature_c is not None:
        parts = [now.condition_description] if now.condition_description else []
        if now.temperature_c is not None:
            parts.append(f"{now.temperature_c:.1f} °C")
        items.append(
            RiskConsideration(
                field="current.condition_description",
                statement=f"Now: {', '.join(parts)}.",
                valid_at=now.observed_at,
            )
        )

    probabilities = [
        point.precipitation_probability_pct
        for point in hours
        if point.precipitation_probability_pct is not None
    ]
    amounts = [
        point.precipitation_mm for point in hours if point.precipitation_mm is not None
    ]
    if probabilities or amounts:
        parts = []
        if probabilities:
            parts.append(f"{max(probabilities):.0f}% peak chance")
        if amounts:
            parts.append(f"{sum(amounts):.1f} mm total")
        statement = f"Rain: {', '.join(parts)}"
        # The wettest hour is the one worth planning around, and it is the only
        # timestamp this answer needs.
        wettest = max(
            (point for point in hours if point.precipitation_mm),
            key=lambda point: point.precipitation_mm or 0.0,
            default=None,
        )
        if wettest is not None and (wettest.precipitation_mm or 0) > 0:
            statement += f"; heaviest around {wettest.valid_at.isoformat()}"
        items.append(
            RiskConsideration(
                field="forecast.hourly.precipitation",
                statement=statement + ".",
                valid_at=(wettest or hours[0]).valid_at if hours else None,
            )
        )

    gusts = [point for point in hours if point.wind_gust_ms is not None]
    if gusts:
        gustiest = max(gusts, key=lambda point: point.wind_gust_ms or -1.0)
        assert gustiest.wind_gust_ms is not None
        items.append(
            RiskConsideration(
                field="forecast.hourly.wind_gust_ms",
                # A gust reported to four decimal places is a unit conversion
                # showing through, not a measurement anyone can act on.
                statement=f"Wind: gusts to {gustiest.wind_gust_ms:.0f} m/s.",
                valid_at=gustiest.valid_at,
            )
        )

    daily_with_temperature = [
        point for point in forecast.daily if point.temperature_max_c is not None
    ]
    if purpose is AdvisoryPurpose.AGRICULTURE and daily_with_temperature:
        warmest = max(daily_with_temperature, key=lambda point: point.temperature_max_c or -999)
        assert warmest.temperature_max_c is not None
        items.append(
            RiskConsideration(
                field="forecast.daily.temperature_max_c",
                statement=(
                    f"Warmest day: {warmest.temperature_max_c:.1f} °C on "
                    f"{warmest.date.isoformat()}."
                ),
                valid_at=warmest.date,
            )
        )
    elif purpose is AdvisoryPurpose.TRAVEL and current.current.visibility_m is not None:
        items.append(
            RiskConsideration(
                field="current.visibility_m",
                statement=f"Visibility: {current.current.visibility_m:.0f} m.",
                valid_at=current.current.observed_at,
            )
        )

    return items
