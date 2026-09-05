"""Safe orchestration for WeatherGPT's conversational interface."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import replace
from datetime import datetime
from typing import Any

from pydantic import ValidationError

from app.ai.backend import BackendAPIClient, BackendAPIError
from app.ai.conversation import ConversationState, InMemoryConversationStore
from app.ai.grounding import CatalogRegistry, GroundedRenderer
from app.ai.intents import DetectedRequest, IntentDetector
from app.ai.models import (
    AlertsArguments,
    AnswerVerdict,
    ChatRequest,
    ChatResponse,
    CurrentWeatherArguments,
    ForecastArguments,
    HistoricalWeatherArguments,
    Intent,
    LLMToolSelection,
    LocationInput,
    LocationRiskResult,
    LocationRiskArguments,
    LocationSearchArguments,
    SourceReference,
    ToolResult,
    AlertConversationContext,
)
from app.ai import brief, verdict
from app.ai.providers import DisabledLLMProvider, LLMProvider, OpenAICompatibleLLMProvider
from app.ai.tools import (
    BackendWeatherTools,
    TOOL_DEFINITIONS,
    ToolInputError,
)
from app.api.v1.historical import HistoricalWeatherResponse
from app.config.logging import get_logger
from app.config.settings import Settings
from app.domain.forecast import Forecast
from app.domain.location import Coordinates, Location
from app.domain.weather import WeatherReport

logger = get_logger(__name__)

_INTENT_TOOL: dict[Intent, str] = {
    Intent.CURRENT_WEATHER: "current_weather",
    Intent.HOURLY_FORECAST: "hourly_forecast",
    Intent.DAILY_FORECAST: "daily_forecast",
    Intent.HISTORICAL_WEATHER: "historical_weather",
    Intent.ALERTS: "alerts",
    Intent.LOCATION_RISK: "location_risk",
    Intent.LOCATION_SEARCH: "location_search",
}

#: Tools a model may route to when the English keyword detector recognises
#: nothing. Ordered, so selection stays deterministic. `historical_weather` and
#: `location_search` are absent deliberately: they need a date range or a place
#: name that only the user can supply, and a model must never fill either in.
_LLM_ROUTABLE: dict[str, Intent] = {
    "current_weather": Intent.CURRENT_WEATHER,
    "hourly_forecast": Intent.HOURLY_FORECAST,
    "daily_forecast": Intent.DAILY_FORECAST,
    "alerts": Intent.ALERTS,
    "location_risk": Intent.LOCATION_RISK,
}


class AIService:
    """Run read-only tools and render their returned facts deterministically."""

    def __init__(
        self,
        *,
        tools: BackendWeatherTools,
        llm: LLMProvider | None = None,
        detector: IntentDetector | None = None,
        conversations: InMemoryConversationStore | None = None,
        catalogs: CatalogRegistry | None = None,
        renderer: GroundedRenderer | None = None,
    ) -> None:
        self._tools = tools
        self._llm = llm or DisabledLLMProvider()
        self._detector = detector or IntentDetector()
        self._conversations = conversations or InMemoryConversationStore()
        self._catalogs = catalogs or CatalogRegistry()
        self._renderer = renderer or GroundedRenderer()

    async def aclose(self) -> None:
        await self._llm.aclose()
        await self._tools.aclose()

    async def chat(self, request: ChatRequest) -> ChatResponse:
        state = await self._conversations.get_or_create(request.session_id)
        if request.alert_context is not None:
            state.alert_context = request.alert_context
            if state.location is None:
                state.location = LocationInput(
                    latitude=request.alert_context.latitude,
                    longitude=request.alert_context.longitude,
                )
        detected = self._detector.detect(
            request.message, state, language_hint=request.language
        )
        catalog, language_fallback = self._catalogs.resolve(detected.language)

        # A local/compatible LLM can route multilingual phrasing to a tool. The
        # deterministic detector remains the authorization boundary: generated
        # arguments are never trusted and model prose is never rendered.
        llm_selection = await self._llm.select_tools(
            message=request.message,
            tools=self._tools.definitions,
            # The language of *this* message, not the session default: telling
            # the model a Hindi question is English made it decline to route.
            context=_tool_context(state, language=detected.language),
        )
        selected_names = {call.name for call in llm_selection.calls if call.name in TOOL_DEFINITIONS}
        tool_name = _INTENT_TOOL.get(detected.intent)
        if tool_name and selected_names and tool_name not in selected_names:
            logger.info(
                "ai.tool_selection.disagreed",
                extra={"detected_intent": detected.intent.value, "llm_tools": sorted(selected_names)},
            )

        if tool_name is None and selected_names:
            # The keyword detector only reads English, so a question in another
            # language reaches here unrouted. A model may pick the read-only
            # tool for it, but only that: it cannot reach a tool outside this
            # map, cannot supply a date range, and cannot supply a weather
            # value. Arguments below stay deterministic.
            routed = next(
                (intent for name, intent in _LLM_ROUTABLE.items() if name in selected_names),
                None,
            )
            location = detected.location or _model_location(llm_selection, detected.language)
            if routed is not None and location is not None:
                logger.info(
                    "ai.tool_selection.routed_unknown",
                    extra={"intent": routed.value, "language": detected.language},
                )
                detected = replace(detected, intent=routed, location=location)
                tool_name = _INTENT_TOOL[routed]

        clarification = self._clarification(detected, catalog)
        if clarification:
            await self._save_context(state, detected)
            return ChatResponse(
                session_id=state.session_id,
                language=detected.language,
                language_fallback=language_fallback,
                intent=detected.intent,
                answer=clarification,
                needs_clarification=True,
            )

        if tool_name is None:
            await self._save_context(state, detected)
            return ChatResponse(
                session_id=state.session_id,
                language=detected.language,
                language_fallback=language_fallback,
                intent=Intent.UNKNOWN,
                answer=catalog.unknown(),
                needs_clarification=True,
            )

        try:
            arguments = self._arguments(detected, request.message)
            # The place the conversation is already about, used only to rank
            # gazetteer candidates. "What about Miyapur?" after a question
            # about Hyderabad should find the Miyapur in Hyderabad.
            result = await self._tools.execute(
                tool_name, arguments, near=_context_point(state, detected)
            )
        except BackendAPIError as exc:
            # Never substitute stale conversation data or model knowledge for a
            # failed backend request. The backend's safe public message is okay
            # to surface for known semantic failures such as a missing location.
            message = (
                exc.error.message
                if exc.error.code in {"LOCATION_NOT_FOUND", "INVALID_LOCATION_QUERY", "INVALID_TIME_RANGE"}
                else catalog.unavailable()
            )
            return ChatResponse(
                session_id=state.session_id,
                language=detected.language,
                language_fallback=language_fallback,
                intent=detected.intent,
                answer=message,
                needs_clarification=exc.error.code in {"LOCATION_NOT_FOUND", "INVALID_LOCATION_QUERY", "INVALID_TIME_RANGE"},
            )
        except (ToolInputError, ValidationError, ValueError) as exc:
            logger.info("ai.tool_input.invalid", extra={"intent": detected.intent.value})
            return ChatResponse(
                session_id=state.session_id,
                language=detected.language,
                language_fallback=language_fallback,
                intent=detected.intent,
                answer=str(exc),
                needs_clarification=True,
            )

        await self._save_context(state, detected, resolved=_answered_location(result))
        sources = _unique_sources(result.sources)
        rendered = self._renderer.render(detected.intent, [result], catalog)
        return ChatResponse(
            session_id=state.session_id,
            language=detected.language,
            language_fallback=language_fallback,
            intent=detected.intent,
            # Provenance is returned structurally in `sources`, not pasted
            # into the answer: a spoken reply should not recite a licence and a
            # URL, and a caller can render the source however it likes.
            answer=rendered.answer,
            tool_results=[result],
            sources=sources,
            safety_note=rendered.safety_note,
            verdict=await _verdict_for(detected, result, request.message, self._llm),
        )

    def _clarification(self, detected: DetectedRequest, catalog) -> str | None:
        if detected.intent is Intent.UNKNOWN:
            return None
        if detected.intent is Intent.LOCATION_SEARCH:
            if not detected.location or not detected.location.location:
                return catalog.label("need_search_place")
            return None
        if detected.location is None:
            return catalog.need_location()
        if detected.intent is Intent.HISTORICAL_WEATHER and not _has_past_window(detected):
            return catalog.need_history_dates()
        return None

    @staticmethod
    def _arguments(detected: DetectedRequest, original_message: str) -> dict[str, Any]:
        if detected.intent is Intent.LOCATION_SEARCH:
            assert detected.location is not None and detected.location.location is not None
            return LocationSearchArguments(query=detected.location.location).model_dump(exclude_none=True)
        assert detected.location is not None
        location = detected.location.model_dump(exclude_none=True)
        # A source named by the user, never one proposed by a model.
        if detected.provider:
            location["provider"] = detected.provider
        if detected.intent is Intent.CURRENT_WEATHER:
            return CurrentWeatherArguments(**location).model_dump(exclude_none=True)
        if detected.intent in {Intent.HOURLY_FORECAST, Intent.DAILY_FORECAST}:
            return ForecastArguments(**location, days=detected.days).model_dump(exclude_none=True)
        if detected.intent is Intent.HISTORICAL_WEATHER:
            assert _has_past_window(detected)
            hours = detected.local_hours or (None, None)
            return HistoricalWeatherArguments(
                **location,
                start=detected.start,
                end=detected.end,
                start_time=detected.start_time,
                end_time=detected.end_time,
                hour_from=hours[0],
                hour_to=hours[1],
            ).model_dump(exclude_none=True)
        if detected.intent is Intent.ALERTS:
            return AlertsArguments(**location).model_dump(exclude_none=True)
        if detected.intent is Intent.LOCATION_RISK:
            # The same window `_verdict_for` reads below, so the considerations
            # and the one-line answer above them are never about different
            # spans of time.
            return LocationRiskArguments(
                **location,
                days=detected.days,
                purpose=detected.purpose,
                window_hours=detected.horizon_hours or 24,
            ).model_dump(exclude_none=True)
        raise ToolInputError("Unsupported request.")

    async def _save_context(
        self,
        state: ConversationState,
        detected: DetectedRequest,
        *,
        resolved: Location | None = None,
    ) -> None:
        if detected.location is not None:
            state.location = detected.location
        if resolved is not None:
            state.resolved = resolved
        state.language = detected.language
        state.purpose = detected.purpose
        state.last_intent = detected.intent
        await self._conversations.save(state)


def build_ai_service(settings: Settings) -> AIService:
    """Create an AI service from configuration without adding an LLM SDK dependency."""
    backend = BackendAPIClient(
        base_url=settings.AI_BACKEND_BASE_URL,
        api_prefix=settings.API_V1_PREFIX,
        timeout_seconds=settings.AI_BACKEND_TIMEOUT_SECONDS,
    )
    # The AI layer reaches its own service over HTTP, so this address must name
    # the port the process is actually listening on. Logged at startup because
    # a mismatch is otherwise invisible until someone asks a question and every
    # answer comes back unavailable. Set AI_BACKEND_BASE_URL when serving on a
    # port other than the default.
    logger.info(
        "ai.backend.configured",
        extra={
            "base_url": settings.AI_BACKEND_BASE_URL,
            "api_prefix": settings.API_V1_PREFIX,
            "timeout_seconds": settings.AI_BACKEND_TIMEOUT_SECONDS,
        },
    )
    provider_name = settings.AI_LLM_PROVIDER.casefold()
    llm: LLMProvider
    if provider_name in {"openai_compatible", "groq"}:
        # Groq speaks the same dialect; only the endpoint and credential differ,
        # so it needs no second adapter.
        llm = OpenAICompatibleLLMProvider(
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            timeout_seconds=settings.AI_LLM_TIMEOUT_SECONDS,
        )
        logger.info(
            "ai.llm.configured",
            extra={"provider": provider_name, "model": settings.llm_model},
        )
    else:
        llm = DisabledLLMProvider()
    return AIService(tools=BackendWeatherTools(backend), llm=llm)


#: A model-proposed location that is really a coordinate pair. Rejected: an
#: invented name fails geocoding loudly, but an invented pair silently resolves
#: to a real point on Earth that nobody asked about.
_COORDINATE_LIKE = re.compile(r"^[\d\s.,;:+-]+$")


def _has_past_window(detected) -> bool:
    """Whether the question named a past window in either form."""
    return (detected.start is not None and detected.end is not None) or (
        detected.start_time is not None and detected.end_time is not None
    )


def _model_location(
    selection: LLMToolSelection, language: str
) -> LocationInput | None:
    """A place name the model read out of a message the detector cannot parse.

    Accepted only for a non-English message. The keyword detector reads
    English, so an English message it could not classify genuinely is not a
    weather question — deferring to the model there would let any off-topic
    request become a weather lookup.

    The name is not treated as a fact: the backend gazetteer must resolve it,
    an unknown name becomes an apology and an ambiguous one becomes a question.
    """
    if language.split("-", 1)[0].casefold() == "en":
        return None
    for call in selection.calls:
        if call.name not in _LLM_ROUTABLE:
            continue
        proposed = call.arguments.get("location")
        if isinstance(proposed, str) and proposed.strip():
            candidate = proposed.strip()[:200]
            if not _COORDINATE_LIKE.match(candidate):
                return LocationInput(location=candidate)
    return None


async def _verdict_for(
    detected: DetectedRequest,
    result: ToolResult,
    question: str,
    llm: LLMProvider,
) -> AnswerVerdict | None:
    """The one-line recommendation for this answer, computed from what came back.

    Parsing the result back into its typed model is deliberate: the verdict
    reads the same validated fields the renderer does, so it cannot be built
    from a shape the backend did not actually return. Anything unparseable
    produces no verdict rather than a hedge.

    The model, when one is configured, only *words* the recommendation the
    composer already reached, from the same brief. Its sentence is rejected if
    it states a figure the brief does not account for, so the wording can
    improve while the facts cannot change. A failure to reach it is not a
    failure of the answer.
    """
    try:
        if detected.intent is Intent.CURRENT_WEATHER:
            data: object = WeatherReport.model_validate(result.data)
        elif detected.intent in {Intent.HOURLY_FORECAST, Intent.DAILY_FORECAST}:
            data = Forecast.model_validate(result.data)
        elif detected.intent is Intent.HISTORICAL_WEATHER:
            data = HistoricalWeatherResponse.model_validate(result.data).observations
        elif detected.intent is Intent.LOCATION_RISK:
            risk = LocationRiskResult.model_validate(result.data)
            data = Forecast.model_validate(risk.forecast)
        else:
            return None
    except ValidationError:
        return None

    # The place the answer turned out to be about, and any alert count that
    # came back with it. Both are context for the recommendation; neither is
    # invented when absent.
    answered = _answered_location(result)
    alerts = result.data.get("active_alert_count") if isinstance(result.data, dict) else None

    found = verdict.evidence(
        intent=detected.intent,
        data=data,
        question=question,
        place=answered.display_name if answered else "",
        horizon_hours=detected.horizon_hours or 24,
        purpose=detected.purpose,
            target_date=detected.target_date,
            local_hours=detected.local_hours,
        active_alerts=alerts if isinstance(alerts, int) else None,
    )
    if found is None:
        return None

    authored: str | None = None
    try:
        authored = await llm.phrase_verdict(
            facts=brief.as_facts(found),
            question=question,
            language=detected.language,
        )
    except Exception:  # noqa: BLE001 - the composed recommendation already answers
        logger.info("ai.verdict.phrasing_unavailable", exc_info=True)

    reading = verdict.render(
        found,
        language=detected.language,
        format_time=_local_hour,
        authored=authored,
    )
    if reading is None:
        return None
    return AnswerVerdict(text=reading.text, icon=reading.icon, caveat=reading.caveat)


def _local_hour(moment: datetime) -> str:
    """A forecast instant on an Indian clock: DD-MM-YYYY and a 12-hour time.

    Only the hour loses its leading zero. Stripping zeros from the whole string
    turns 05-09-2026 into 5-09-2026, which reads as a different convention.
    """
    hour = moment.strftime("%I").lstrip("0") or "12"
    return f"{moment.strftime('%d-%m-%Y')}, {hour}:{moment.strftime('%M %p')}"


def _answered_location(result: ToolResult) -> Location | None:
    """The place an answer turned out to be about, for the next question."""
    payload = result.data.get("location") if isinstance(result.data, dict) else None
    if not isinstance(payload, dict):
        return None
    try:
        return Location.model_validate(payload)
    except ValidationError:
        return None


def _context_point(state: ConversationState, detected) -> Coordinates | None:
    """Where the conversation is, as a point, when it has one.

    Only a place already resolved to coordinates counts. A name still waiting
    to be resolved cannot bias its own resolution.
    """
    if detected.location is not None and detected.location.coordinates is not None:
        return None  # the question already named a point; nothing to bias
    if state.resolved is not None:
        return state.resolved.coordinates
    return state.location.coordinates if state.location else None


def _tool_context(state: ConversationState, *, language: str | None = None) -> dict[str, Any]:
    """Only non-weather context is available to the model's tool-selection step."""
    return {
        "location": state.location.model_dump(exclude_none=True) if state.location else None,
        "language": language or state.language,
        "purpose": state.purpose.value,
        "last_intent": state.last_intent.value,
        "alert_context": state.alert_context.model_dump(mode="json", exclude_none=True) if state.alert_context else None,
    }


def _unique_sources(sources: Iterable[SourceReference]) -> list[SourceReference]:
    unique: list[SourceReference] = []
    seen: set[tuple[str, str | None, str, str]] = set()
    for source in sources:
        key = (source.provider_id, source.model, source.fetched_at.isoformat(), source.tool)
        if key not in seen:
            seen.add(key)
            unique.append(source)
    return unique
