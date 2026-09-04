"""Tests for the downstream-only, grounded conversational layer."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx
import pytest

from app.ai.backend import BackendAPIClient
from app.ai.models import ChatRequest, LLMToolCall, LLMToolSelection
from app.ai.providers import DisabledLLMProvider
from app.ai.service import AIService
from app.ai.tools import BackendWeatherTools
from app.api.v1.alerts import AlertListResponse, DISCLAIMER
from app.api.v1.historical import HistoricalObservation, HistoricalWeatherResponse, TimeRange
from app.api.v1.locations import LocationSearchResponse
from app.domain.forecast import DailyForecastPoint, Forecast, HourlyForecastPoint
from app.domain.location import Coordinates, Location
from app.domain.provenance import DataProvenance
from app.domain.weather import CurrentWeather, WeatherCondition, WeatherReport

NOW = datetime(2026, 9, 4, 7, 30, tzinfo=timezone.utc)


def _provenance() -> DataProvenance:
    return DataProvenance(
        provider_id="validated-source",
        provider_name="Validated Source",
        model="model-a",
        fetched_at=NOW,
        source_url="https://weather.example/contract",
        license="CC-BY-4.0",
        attribution="Weather data by Validated Source",
    )


def _report() -> WeatherReport:
    return WeatherReport(
        location=Location(
            coordinates=Coordinates(latitude=28.61, longitude=77.21),
            name="New Delhi",
            country="India",
            timezone="Asia/Kolkata",
        ),
        current=CurrentWeather(
            observed_at=NOW,
            temperature_c=31.4,
            condition=WeatherCondition.PARTLY_CLOUDY,
            condition_description="Partly cloudy",
        ),
        provenance=_provenance(),
    )


def _forecast() -> Forecast:
    return Forecast(
        location=_report().location,
        daily=[
            DailyForecastPoint(
                date=NOW.date(),
                temperature_min_c=26.0,
                temperature_max_c=34.0,
                precipitation_sum_mm=4.0,
                condition=WeatherCondition.RAIN,
                condition_description="Slight rain",
            )
        ],
        provenance=_provenance(),
    )


def _hourly_forecast() -> Forecast:
    forecast = _forecast()
    return forecast.model_copy(
        update={
            "hourly": [
                HourlyForecastPoint(
                    valid_at=NOW,
                    temperature_c=31.0,
                    precipitation_probability_pct=45.0,
                    precipitation_mm=0.4,
                    wind_speed_ms=5.0,
                    wind_gust_ms=8.0,
                    condition=WeatherCondition.RAIN_SHOWERS,
                    condition_description="Slight rain showers",
                )
            ]
        }
    )


def _gazetteer(*results: Location) -> httpx.Response:
    """The gazetteer response every named place now goes through."""
    found = list(results) or [_report().location]
    return httpx.Response(
        200,
        json=LocationSearchResponse(
            query="q", count=len(found), results=found
        ).model_dump(mode="json"),
    )


def _service(
    handler,
    *,
    llm=None,
    places: tuple[Location, ...] = (),
    resolve_places: bool = True,
) -> AIService:
    """A service whose backend is `handler`, with the gazetteer answered for it.

    A named place is resolved to a point before any weather is fetched, so a
    handler that only knows the weather routes would never be reached. Tests
    that want to assert on the gazetteer call pass their own handler through
    without going via this shim.
    """

    def with_places(request: httpx.Request) -> httpx.Response:
        if resolve_places and request.url.path == "/api/v1/locations/search":
            return _gazetteer(*places)
        return handler(request)

    client = httpx.AsyncClient(
        base_url="https://backend.test", transport=httpx.MockTransport(with_places)
    )
    backend = BackendAPIClient(
        base_url="https://backend.test", api_prefix="/api/v1", client=client
    )
    return AIService(tools=BackendWeatherTools(backend), llm=llm or DisabledLLMProvider())


@pytest.mark.asyncio
async def test_current_answer_uses_the_backend_contract_and_keeps_provenance():
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.method == "GET"
        assert request.url.path == "/api/v1/weather/current"
        # The place was resolved to a point first, so the weather route is
        # asked about coordinates rather than about a name.
        assert float(request.url.params["latitude"]) == 28.61
        assert float(request.url.params["longitude"]) == 77.21
        return httpx.Response(200, json=_report().model_dump(mode="json"))

    service = _service(handler)
    response = await service.chat(ChatRequest(message="What is the weather in New Delhi?"))
    await service.aclose()

    assert len(calls) == 1
    assert response.intent.value == "current_weather"
    assert "31.4 °C" in response.answer
    # Provenance travels in `sources`, not inside the prose answer.
    assert response.sources[0].provider_id == "validated-source"
    assert response.sources[0].fetched_at == NOW
    assert "Sources and freshness" not in response.answer


@pytest.mark.asyncio
async def test_unavailable_backend_never_falls_back_to_model_weather_knowledge():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503,
            json={
                "error": {
                    "code": "WEATHER_DATA_UNAVAILABLE",
                    "message": "No source produced validated data.",
                }
            },
        )

    service = _service(handler)
    response = await service.chat(ChatRequest(message="What is the weather in New Delhi?"))
    await service.aclose()

    assert response.tool_results == []
    assert response.sources == []
    assert "temporarily unavailable" in response.answer
    assert "31.4" not in response.answer


@pytest.mark.asyncio
async def test_alert_tool_is_get_only_and_preserves_the_not_official_disclaimer():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/v1/weather/alerts"
        return httpx.Response(
            200,
            json=AlertListResponse(
                requested={"latitude": 28.61, "longitude": 77.21},
                search_radius_km=25.0,
                count=0,
                truncated=False,
                disclaimer=DISCLAIMER,
                alerts=[],
            ).model_dump(mode="json"),
        )

    service = _service(handler)
    response = await service.chat(ChatRequest(message="Are there alerts at 28.61, 77.21?"))
    await service.aclose()

    assert response.intent.value == "alerts"
    assert response.tool_results[0].tool == "alerts"
    assert "not official meteorological warnings" in response.answer


@pytest.mark.asyncio
async def test_follow_up_reuses_location_but_retrieves_fresh_forecast_data():
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path.endswith("/current"):
            return httpx.Response(200, json=_report().model_dump(mode="json"))
        assert request.url.path == "/api/v1/forecast"
        assert float(request.url.params["latitude"]) == 28.61
        return httpx.Response(200, json=_forecast().model_dump(mode="json"))

    service = _service(handler)
    first = await service.chat(ChatRequest(message="Weather in New Delhi?"))
    second = await service.chat(ChatRequest(message="What about tomorrow?", session_id=first.session_id))
    await service.aclose()

    assert [p for p in paths if not p.endswith("/search")] == [
        "/api/v1/weather/current",
        "/api/v1/forecast",
    ]
    assert second.intent.value == "daily_forecast"
    assert second.tool_results[0].tool == "daily_forecast"
    assert "34 °C" in second.answer


@pytest.mark.asyncio
async def test_hourly_historical_and_location_risk_tools_use_only_their_api_contracts():
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/api/v1/weather/current":
            return httpx.Response(200, json=_report().model_dump(mode="json"))
        if request.url.path == "/api/v1/forecast":
            return httpx.Response(200, json=_hourly_forecast().model_dump(mode="json"))
        if request.url.path == "/api/v1/locations/search":
            return httpx.Response(
                200,
                json=LocationSearchResponse(
                    query="New Delhi", count=1, results=[_report().location]
                ).model_dump(mode="json"),
            )
        if request.url.path == "/api/v1/weather/historical":
            return httpx.Response(
                200,
                json=HistoricalWeatherResponse(
                    requested={"latitude": 28.61, "longitude": 77.21},
                    range=TimeRange(start=NOW, end=NOW),
                    search_radius_km=25.0,
                    count=1,
                    truncated=False,
                    observations=[
                        HistoricalObservation(
                            latitude=28.61,
                            longitude=77.21,
                            distance_km=0.0,
                            weather=_report().current,
                            provenance=_provenance(),
                        )
                    ],
                ).model_dump(mode="json"),
            )
        if request.url.path == "/api/v1/weather/alerts":
            return httpx.Response(
                200,
                json=AlertListResponse(
                    requested={"latitude": 28.61, "longitude": 77.21},
                    search_radius_km=25.0,
                    count=0,
                    truncated=False,
                    disclaimer=DISCLAIMER,
                    alerts=[],
                ).model_dump(mode="json"),
            )
        raise AssertionError(f"unexpected backend path {request.url.path}")

    service = _service(handler)
    hourly = await service.chat(ChatRequest(message="Hourly forecast in New Delhi"))
    historical = await service.chat(ChatRequest(message="What was weather in New Delhi yesterday?"))
    risk = await service.chat(ChatRequest(message="Travel risk at 28.61, 77.21"))
    await service.aclose()

    assert hourly.tool_results[0].tool == "hourly_forecast"
    assert "Slight rain showers" in hourly.answer
    assert historical.tool_results[0].tool == "historical_weather"
    assert historical.sources[0].provider_id == "validated-source"
    assert risk.tool_results[0].tool == "location_risk"
    assert risk.safety_note == DISCLAIMER
    assert "/api/v1/weather/historical" in paths
    assert "/api/v1/weather/alerts" in paths


class _InventingSelector:
    """Simulates an unsafe model call; its arguments must never become facts."""

    async def select_tools(self, **kwargs: Any) -> LLMToolSelection:
        return LLMToolSelection(
            calls=[LLMToolCall(name="current_weather", arguments={"location": "Invented City"})]
        )

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_model_tool_arguments_cannot_turn_an_unknown_request_into_a_weather_lookup():
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("an unknown request must not call the backend")

    service = _service(handler, llm=_InventingSelector())
    response = await service.chat(ChatRequest(message="Tell me a joke."))
    await service.aclose()

    assert response.intent.value == "unknown"
    assert response.tool_results == []
    assert "Invented City" not in response.answer


class _Selector:
    """Returns a fixed tool choice, as a local model would."""

    def __init__(self, name: str) -> None:
        self._name = name

    async def select_tools(self, **kwargs: Any) -> LLMToolSelection:
        return LLMToolSelection(calls=[LLMToolCall(name=self._name, arguments={})])

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_a_place_named_without_a_preposition_still_reaches_the_search_tool():
    queries: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/locations/search"
        queries.append(request.url.params["q"])
        return httpx.Response(
            200,
            json=LocationSearchResponse(
                query="Delhi", count=1, results=[_report().location]
            ).model_dump(mode="json"),
        )

    service = _service(handler, resolve_places=False)
    for message in ("find Delhi", "where is Delhi", "search Mumbai"):
        response = await service.chat(ChatRequest(message=message))
        assert response.needs_clarification is False, message
    await service.aclose()

    assert queries == ["Delhi", "Delhi", "Mumbai"]


@pytest.mark.asyncio
async def test_a_travel_destination_is_read_as_the_location_but_an_infinitive_is_not():
    searched: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/v1/locations/search":
            searched.append(request.url.params["q"])
            return httpx.Response(
                200,
                json=LocationSearchResponse(
                    query="Pune", count=1, results=[_report().location]
                ).model_dump(mode="json"),
            )
        if path == "/api/v1/forecast":
            return httpx.Response(200, json=_hourly_forecast().model_dump(mode="json"))
        if path == "/api/v1/weather/alerts":
            return httpx.Response(
                200,
                json=AlertListResponse(
                    requested={"latitude": 28.61, "longitude": 77.21},
                    search_radius_km=25.0,
                    count=0,
                    truncated=False,
                    disclaimer=DISCLAIMER,
                    alerts=[],
                ).model_dump(mode="json"),
            )
        return httpx.Response(200, json=_report().model_dump(mode="json"))

    service = _service(handler, resolve_places=False)
    travel = await service.chat(ChatRequest(message="Is it safe to travel to Pune?"))
    # "going to rain" is not a destination; inventing a place here would send a
    # weather word to the geocoder and answer about the wrong point on Earth.
    weather_verb = await service.chat(
        ChatRequest(message="Is it going to rain?", session_id=travel.session_id)
    )
    # A question with no place in it, asked in a conversation that has one,
    # is about that one.
    fresh = await service.chat(ChatRequest(message="Is it going to rain?"))
    await service.aclose()

    assert travel.intent.value == "location_risk"
    # "rain" never reached the geocoder; only the place a person named did.
    assert set(searched) == {"Pune"}
    assert weather_verb.needs_clarification is False
    assert fresh.needs_clarification is True
    assert "Which location" in fresh.answer


@pytest.mark.asyncio
async def test_a_model_may_route_a_non_english_question_only_to_a_location_already_known():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, json=_report().model_dump(mode="json"))

    # The keyword detector reads no Devanagari, so routing must come from the
    # model; the coordinates come from the user, never from the model.
    service = _service(handler, llm=_Selector("current_weather"))
    response = await service.chat(ChatRequest(message="मौसम 28.61, 77.21"))
    await service.aclose()

    assert calls == ["/api/v1/weather/current"]
    assert response.intent.value == "current_weather"
    assert response.language == "hi"
    assert response.language_fallback is False
    # Rendered in Hindi, but the value is copied from the backend response.
    assert "तापमान" in response.answer
    assert "31.4 °C" in response.answer
    assert "स्रोत और नवीनता" not in response.answer
    assert response.sources[0].provider_id == "validated-source"


@pytest.mark.asyncio
async def test_a_model_cannot_route_to_a_tool_whose_arguments_only_a_user_can_supply():
    """A model asking for history does not get history.

    The question names a point and asks about now, so it is answered from
    current conditions. What the model wanted — a date range only a person can
    supply — is never reached.
    """
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path.endswith("/historical"):  # pragma: no cover
            raise AssertionError("historical_weather needs a date range from the user")
        return httpx.Response(200, json=_report().model_dump(mode="json"))

    service = _service(handler, llm=_Selector("historical_weather"))
    response = await service.chat(ChatRequest(message="मौसम 28.61, 77.21"))
    await service.aclose()

    assert response.intent.value == "current_weather"
    assert paths == ["/api/v1/weather/current"]


class _LocatingSelector:
    """A model naming the place it read out of a non-English question."""

    def __init__(self, name: str, location: str) -> None:
        self._name = name
        self._location = location

    async def select_tools(self, **kwargs: Any) -> LLMToolSelection:
        return LLMToolSelection(
            calls=[LLMToolCall(name=self._name, arguments={"location": self._location})]
        )

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_a_hindi_question_resolves_its_place_through_the_backend_gazetteer():
    paths: list[str] = []

    queries: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/api/v1/locations/search":
            queries.append(request.url.params["q"])
            return httpx.Response(
                200,
                json=LocationSearchResponse(
                    query="q", count=1, results=[_report().location]
                ).model_dump(mode="json"),
            )
        if request.url.path == "/api/v1/weather/current":
            assert float(request.url.params["latitude"]) == 28.61
            return httpx.Response(200, json=_report().model_dump(mode="json"))
        raise AssertionError(f"unexpected path {request.url.path}")

    service = _service(
        handler, llm=_LocatingSelector("current_weather", "Delhi"), resolve_places=False
    )
    response = await service.chat(ChatRequest(message="दिल्ली में मौसम कैसा है?"))
    await service.aclose()

    # The place is read out of the Hindi sentence and sent to the gazetteer,
    # which is what resolves it — no transliteration table, no city dictionary,
    # and no place name taken from the model.
    assert queries == ["दिल्ली"]
    assert [p for p in paths if not p.endswith("/search")] == ["/api/v1/weather/current"]
    assert response.intent.value == "current_weather"
    assert "31.4 °C" in response.answer


@pytest.mark.asyncio
async def test_a_model_named_place_is_refused_for_an_english_question():
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("an off-topic English request must not call the backend")

    # English is a language the detector reads, so its UNKNOWN is informative.
    service = _service(handler, llm=_LocatingSelector("current_weather", "Invented City"))
    response = await service.chat(ChatRequest(message="Tell me a joke."))
    await service.aclose()

    assert response.intent.value == "unknown"
    assert response.tool_results == []


@pytest.mark.asyncio
async def test_model_supplied_coordinates_are_never_accepted_as_a_location():
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("a model must not place the user on the map")

    service = _service(handler, llm=_LocatingSelector("current_weather", "51.5, -0.12"))
    response = await service.chat(ChatRequest(message="मौसम कैसा है?"))
    await service.aclose()

    # The question is a weather question with no place in it and no place in
    # the session, so it becomes a question back — not a lookup at coordinates
    # a model produced. Nothing reached the backend.
    assert response.needs_clarification is True
    assert response.tool_results == []


def test_groq_selects_the_hosted_endpoint_and_its_own_credential():
    from app.config.settings import Settings

    # _env_file=None so the result is the code's defaults, not whatever the
    # developer happens to have in backend/.env.
    groq = Settings(_env_file=None, AI_LLM_PROVIDER="groq", GROQ_API_KEY="test-key")
    assert groq.llm_base_url == "https://api.groq.com/openai/v1"
    assert groq.llm_model == "openai/gpt-oss-120b"
    assert groq.llm_api_key == "test-key"

    # The default stays local and credential-free.
    local = Settings(_env_file=None)
    assert local.AI_LLM_PROVIDER == "disabled"
    assert local.llm_base_url.startswith("http://127.0.0.1")
    assert local.llm_api_key is None


@pytest.mark.asyncio
async def test_the_groq_provider_sends_tools_and_parses_a_tool_call():
    from app.ai.providers import OpenAICompatibleLLMProvider
    from app.ai.tools import TOOL_DEFINITIONS

    sent: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        sent.update(_json.loads(request.content))
        assert request.headers["Authorization"] == "Bearer test-key"
        # The shape Groq returns for a tool call.
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "current_weather",
                                        "arguments": '{"location": "Delhi"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )

    client = httpx.AsyncClient(
        base_url="https://api.groq.test", transport=httpx.MockTransport(handler)
    )
    provider = OpenAICompatibleLLMProvider(
        base_url="https://api.groq.test",
        model="openai/gpt-oss-120b",
        api_key="test-key",
        client=client,
    )
    selection = await provider.select_tools(
        message="मौसम",
        tools=[d.openai_schema() for d in TOOL_DEFINITIONS.values()],
        context={},
    )
    await client.aclose()

    assert sent["model"] == "openai/gpt-oss-120b"
    assert sent["temperature"] == 0
    assert {tool["function"]["name"] for tool in sent["tools"]} >= {"current_weather", "alerts"}
    assert selection.calls[0].name == "current_weather"
    assert selection.calls[0].arguments == {"location": "Delhi"}


@pytest.mark.asyncio
async def test_a_dropped_connection_is_retried_and_then_reported_not_silently_empty(caplog):
    import logging

    from app.ai.providers import OpenAICompatibleLLMProvider

    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        raise httpx.ReadError("connection reset")

    client = httpx.AsyncClient(
        base_url="https://api.groq.test", transport=httpx.MockTransport(handler)
    )
    provider = OpenAICompatibleLLMProvider(
        base_url="https://api.groq.test", model="m", api_key="k", client=client
    )
    with caplog.at_level(logging.WARNING):
        selection = await provider.select_tools(message="hi", tools=[], context={})
    await client.aclose()

    assert attempts["n"] == 3, "a transient transport failure should be retried"
    assert selection.calls == []
    # Without this the request degrades to deterministic routing with no trace.
    assert "ai.llm.select_tools_failed" in caplog.text


@pytest.mark.asyncio
async def test_a_rate_limited_response_is_retried():
    from app.ai.providers import OpenAICompatibleLLMProvider

    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx.Response(429, json={"error": "slow down"})
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "current_weather",
                                        "arguments": '{"location": "Delhi"}',
                                    }
                                }
                            ]
                        }
                    }
                ]
            },
        )

    client = httpx.AsyncClient(
        base_url="https://api.groq.test", transport=httpx.MockTransport(handler)
    )
    provider = OpenAICompatibleLLMProvider(
        base_url="https://api.groq.test", model="m", api_key="k", client=client
    )
    selection = await provider.select_tools(message="hi", tools=[], context={})
    await client.aclose()

    assert attempts["n"] == 3
    assert selection.calls[0].name == "current_weather"


@pytest.mark.asyncio
async def test_a_non_latin_question_reaches_the_model_unescaped():
    from app.ai.providers import OpenAICompatibleLLMProvider

    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.content.decode("utf-8")
        return httpx.Response(200, json={"choices": [{"message": {}}]})

    client = httpx.AsyncClient(
        base_url="https://api.groq.test", transport=httpx.MockTransport(handler)
    )
    provider = OpenAICompatibleLLMProvider(
        base_url="https://api.groq.test", model="m", api_key="k", client=client
    )
    await provider.select_tools(message="मुंबई में मौसम", tools=[], context={})
    await client.aclose()

    # json.dumps defaults to ensure_ascii=True, which would send escapes and
    # measurably stopped the model routing Hindi questions.
    assert "मुंबई" in seen["body"]
    assert "\\u092e" not in seen["body"]


@pytest.mark.asyncio
async def test_the_chat_route_is_mounted_and_returns_the_grounded_contract(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_report().model_dump(mode="json"))

    monkeypatch.setattr("app.main.build_ai_service", lambda settings: _service(handler))

    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app()) as client:
        response = client.post("/api/v1/ai/chat", json={"message": "weather in New Delhi"})

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "current_weather"
    assert body["sources"][0]["provider_id"] == "validated-source"
    assert "31.4 °C" in body["answer"]
