"""LLM provider abstraction used only for tool selection.

The response renderer does not accept model prose.  This makes a local/free
model useful for multilingual intent interpretation and function selection
without allowing it to invent meteorological values or safety assertions.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Protocol

import httpx

from app.ai.models import LLMToolCall, LLMToolSelection
from app.config.logging import get_logger

logger = get_logger(__name__)

#: Total attempts per selection, including the first. Deliberately small: on a
#: free tier, retrying harder is what causes the next rate limit.
_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 0.5
#: Ceiling on an honoured Retry-After. A long one means the quota is spent, and
#: waiting it out would stall the request; deterministic routing answers sooner.
_MAX_RETRY_AFTER_SECONDS = 2.0


def _retry_delay(exc: Exception, attempt: int) -> float:
    """Backoff before the next attempt, honouring Retry-After when Groq sends one.

    A server that says how long to wait knows better than a fixed curve, so
    waiting exactly that long makes the retry likelier to succeed without
    adding attempts. Bounded at both ends: never shorter than the backoff,
    never long enough to hold the demo up.
    """
    backoff = _RETRY_BACKOFF_SECONDS * (attempt + 1)
    response = getattr(exc, "response", None)
    if response is None or response.status_code != 429:
        return backoff
    header = response.headers.get("retry-after")
    if not header:
        return backoff
    try:
        requested = float(header)
    except ValueError:
        # The header also permits an HTTP date; the backoff covers that case.
        return backoff
    return min(max(requested, backoff), _MAX_RETRY_AFTER_SECONDS)


VERDICT_SYSTEM_PROMPT = """You write ONE short recommendation answering the
user's question, using ONLY the retrieved weather facts given to you.

Hard rules:
- Use only the numbers listed in the facts. Never state a figure that is not
  there. Never estimate, round differently, or infer a value.
- If the facts say a variable is MISSING, say plainly that you cannot judge
  that part. Never substitute a different variable for it and never imply the
  activity is safe when a listed variable is missing.
- Answer the question that was asked. Do not describe the weather generically,
  and do not default to talking about rain unless the facts make rain the
  relevant thing.
- Do not discuss product labels, chemical rainfast periods, regulations or
  equipment specifications. You are answering on weather only.
- Do not invent alerts or warnings.
- No preamble, no bullet points, no reasoning steps. One or two sentences,
  written as advice to the person who asked. Reply in {language}.
"""

TOOL_SELECTION_SYSTEM_PROMPT = """You route weather questions to one read-only
function. You never answer the user; something else writes the answer from the
data the function returns.

Pick the function the question asks for:
- current_weather - conditions now
- hourly_forecast / daily_forecast - what is coming
- alerts - whether any alert, warning or advisory already exists. Reading these
  is always allowed.
- location_risk - travel or farming planning
- location_search - finding a place

Always pass the place the user named, written in English so the gazetteer can
resolve it: "मुंबई" becomes "Mumbai". Questions arrive in any language; one
asked in Hindi is still a weather question and still gets a function call.

Make no call at all when the message is not about weather, or names no place
and none is given in the context.

You are choosing a function, not producing facts. Never state a temperature,
rainfall or wind value, never assert that an official body has issued a
warning, and never create, change, cancel or escalate an alert - you have no
function for any of that, and the data comes from the weather service."""


class LLMProvider(Protocol):
    """Minimal provider contract that avoids locking the layer to one SDK."""

    async def select_tools(
        self,
        *,
        message: str,
        tools: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> LLMToolSelection:
        """Select read-only functions. Returned prose is intentionally discarded."""

    async def phrase_verdict(self, *, facts: str, question: str, language: str) -> str | None:
        """Word a recommendation from ``facts``. Checked by the caller before use."""

    async def aclose(self) -> None:
        """Release any provider-owned transport."""


class DisabledLLMProvider:
    """Default provider: deterministic intent detection remains fully functional."""

    async def select_tools(
        self,
        *,
        message: str,
        tools: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> LLMToolSelection:
        return LLMToolSelection()

    async def phrase_verdict(self, *, facts: str, question: str, language: str) -> str | None:
        # No model, so no rephrasing. The composed recommendation stands.
        return None

    async def aclose(self) -> None:
        return None


class OpenAICompatibleLLMProvider:
    """Tool-calling adapter for local Ollama, vLLM, llama.cpp, or compatible APIs.

    The default configuration targets Ollama's OpenAI-compatible local endpoint;
    no cloud account or proprietary SDK is required.  The provider is optional
    and a transient local-model failure falls back to deterministic routing.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 20.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._model = model
        self._owns_client = client is None
        # Sent per request rather than baked into the transport: an injected
        # client would otherwise silently drop the credential, and a missing
        # Authorization header reaches Groq as an opaque 401.
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=timeout_seconds
        )

    async def select_tools(
        self,
        *,
        message: str,
        tools: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> LLMToolSelection:
        # Context is structured and deliberately short. Tool outputs and
        # meteorological values are never sent to the model for answer-writing.
        user_payload = {"message": message, "context": context}
        request = {
            "model": self._model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": TOOL_SELECTION_SYSTEM_PROMPT},
                # ensure_ascii=False: the default escapes non-Latin text, so a
                # Hindi question would reach the model as "म..." rather
                # than "मुंबई", and the model would decline to route it.
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False),
                },
            ],
            "tools": tools,
            "tool_choice": "auto",
        }

        body = await self._post(request)
        if body is None:
            return LLMToolSelection()

        choices = body.get("choices") if isinstance(body, dict) else None
        if not isinstance(choices, list) or not choices:
            return LLMToolSelection()
        message_body = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        raw_calls = message_body.get("tool_calls", []) if isinstance(message_body, dict) else []

        calls: list[LLMToolCall] = []
        for raw in raw_calls if isinstance(raw_calls, list) else []:
            function = raw.get("function", {}) if isinstance(raw, dict) else {}
            name = function.get("name") if isinstance(function, dict) else None
            raw_arguments = function.get("arguments", "{}") if isinstance(function, dict) else "{}"
            if not isinstance(name, str):
                continue
            try:
                arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
            except json.JSONDecodeError:
                continue
            if isinstance(arguments, dict):
                calls.append(LLMToolCall(name=name, arguments=arguments))
        return LLMToolSelection(calls=calls)

    async def phrase_verdict(
        self, *, facts: str, question: str, language: str
    ) -> str | None:
        """Word the recommendation the composer already reached, from the same facts.

        The model receives the question and a list of retrieved figures, and
        nothing else — no tools, no history, no ability to fetch. It cannot add
        a fact because it is not given a way to obtain one, and the caller
        rejects the sentence outright if it states a figure the brief does not
        account for. A failure here is not a failure of the answer: the
        composed sentence is already correct, this only reads better.
        """
        payload = await self._post(
            {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": VERDICT_SYSTEM_PROMPT.format(language=language)},
                    {
                        "role": "user",
                        "content": (
                            f"Question: {question}\n\nRetrieved weather facts:\n{facts}"
                        ),
                    },
                ],
                "temperature": 0.2,
                "max_tokens": 160,
            }
        )
        if not payload:
            return None
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return None
        if not isinstance(content, str):
            return None
        # One line. A model that wrote a paragraph gets its first two sentences.
        cleaned = " ".join(content.strip().split())
        return cleaned[:400] or None

    async def _post(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """One completion, retrying transient failures. None means give up.

        Hosted endpoints reset connections and rate-limit bursts, and a dropped
        connection previously looked exactly like "the model chose no tool" —
        the request silently degraded to deterministic routing with nothing in
        the log to say why. Failures are now named.
        """
        for attempt in range(_MAX_ATTEMPTS):
            try:
                response = await self._client.post(
                    "/chat/completions", json=request, headers=self._headers
                )
                if response.status_code == 429 or response.status_code >= 500:
                    # Throttling and upstream faults are worth another try.
                    raise httpx.HTTPStatusError(
                        f"status {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPError, ValueError, TypeError) as exc:
                last = exc
                if attempt + 1 < _MAX_ATTEMPTS:
                    await asyncio.sleep(_retry_delay(exc, attempt))

        # Tool selection is an optimisation, never a correctness requirement:
        # deterministic routing still answers. So this is a warning, not an
        # error, but it must be visible.
        logger.warning(
            "ai.llm.select_tools_failed",
            extra={"model": self._model, "reason": type(last).__name__},
        )
        return None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
