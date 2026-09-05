"""Short-lived, in-memory conversation context.

The store intentionally holds only the small amount of structured context that
is safe to reuse for a follow-up: location, language, purpose and last intent.
Weather results are never reused as current data; every new question executes a
fresh backend tool call.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from pydantic import BaseModel

from app.ai.models import AdvisoryPurpose, AlertConversationContext, Intent, LocationInput
from app.domain.location import Location


class ConversationState(BaseModel):
    session_id: str
    updated_at: datetime
    location: LocationInput | None = None
    #: The place the last answer was actually about, as the gazetteer
    #: resolved it. A name still waiting to be resolved cannot bias its own
    #: resolution, so only a settled point is kept here.
    resolved: Location | None = None
    language: str = "en"
    purpose: AdvisoryPurpose = AdvisoryPurpose.GENERAL
    last_intent: Intent = Intent.UNKNOWN
    alert_context: AlertConversationContext | None = None


class InMemoryConversationStore:
    """Bounded best-effort context store for a single process.

    It is deliberately not a source of truth and can be replaced by a scoped
    encrypted session store at deployment time without changing the orchestrator.
    """

    def __init__(self, *, ttl: timedelta = timedelta(hours=2), max_sessions: int = 10_000):
        self._ttl = ttl
        self._max_sessions = max_sessions
        self._items: dict[str, ConversationState] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(self, session_id: str | None) -> ConversationState:
        now = datetime.now(timezone.utc)
        async with self._lock:
            self._prune(now)
            if session_id and session_id in self._items:
                return self._items[session_id].model_copy(deep=True)
            identifier = session_id or uuid4().hex
            state = ConversationState(session_id=identifier, updated_at=now)
            self._items[identifier] = state
            self._trim()
            return state.model_copy(deep=True)

    async def save(self, state: ConversationState) -> None:
        async with self._lock:
            state.updated_at = datetime.now(timezone.utc)
            self._items[state.session_id] = state.model_copy(deep=True)
            self._prune(state.updated_at)
            self._trim()

    def _prune(self, now: datetime) -> None:
        cutoff = now - self._ttl
        expired = [key for key, value in self._items.items() if value.updated_at < cutoff]
        for key in expired:
            del self._items[key]

    def _trim(self) -> None:
        overflow = len(self._items) - self._max_sessions
        if overflow <= 0:
            return
        oldest = sorted(self._items.values(), key=lambda item: item.updated_at)[:overflow]
        for item in oldest:
            del self._items[item.session_id]
