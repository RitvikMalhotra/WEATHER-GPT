"""Public models and safe internal value objects for the AI layer."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.alert import (
    AlertEvidence,
    AlertKind,
    AlertSeverity,
    AlertSourceType,
    AlertStatus,
    AlertType,
)
from app.domain.location import Coordinates, Location
from app.domain.provenance import DataProvenance


class Intent(str, Enum):
    """Supported read-only conversational intents."""

    CURRENT_WEATHER = "current_weather"
    HOURLY_FORECAST = "hourly_forecast"
    DAILY_FORECAST = "daily_forecast"
    HISTORICAL_WEATHER = "historical_weather"
    ALERTS = "alerts"
    LOCATION_RISK = "location_risk"
    LOCATION_SEARCH = "location_search"
    UNKNOWN = "unknown"


class AdvisoryPurpose(str, Enum):
    """Non-safety-critical contexts that can shape a factual advisory."""

    GENERAL = "general"
    AGRICULTURE = "agriculture"
    TRAVEL = "travel"
    #: Boating, fishing and anything else decided at sea. Kept apart from
    #: TRAVEL because the variables that matter differ — and because the ones
    #: that matter most (wave height, swell, sea state) are not in this
    #: system's forecast at all, which an answer has to say rather than skip.
    MARINE = "marine"
    OUTDOOR_EVENT = "outdoor_event"


class ToolInputModel(BaseModel):
    """Function inputs are closed schemas; generated extra fields are rejected."""

    model_config = ConfigDict(extra="forbid")


class LocationInput(ToolInputModel):
    """A location supplied as one complete coordinate pair or a place name."""

    latitude: float | None = Field(default=None, ge=-90.0, le=90.0)
    longitude: float | None = Field(default=None, ge=-180.0, le=180.0)
    location: str | None = Field(default=None, min_length=1, max_length=200)

    @model_validator(mode="after")
    def _require_one_location_form(self) -> "LocationInput":
        has_coordinates = self.latitude is not None and self.longitude is not None
        half_coordinates = (self.latitude is None) != (self.longitude is None)
        if half_coordinates:
            raise ValueError("latitude and longitude must be supplied together")
        if has_coordinates == bool(self.location and self.location.strip()):
            raise ValueError("provide either coordinates or a location name")
        if self.location:
            self.location = self.location.strip()
        return self

    @property
    def coordinates(self) -> Coordinates | None:
        if self.latitude is None or self.longitude is None:
            return None
        return Coordinates(latitude=self.latitude, longitude=self.longitude)


class CurrentWeatherArguments(LocationInput):
    provider: str | None = Field(default=None, max_length=100)


class ForecastArguments(LocationInput):
    days: int = Field(default=7, ge=1, le=16)
    provider: str | None = Field(default=None, max_length=100)


class HistoricalWeatherArguments(LocationInput):
    """A past window, written the way the question was asked.

    "Yesterday" is a calendar date, and belongs to the location's own day;
    "six hours ago" is an instant, and belongs to the clock. Carrying both
    forms rather than flattening one into the other is what keeps an hourly
    question from being answered with a daily average.
    """

    start: date | None = Field(default=None, description="Inclusive ISO-8601 start date.")
    end: date | None = Field(default=None, description="Inclusive ISO-8601 end date.")
    start_time: datetime | None = Field(
        default=None, description="Inclusive start instant, for an hour-precision question."
    )
    end_time: datetime | None = Field(
        default=None, description="Inclusive end instant, for an hour-precision question."
    )
    hour_from: int | None = Field(
        default=None, ge=0, le=23, description="Keep only observations from this local hour."
    )
    hour_to: int | None = Field(
        default=None, ge=1, le=24, description="Keep only observations before this local hour."
    )
    radius_km: float | None = Field(default=None, gt=0.0, le=500.0)
    provider: str | None = Field(default=None, max_length=100)

    @model_validator(mode="after")
    def _validate_window(self) -> "HistoricalWeatherArguments":
        has_dates = self.start is not None and self.end is not None
        has_times = self.start_time is not None and self.end_time is not None
        if not (has_dates or has_times):
            raise ValueError("provide either a date range or a start and end instant")
        if has_dates and self.end < self.start:
            raise ValueError("end must not precede start")
        if has_times and self.end_time < self.start_time:
            raise ValueError("end_time must not precede start_time")
        if (self.hour_from is None) != (self.hour_to is None):
            raise ValueError("hour_from and hour_to must be supplied together")
        return self


class AlertsArguments(LocationInput):
    radius_km: float | None = Field(default=None, gt=0.0, le=500.0)
    provider: str | None = Field(default=None, max_length=100)


class LocationRiskArguments(ForecastArguments):
    purpose: AdvisoryPurpose = AdvisoryPurpose.GENERAL
    #: The window the considerations are read over. It is the same window the
    #: verdict reads, passed rather than assumed: the two are shown one above
    #: the other, and a body reporting one hour beside a verdict reporting a
    #: day reads as a contradiction even when both are true.
    window_hours: int = Field(default=24, ge=1, le=168)


class LocationSearchArguments(ToolInputModel):
    query: str = Field(min_length=1, max_length=200)
    limit: int = Field(default=5, ge=1, le=20)


ToolArguments = (
    CurrentWeatherArguments
    | ForecastArguments
    | HistoricalWeatherArguments
    | AlertsArguments
    | LocationRiskArguments
    | LocationSearchArguments
)


class SourceReference(BaseModel):
    """Provenance surfaced alongside an AI answer rather than hidden in prose."""

    tool: str
    provider_id: str
    provider_name: str
    model: str | None = None
    fetched_at: datetime
    model_run_at: datetime | None = None
    source_url: str | None = None
    license: str | None = None
    attribution: str | None = None
    cached: bool = False

    @classmethod
    def from_provenance(cls, tool: str, provenance: DataProvenance) -> "SourceReference":
        return cls(tool=tool, **provenance.model_dump())


class ToolResult(BaseModel):
    """A serialisable, validated result of one backend-only tool invocation."""

    tool: str
    data: dict[str, Any]
    sources: list[SourceReference] = Field(default_factory=list)


class RiskConsideration(BaseModel):
    """A bounded advisory tied to a named backend field, not a new alert."""

    field: str
    statement: str
    valid_at: datetime | date | None = None


class LocationRiskResult(BaseModel):
    """Read-only composition of current, forecast and existing alert data."""

    purpose: AdvisoryPurpose
    location: Location
    considerations: list[RiskConsideration]
    #: ``None`` when the alert store could not be read for this request. It is
    #: deliberately not ``0``: "no alerts stand" and "nobody could look" are
    #: different answers, and only one of them was actually established.
    active_alert_count: int | None = None
    alert_disclaimer: str | None = None
    current: dict[str, Any]
    forecast: dict[str, Any]
    alerts: dict[str, Any] | None = None


class AlertConversationContext(BaseModel):
    """The alert a person clicked, carried into the existing conversation."""

    id: str
    alert_type: AlertType
    severity: AlertSeverity
    status: AlertStatus
    source_type: AlertSourceType
    kind: AlertKind
    rule_id: str
    title: str
    description: str
    latitude: float
    longitude: float
    location_name: str | None = None
    admin1: str | None = None
    country: str | None = None
    timezone: str | None = None
    triggered_at: datetime
    valid_from: datetime
    valid_until: datetime
    resolved_at: datetime | None = None
    evidence: AlertEvidence
    provenance: DataProvenance


class ChatRequest(BaseModel):
    """Input to the AI endpoint.

    ``session_id`` is optional: clients that retain it receive location and
    intent context on follow-up questions.  It is an opaque identifier; no
    authentication or durable transcript storage is implied by this layer.
    """

    message: str = Field(min_length=1, max_length=4_000)
    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    alert_context: AlertConversationContext | None = Field(
        default=None, description="The exact alert selected from the Alerts panel."
    )
    language: str | None = Field(
        default=None,
        min_length=2,
        max_length=16,
        description="BCP-47 language hint, for example en or hi-IN.",
    )


class AnswerVerdict(BaseModel):
    """A one-line reading of the data in the answer beside it.

    Computed from values the backend returned in this turn. It interprets those
    values; it never adds one, and it never issues a warning — official warnings
    belong to meteorological services and to the alert engine.
    """

    text: str = Field(description="The reading, in the language of the answer.")
    icon: str = Field(default="", description="A single glyph summarising it.")
    caveat: str | None = Field(
        default=None,
        description=(
            "Present when the question asked for advice outside what weather "
            "data can settle, such as whether to re-apply a pesticide."
        ),
    )


class ChatResponse(BaseModel):
    """Grounded conversational response with machine-readable evidence."""

    session_id: str
    language: str = Field(description="Requested/detected response language tag.")
    language_fallback: bool = Field(
        default=False,
        description="True when a catalog for the requested language is unavailable.",
    )
    intent: Intent
    answer: str
    needs_clarification: bool = False
    tool_results: list[ToolResult] = Field(default_factory=list)
    sources: list[SourceReference] = Field(default_factory=list)
    safety_note: str | None = None
    verdict: AnswerVerdict | None = Field(
        default=None,
        description=(
            "A one-line reading of the retrieved data. Absent for a "
            "clarification, a location search, or any result whose fields do "
            "not support one."
        ),
    )


class LLMToolCall(BaseModel):
    """Normalized function call emitted by an LLM provider."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class LLMToolSelection(BaseModel):
    """Only tool selection is accepted from an LLM; prose is intentionally absent."""

    calls: list[LLMToolCall] = Field(default_factory=list)


class BackendError(BaseModel):
    """Safe subset of WeatherGPT's error envelope used in user-facing failures."""

    code: str = "BACKEND_UNAVAILABLE"
    message: str = "Weather backend data is temporarily unavailable."
    request_id: str | None = None


ToolName = Literal[
    "current_weather",
    "hourly_forecast",
    "daily_forecast",
    "historical_weather",
    "alerts",
    "location_risk",
    "location_search",
]
