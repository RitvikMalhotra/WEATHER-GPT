"""The canonical alert model.

An alert is a *deterministic* statement: a named rule compared a validated
measurement against a configured threshold and the comparison held. Nothing in
this system infers an alert from prose, and no language model participates in
deciding that one exists. The AI layer arriving in a later phase explains an
alert that already exists; it never creates one.

Three distinctions are carried explicitly in the model because collapsing any of
them would be a safety failure:

``source_type``
    :attr:`AlertSourceType.DETERMINISTIC_RULE` — WeatherGPT applied its own
    threshold — versus :attr:`AlertSourceType.OFFICIAL_WARNING`, which may only
    be used for a warning actually issued by a meteorological authority. This
    build never produces the latter.

``kind``
    :attr:`AlertKind.OBSERVED` — the threshold was crossed in data describing
    conditions that occurred — versus :attr:`AlertKind.FORECAST_RISK`, where a
    *prediction* crosses it. "It is raining heavily" and "heavy rain is
    forecast" are different claims.

``severity``
    WeatherGPT's own ladder, not any authority's colour code.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.domain.location import Location
from app.domain.provenance import DataProvenance


class AlertSeverity(str, Enum):
    """WeatherGPT's severity ladder.

    These are *this system's* labels. They are deliberately not named after any
    meteorological agency's warning colours, because matching a label would
    imply an authority this system does not have.
    """

    INFO = "info"
    WATCH = "watch"
    WARNING = "warning"
    SEVERE = "severe"
    EXTREME = "extreme"

    @property
    def rank(self) -> int:
        """Position on the ladder, for ordering and comparison."""
        return _SEVERITY_RANK[self]


_SEVERITY_RANK: dict[AlertSeverity, int] = {
    AlertSeverity.INFO: 0,
    AlertSeverity.WATCH: 1,
    AlertSeverity.WARNING: 2,
    AlertSeverity.SEVERE: 3,
    AlertSeverity.EXTREME: 4,
}


class AlertType(str, Enum):
    """The hazard an alert is about."""

    HEAVY_RAINFALL = "heavy_rainfall"
    EXTREME_HEAT = "extreme_heat"
    HIGH_WIND = "high_wind"
    SEVERE_PRECIPITATION_PROBABILITY = "severe_precipitation_probability"


class AlertStatus(str, Enum):
    """Lifecycle state.

    Records are never deleted on leaving ``ACTIVE``: the history of what fired,
    and what turned out not to matter, is the raw material for false-alarm
    analysis and any future evaluation of the rules.
    """

    ACTIVE = "active"
    #: Validity window elapsed without the condition being seen again.
    EXPIRED = "expired"
    #: Re-evaluated at the same place and the condition no longer held.
    RESOLVED = "resolved"


class AlertSourceType(str, Enum):
    """Who decided this alert exists.

    The distinction is not cosmetic. "Rainfall exceeded WeatherGPT's threshold"
    and "the meteorological service issued a warning" are different claims with
    different authority, and one may never be presented as the other.
    """

    DETERMINISTIC_RULE = "deterministic_rule"
    OFFICIAL_WARNING = "official_warning"


class AlertKind(str, Enum):
    """Whether the threshold was crossed in observed or predicted data."""

    OBSERVED = "observed"
    FORECAST_RISK = "forecast_risk"


class AlertEvidence(BaseModel):
    """Why this alert exists, in machine-checkable terms.

    Every alert carries one. An alert that says "severe weather detected"
    without naming the variable, the value and the threshold cannot be audited,
    debugged, or honestly explained to a user.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "rule_id": "HIGH_WIND_01",
                "variable": "wind_speed_ms",
                "observed_value": 24.7,
                "threshold": 20.8,
                "unit": "m/s",
                "comparison": ">=",
                "sample_window": "hour",
            }
        }
    )

    rule_id: str = Field(description="Identifier of the rule that fired.")
    variable: str = Field(
        description="Canonical field that was compared.", examples=["wind_speed_ms"]
    )
    observed_value: float = Field(description="The value the rule read.")
    threshold: float = Field(description="The band threshold that was met.")
    unit: str = Field(description="Unit of both value and threshold.", examples=["m/s"])
    comparison: str = Field(
        default=">=", description="Comparison applied between value and threshold."
    )
    sample_window: str = Field(
        description="Accumulation window the value describes.",
        examples=["observation", "hour", "day"],
    )
    context: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional supporting values — other variables read at the same "
            "point. Provider-specific detail lives here rather than in columns."
        ),
    )


class Alert(BaseModel):
    """A triggered rule, with everything needed to audit and explain it."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "alert_type": "high_wind",
                "severity": "warning",
                "status": "active",
                "source_type": "deterministic_rule",
                "kind": "forecast_risk",
                "rule_id": "HIGH_WIND_01",
                "title": "High wind",
                "description": (
                    "Forecast wind speed of 24.7 m/s meets the WeatherGPT "
                    "warning threshold of 20.8 m/s."
                ),
                "triggered_at": "2026-09-04T08:00:00Z",
                "valid_from": "2026-09-04T09:00:00Z",
                "valid_until": "2026-09-04T10:00:00Z",
            }
        }
    )

    id: UUID | None = Field(
        default=None, description="Assigned when the alert is persisted."
    )

    alert_type: AlertType = Field(description="The hazard this alert is about.")
    severity: AlertSeverity = Field(description="WeatherGPT's severity band.")
    status: AlertStatus = Field(
        default=AlertStatus.ACTIVE, description="Lifecycle state."
    )
    source_type: AlertSourceType = Field(
        default=AlertSourceType.DETERMINISTIC_RULE,
        description="Whether a WeatherGPT rule or an official authority issued this.",
    )
    kind: AlertKind = Field(
        description="Whether the threshold was crossed in observed or predicted data."
    )

    rule_id: str = Field(description="Identifier of the rule that fired.")
    title: str = Field(description="Short hazard label.")
    description: str = Field(
        description=(
            "Factual sentence naming the value, the threshold and whether it is "
            "observed or forecast. Not an interpretation."
        )
    )

    location: Location = Field(description="Where the alert applies.")

    triggered_at: datetime = Field(
        description="When the evaluation that produced this alert ran."
    )
    valid_from: datetime = Field(
        description="Start of the period the alert describes."
    )
    valid_until: datetime = Field(
        description="End of the period the alert describes."
    )
    resolved_at: datetime | None = Field(
        default=None, description="When the condition was observed to have ended."
    )

    evidence: AlertEvidence = Field(description="Why this alert exists.")
    provenance: DataProvenance = Field(
        description="Which source and model produced the data that triggered it."
    )

    created_at: datetime | None = Field(
        default=None, description="When the record was first written."
    )
    updated_at: datetime | None = Field(
        default=None, description="When the record was last refreshed."
    )

    @property
    def is_official(self) -> bool:
        """True only for warnings issued by an actual authority."""
        return self.source_type is AlertSourceType.OFFICIAL_WARNING


__all__ = [
    "Alert",
    "AlertEvidence",
    "AlertKind",
    "AlertSeverity",
    "AlertSourceType",
    "AlertStatus",
    "AlertType",
]
