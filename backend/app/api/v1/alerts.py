"""Alert retrieval endpoint.

Serves alerts the deterministic engine has already produced. Nothing is
evaluated here — a GET does not create alerts, and the response is a set of
structured schemas rather than database rows.

Every response is explicit about what it is showing: a WeatherGPT rule result,
observed or forecast, at a severity on WeatherGPT's own ladder. None of those
three facts is safe to leave implicit.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Query, status
from pydantic import BaseModel, ConfigDict, Field

from app.api.v1.params import (
    WEATHER_ERROR_RESPONSES,
    InvalidLocationQueryError,
    ProviderQuery,
)
from app.core.dependencies import AlertServiceDep, SettingsDep
from app.db.repositories import AlertFilter
from app.domain.alert import (
    AlertEvidence,
    AlertKind,
    AlertSeverity,
    AlertSourceType,
    AlertStatus,
    AlertType,
)
from app.domain.provenance import DataProvenance
from app.services.alerts import AlertMatch

router = APIRouter(prefix="/weather", tags=["Alerts"])


class AlertSummary(BaseModel):
    """One alert, with the evidence that produced it."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "5f2c1e90-8a4b-4c2e-9d3f-1b7a6e5c4d3a",
                "alert_type": "high_wind",
                "severity": "warning",
                "status": "active",
                "source_type": "deterministic_rule",
                "kind": "forecast_risk",
                "rule_id": "HIGH_WIND_01",
                "title": "High wind",
                "description": (
                    "Forecast wind speed over one hour of 24.7 m/s meets the "
                    "WeatherGPT warning threshold of 20.8 m/s."
                ),
                "latitude": 17.385,
                "longitude": 78.4867,
                "distance_km": 3.5,
                "triggered_at": "2026-09-04T08:00:00Z",
                "valid_from": "2026-09-04T09:00:00Z",
                "valid_until": "2026-09-04T10:00:00Z",
                "evidence": {
                    "rule_id": "HIGH_WIND_01",
                    "variable": "wind_speed_ms",
                    "observed_value": 24.7,
                    "threshold": 20.8,
                    "unit": "m/s",
                    "comparison": ">=",
                    "sample_window": "hour",
                },
            }
        }
    )

    id: str = Field(description="Stable identifier of this alert.")
    alert_type: AlertType = Field(description="The hazard.")
    severity: AlertSeverity = Field(
        description=(
            "WeatherGPT's severity band. Not an official warning category — see "
            "`source_type`."
        )
    )
    status: AlertStatus = Field(description="Lifecycle state.")
    source_type: AlertSourceType = Field(
        description=(
            "`deterministic_rule` means WeatherGPT applied its own threshold. "
            "`official_warning` would mean a meteorological authority issued it; "
            "this build never produces that value."
        )
    )
    kind: AlertKind = Field(
        description=(
            "`observed` means the threshold was crossed in data describing "
            "conditions that occurred. `forecast_risk` means a prediction "
            "crosses it — the event has not happened."
        )
    )

    rule_id: str = Field(description="Which rule fired.")
    title: str = Field(description="Short hazard label.")
    description: str = Field(
        description="Factual statement of value, threshold and certainty."
    )

    latitude: float = Field(description="Latitude the alert applies to.")
    longitude: float = Field(description="Longitude the alert applies to.")
    distance_km: float = Field(
        description="Distance from the requested point; 0 for a non-spatial query."
    )
    timezone: str | None = Field(default=None, description="IANA timezone.")

    triggered_at: datetime = Field(description="When the evaluation ran.")
    valid_from: datetime = Field(description="Start of the period described.")
    valid_until: datetime = Field(description="End of the period described.")
    resolved_at: datetime | None = Field(
        default=None, description="When the condition was seen to have lifted."
    )

    evidence: AlertEvidence = Field(description="Why this alert exists.")
    provenance: DataProvenance = Field(
        description="Which source and model produced the triggering data."
    )

    @classmethod
    def from_match(cls, match: AlertMatch) -> "AlertSummary":
        alert = match.alert
        return cls(
            id=str(alert.id),
            alert_type=alert.alert_type,
            severity=alert.severity,
            status=alert.status,
            source_type=alert.source_type,
            kind=alert.kind,
            rule_id=alert.rule_id,
            title=alert.title,
            description=alert.description,
            latitude=alert.location.coordinates.latitude,
            longitude=alert.location.coordinates.longitude,
            distance_km=round(match.distance_m / 1000.0, 3),
            timezone=alert.location.timezone,
            triggered_at=alert.triggered_at,
            valid_from=alert.valid_from,
            valid_until=alert.valid_until,
            resolved_at=alert.resolved_at,
            evidence=alert.evidence,
            provenance=alert.provenance,
        )


class AlertListResponse(BaseModel):
    """Alerts matching a query."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "requested": {"latitude": 17.385, "longitude": 78.4867},
                "search_radius_km": 25.0,
                "count": 1,
                "truncated": False,
                "disclaimer": (
                    "WeatherGPT alerts are produced by configurable thresholds "
                    "applied to validated provider data. They are not official "
                    "meteorological warnings."
                ),
                "alerts": [],
            }
        }
    )

    requested: dict[str, float] | None = Field(
        default=None, description="Coordinates queried, when the search was spatial."
    )
    search_radius_km: float | None = Field(
        default=None, description="Radius searched, when the search was spatial."
    )
    count: int = Field(description="Number of alerts returned.")
    truncated: bool = Field(
        description="True when more alerts matched than the result limit allows."
    )
    disclaimer: str = Field(
        description="Standing statement of what these alerts are and are not."
    )
    alerts: list[AlertSummary] = Field(
        description="Matching alerts, most severe first."
    )


#: Returned on every alert response. The distinction between a threshold this
#: system applied and a warning an authority issued must never be left to the
#: reader to infer.
DISCLAIMER = (
    "WeatherGPT alerts are produced by configurable thresholds applied to "
    "validated provider data. They are not official meteorological warnings and "
    "carry no authority. Check your national meteorological service for "
    "official warnings."
)


@router.get(
    "/alerts",
    response_model=AlertListResponse,
    status_code=status.HTTP_200_OK,
    summary="Weather alerts",
    description=(
        "Alerts produced by WeatherGPT's deterministic rule engine.\n\n"
        "**These are not official warnings.** Each alert records which rule "
        "fired, the value it read, the threshold it met and the source that "
        "supplied the data — see `evidence` and `provenance`. `source_type` is "
        "always `deterministic_rule`; `official_warning` is reserved for "
        "ingestion of an actual authority's feed and is never produced here.\n\n"
        "`kind` separates `observed` from `forecast_risk`. A forecast risk means "
        "a *prediction* crosses a threshold — not that the event has occurred.\n\n"
        "Pass `latitude` and `longitude` to search spatially; the radius is "
        "evaluated in PostGIS against a geographic index. Without coordinates "
        "the filters still apply, unbounded by location.\n\n"
        "Alerts are evaluated when weather is fetched, not when this endpoint is "
        "called: a read never creates an alert."
    ),
    response_description="Matching alerts with evidence and provenance.",
    responses=WEATHER_ERROR_RESPONSES,
)
async def list_alerts(
    service: AlertServiceDep,
    settings: SettingsDep,
    latitude: Annotated[
        float | None,
        Query(ge=-90.0, le=90.0, description="Latitude. Requires `longitude`."),
    ] = None,
    longitude: Annotated[
        float | None,
        Query(ge=-180.0, le=180.0, description="Longitude. Requires `latitude`."),
    ] = None,
    radius_km: Annotated[
        float | None, Query(gt=0.0, description="Search radius in kilometres.")
    ] = None,
    severity: Annotated[
        list[AlertSeverity] | None,
        Query(description="Restrict to these severities. Repeatable."),
    ] = None,
    alert_type: Annotated[
        list[AlertType] | None,
        Query(description="Restrict to these hazards. Repeatable."),
    ] = None,
    alert_status: Annotated[
        list[AlertStatus] | None,
        Query(
            alias="status",
            description="Lifecycle states to include. Defaults to active only.",
        ),
    ] = None,
    kind: Annotated[
        list[AlertKind] | None,
        Query(description="Restrict to observed or forecast risk. Repeatable."),
    ] = None,
    rule_id: Annotated[
        str | None, Query(description="Restrict to one rule.")
    ] = None,
    provider: ProviderQuery = None,
    limit: Annotated[
        int | None, Query(ge=1, le=500, description="Maximum alerts to return.")
    ] = None,
) -> AlertListResponse:
    spatial = latitude is not None and longitude is not None
    if (latitude is None) != (longitude is None):
        raise InvalidLocationQueryError(
            "'latitude' and 'longitude' must be supplied together.",
            details={"latitude": latitude, "longitude": longitude},
        )

    radius = (
        min(radius_km or settings.ALERT_DEFAULT_RADIUS_KM, settings.ALERT_MAX_RADIUS_KM)
        if spatial
        else None
    )
    capped = min(limit or settings.ALERT_MAX_RESULTS, settings.ALERT_MAX_RESULTS)

    matches = await service.search(
        AlertFilter(
            latitude=latitude,
            longitude=longitude,
            radius_m=radius * 1000.0 if radius is not None else None,
            severities=_values(severity),
            alert_types=_values(alert_type),
            # Defaulting to active alone keeps the common question — "what is
            # happening near me now" — from returning months of history.
            statuses=_values(alert_status) or (AlertStatus.ACTIVE.value,),
            kinds=_values(kind),
            provider_id=provider,
            rule_id=rule_id,
            limit=capped,
        )
    )

    return AlertListResponse(
        requested=(
            {"latitude": latitude, "longitude": longitude} if spatial else None
        ),
        search_radius_km=radius,
        count=len(matches),
        truncated=len(matches) >= capped,
        disclaimer=DISCLAIMER,
        alerts=[AlertSummary.from_match(match) for match in matches],
    )


def _values(selected: list[Any] | None) -> tuple[str, ...]:
    """Enum members to their stored string values."""
    return tuple(item.value for item in selected) if selected else ()
