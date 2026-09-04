"""Watched locations: the standing instruction behind the Alerts panel.

These endpoints manage *what to watch*. They never produce an alert — every
alert returned here was written to ``weather_alerts`` by the deterministic
engine on the ordinary weather path, and is read back unchanged.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Header, Path, status
from pydantic import BaseModel, ConfigDict, Field

from app.api.v1.alerts import AlertSummary
from app.api.v1.params import WEATHER_ERROR_RESPONSES
from app.core.dependencies import SubscriptionServiceDep
from app.core.exceptions import WeatherGPTError
from app.domain.location import Coordinates, Location
from app.services.subscriptions import (
    WATCH_RADIUS_KM,
    Watch,
    WatchStatus,
    available_alert_types,
)

router = APIRouter(prefix="/alerts/subscriptions", tags=["Alerts"])


class SubscriptionNotFoundError(WeatherGPTError):
    """No watch with that id belongs to this client."""

    code = "SUBSCRIPTION_NOT_FOUND"
    status_code = 404
    message = "No watched location with that id."


#: Identifies the browser holding a set of watches. Not an account and not a
#: credential: it authorises nothing, carries no personal data, and exists so a
#: demo needs no sign-in. A deployment with real users replaces this header
#: with its own authenticated subject without changing anything below.
ClientKey = Annotated[
    str,
    Header(
        alias="X-WeatherGPT-Client",
        min_length=8,
        max_length=64,
        description="Opaque identifier for the browser holding these watches.",
    ),
]


class WatchLocation(BaseModel):
    """The place a watch was created for, as the gazetteer resolved it."""

    name: str = Field(description="Human-facing label, e.g. 'Miyapur, Hyderabad'.")
    latitude: float
    longitude: float
    admin1: str | None = None
    country: str | None = None
    timezone: str | None = None

    @classmethod
    def of(cls, location: Location) -> "WatchLocation":
        return cls(
            # The bare name. `admin1` and `country` travel in their own fields,
            # and a client that shows both would otherwise read
            # "Hyderabad, Telangana, India — Telangana, India".
            name=location.name or location.display_name,
            latitude=location.coordinates.latitude,
            longitude=location.coordinates.longitude,
            admin1=location.admin1,
            country=location.country,
            timezone=location.timezone,
        )


class CreateSubscriptionRequest(BaseModel):
    """A place to start watching.

    Coordinates are preferred and are what a confirmed pick from the location
    search sends. A bare ``query`` is accepted for convenience and is resolved
    here; when the name is ambiguous the request is refused with the candidates
    rather than resolved to a guess.
    """

    model_config = ConfigDict(extra="forbid")

    query: str | None = Field(default=None, min_length=1, max_length=200)
    latitude: float | None = Field(default=None, ge=-90.0, le=90.0)
    longitude: float | None = Field(default=None, ge=-180.0, le=180.0)
    name: str | None = Field(default=None, max_length=200)
    admin1: str | None = Field(default=None, max_length=128)
    country: str | None = Field(default=None, max_length=128)
    timezone: str | None = Field(default=None, max_length=64)
    alert_types: list[str] = Field(
        default_factory=list,
        description=(
            "Alert types to surface. Empty means every rule the engine runs, "
            "which is the default that cannot omit a hazard nobody ticked."
        ),
    )


class SubscriptionResponse(BaseModel):
    """A watched location and the alerts currently standing against it."""

    id: str
    location: WatchLocation
    alert_types: list[str]
    enabled: bool
    created_at: datetime
    last_evaluated_at: datetime | None = Field(
        default=None,
        description=(
            "When the pipeline last ran for this point. Null means it has not "
            "been checked yet, which is not the same as nothing being found."
        ),
    )
    evaluated: bool = Field(
        description="False until the first evaluation has completed."
    )
    search_radius_km: float = WATCH_RADIUS_KM
    alerts: list[AlertSummary] = Field(default_factory=list)

    @classmethod
    def of(cls, watch: Watch, alerts=(), evaluated: bool | None = None) -> "SubscriptionResponse":
        return cls(
            id=str(watch.id),
            location=WatchLocation.of(watch.location),
            alert_types=list(watch.alert_types),
            enabled=watch.enabled,
            created_at=watch.created_at,
            last_evaluated_at=watch.last_evaluated_at,
            evaluated=(
                watch.last_evaluated_at is not None if evaluated is None else evaluated
            ),
            alerts=[AlertSummary.from_match(match) for match in alerts],
        )

    @classmethod
    def of_status(cls, status_: WatchStatus) -> "SubscriptionResponse":
        return cls.of(status_.watch, status_.alerts, status_.evaluated)


class SubscriptionListResponse(BaseModel):
    """Everything one client is watching."""

    count: int
    available_alert_types: list[str] = Field(
        description=(
            "The rule families the deterministic engine runs, read from the "
            "domain enum so the panel cannot advertise a rule that does not exist."
        )
    )
    subscriptions: list[SubscriptionResponse]


@router.get(
    "",
    response_model=SubscriptionListResponse,
    status_code=status.HTTP_200_OK,
    summary="List watched locations",
    description=(
        "Every location this client is watching, each with the alerts the "
        "deterministic engine currently holds for it.\n\n"
        "Alerts are read from the alert store; nothing is evaluated by this "
        "call. A watch showing no alerts and `evaluated: true` genuinely has "
        "none standing."
    ),
    responses=WEATHER_ERROR_RESPONSES,
)
async def list_subscriptions(
    service: SubscriptionServiceDep, client: ClientKey
) -> SubscriptionListResponse:
    statuses = await service.status_for(client)
    return SubscriptionListResponse(
        count=len(statuses),
        available_alert_types=available_alert_types(),
        subscriptions=[SubscriptionResponse.of_status(item) for item in statuses],
    )


@router.post(
    "",
    response_model=SubscriptionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Watch a location",
    description=(
        "Start watching a place. Send coordinates from a confirmed search "
        "result, or a `query` to be resolved here.\n\n"
        "An ambiguous name is refused with `409 AMBIGUOUS_LOCATION` and the "
        "candidates attached, so the caller can ask which was meant instead of "
        "watching the wrong town.\n\n"
        "The first evaluation runs immediately, so the panel shows a real "
        "answer rather than an empty state that looks like one."
    ),
    responses=WEATHER_ERROR_RESPONSES,
)
async def create_subscription(
    payload: CreateSubscriptionRequest,
    service: SubscriptionServiceDep,
    client: ClientKey,
) -> SubscriptionResponse:
    if payload.latitude is not None and payload.longitude is not None:
        location = Location(
            coordinates=Coordinates(
                latitude=payload.latitude, longitude=payload.longitude
            ),
            name=payload.name,
            admin1=payload.admin1,
            country=payload.country,
            timezone=payload.timezone,
        )
    elif payload.query:
        location = await service.resolve(payload.query)
    else:
        raise SubscriptionNotFoundError(
            "Provide either coordinates or a location query.",
        )

    watch = await service.create(
        owner_key=client,
        location=location,
        alert_types=tuple(payload.alert_types),
    )
    # Evaluate once now. Waiting for the next sweep would show "no alerts"
    # before anything had actually looked, which reads as an all-clear.
    alerts = await service.evaluate(watch)
    return SubscriptionResponse.of(watch, alerts, evaluated=True)


@router.post(
    "/{subscription_id}/refresh",
    response_model=SubscriptionResponse,
    status_code=status.HTTP_200_OK,
    summary="Re-evaluate a watched location now",
    description=(
        "Runs the ordinary weather pipeline for this point, which is what "
        "causes the deterministic engine to evaluate it, then returns what "
        "stands. Used by the panel's manual refresh; the monitor does the same "
        "thing on a schedule."
    ),
    responses=WEATHER_ERROR_RESPONSES,
)
async def refresh_subscription(
    service: SubscriptionServiceDep,
    client: ClientKey,
    subscription_id: Annotated[uuid.UUID, Path(description="The watch to refresh.")],
) -> SubscriptionResponse:
    watch = await _find(service, client, subscription_id)
    alerts = await service.evaluate(watch)
    return SubscriptionResponse.of(watch, alerts, evaluated=True)


class ToggleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool


@router.patch(
    "/{subscription_id}",
    response_model=SubscriptionResponse,
    status_code=status.HTTP_200_OK,
    summary="Pause or resume a watched location",
    responses=WEATHER_ERROR_RESPONSES,
)
async def toggle_subscription(
    payload: ToggleRequest,
    service: SubscriptionServiceDep,
    client: ClientKey,
    subscription_id: Annotated[uuid.UUID, Path(description="The watch to change.")],
) -> SubscriptionResponse:
    watch = await service.set_enabled(client, subscription_id, payload.enabled)
    if watch is None:
        raise SubscriptionNotFoundError(details={"id": str(subscription_id)})
    alerts = await service.alerts_for(watch) if watch.enabled else []
    return SubscriptionResponse.of(watch, alerts)


@router.delete(
    "/{subscription_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Stop watching a location",
    description=(
        "Removes the watch. Alerts it produced stay in the alert store: they "
        "are the record of what fired, and deleting them would erase history."
    ),
    responses=WEATHER_ERROR_RESPONSES,
)
async def delete_subscription(
    service: SubscriptionServiceDep,
    client: ClientKey,
    subscription_id: Annotated[uuid.UUID, Path(description="The watch to remove.")],
) -> None:
    if not await service.delete(client, subscription_id):
        raise SubscriptionNotFoundError(details={"id": str(subscription_id)})


async def _find(service, client: str, subscription_id: uuid.UUID) -> Watch:
    for watch in await service.list_for(client):
        if watch.id == subscription_id:
            return watch
    raise SubscriptionNotFoundError(details={"id": str(subscription_id)})
