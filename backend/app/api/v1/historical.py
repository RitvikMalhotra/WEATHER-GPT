"""Historical weather endpoint.

Serves what the system has stored, which is the one thing a provider cannot
re-fetch: the past as it was recorded, with the provenance of each record.

Records are returned as response schemas built from the canonical model, never
as database rows. Nothing here leaks a column name, a row id or a driver detail.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Annotated

from fastapi import APIRouter, Query, status
from pydantic import BaseModel, ConfigDict, Field

from app.api.v1.params import WEATHER_ERROR_RESPONSES, ProviderQuery
from app.core.dependencies import HistoryServiceDep, SettingsDep
from app.core.exceptions import WeatherGPTError
from app.domain.provenance import DataProvenance
from app.domain.weather import CurrentWeather
from app.services.history import HistoricalRecord, HistoryQuery

router = APIRouter(prefix="/weather", tags=["Weather"])


class InvalidTimeRangeError(WeatherGPTError):
    """The requested window is unparseable, inverted or too wide."""

    code = "INVALID_TIME_RANGE"
    status_code = 422
    message = "The requested time range is not valid."


class HistoricalObservation(BaseModel):
    """One stored observation, with where it came from and how far away it is."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "latitude": 17.385,
                "longitude": 78.4867,
                "distance_km": 0.0,
                "weather": {
                    "observed_at": "2026-08-15T06:00:00Z",
                    "temperature_c": 28.4,
                    "condition": "rain",
                },
                "provenance": {
                    "provider_id": "open-meteo",
                    "provider_name": "Open-Meteo",
                    "fetched_at": "2026-08-15T06:02:00Z",
                },
            }
        }
    )

    latitude: float = Field(description="Latitude the observation was recorded at.")
    longitude: float = Field(description="Longitude the observation was recorded at.")
    distance_km: float = Field(
        description="Great-circle distance from the requested point, in kilometres."
    )
    weather: CurrentWeather = Field(description="The canonical measurement record.")
    provenance: DataProvenance = Field(
        description="Source, model and fetch time behind this record."
    )

    @classmethod
    def from_record(cls, record: HistoricalRecord) -> "HistoricalObservation":
        return cls(
            latitude=record.latitude,
            longitude=record.longitude,
            distance_km=round(record.distance_m / 1000.0, 3),
            weather=record.weather,
            provenance=record.provenance,
        )


class TimeRange(BaseModel):
    """The window that was searched, as the server interpreted it."""

    start: datetime = Field(description="Inclusive start of the window (UTC).")
    end: datetime = Field(description="Inclusive end of the window (UTC).")


class HistoricalWeatherResponse(BaseModel):
    """Stored observations near a point over a time window."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "requested": {"latitude": 17.385, "longitude": 78.4867},
                "range": {
                    "start": "2026-08-01T00:00:00Z",
                    "end": "2026-08-31T23:59:59Z",
                },
                "search_radius_km": 25.0,
                "count": 1,
                "truncated": False,
                "observations": [],
            }
        }
    )

    requested: dict[str, float] = Field(
        description="The coordinates the caller asked about."
    )
    range: TimeRange = Field(description="The window that was searched.")
    search_radius_km: float = Field(
        description="Radius searched around the requested point."
    )
    count: int = Field(description="Number of observations returned.")
    truncated: bool = Field(
        description="True when more records matched than the result limit allows."
    )
    observations: list[HistoricalObservation] = Field(
        description="Matching records, oldest first."
    )


def _parse_boundary(raw: str, *, field: str, end_of_day: bool) -> datetime:
    """Parse a date or timestamp into an aware UTC instant.

    A bare date is expanded to cover the whole day, so ``end=2026-08-31``
    includes that day's observations rather than stopping at its midnight — the
    reading a caller writing a date almost always intends.
    """
    text = raw.strip()

    # A bare date must be detected *before* parsing: datetime.fromisoformat
    # happily accepts "2026-08-31" and returns midnight, which would silently
    # drop that day's observations from an inclusive range.
    if "T" not in text and " " not in text:
        try:
            day = date.fromisoformat(text)
        except ValueError:
            raise InvalidTimeRangeError(
                f"{field!r} must be an ISO-8601 date or timestamp, got {raw!r}.",
                details={"field": field, "value": raw},
            ) from None
        return datetime.combine(
            day, time.max if end_of_day else time.min, tzinfo=timezone.utc
        )

    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        raise InvalidTimeRangeError(
            f"{field!r} must be an ISO-8601 date or timestamp, got {raw!r}.",
            details={"field": field, "value": raw},
        ) from None

    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


@router.get(
    "/historical",
    response_model=HistoricalWeatherResponse,
    status_code=status.HTTP_200_OK,
    summary="Historical observations",
    description=(
        "Observations this deployment has stored for a point and time window.\n\n"
        "Only data that passed validation on the way in is stored, so everything "
        "returned here met the same standard as a live response, and each record "
        "carries the source and model that produced it.\n\n"
        "The search is spatial: `radius_km` around the requested point, evaluated "
        "in PostGIS against a geographic index. Coordinates are matched by "
        "proximity rather than equality because a provider serves the nearest "
        "grid point, which shifts slightly between requests.\n\n"
        "`start` and `end` accept a date (`2026-08-01`) or a timestamp "
        "(`2026-08-01T06:00:00Z`). A bare `end` date covers the whole day. Both "
        "bounds are inclusive and interpreted as UTC when no offset is given.\n\n"
        "An empty result is a 200 with `count: 0` — 'nothing recorded there' is a "
        "valid answer, not an error."
    ),
    response_description="Stored observations with provenance.",
    responses=WEATHER_ERROR_RESPONSES,
)
async def historical_weather(
    service: HistoryServiceDep,
    settings: SettingsDep,
    latitude: Annotated[
        float, Query(ge=-90.0, le=90.0, description="Latitude in decimal degrees.")
    ],
    longitude: Annotated[
        float, Query(ge=-180.0, le=180.0, description="Longitude in decimal degrees.")
    ],
    start: Annotated[
        str, Query(description="Inclusive start, as a date or timestamp.")
    ],
    end: Annotated[str, Query(description="Inclusive end, as a date or timestamp.")],
    radius_km: Annotated[
        float | None,
        Query(gt=0.0, description="Search radius in kilometres."),
    ] = None,
    provider: ProviderQuery = None,
) -> HistoricalWeatherResponse:
    window_start = _parse_boundary(start, field="start", end_of_day=False)
    window_end = _parse_boundary(end, field="end", end_of_day=True)

    if window_end < window_start:
        raise InvalidTimeRangeError(
            "'end' must not precede 'start'.",
            details={"start": window_start.isoformat(), "end": window_end.isoformat()},
        )

    span = window_end - window_start
    if span > timedelta(days=settings.HISTORY_MAX_RANGE_DAYS):
        raise InvalidTimeRangeError(
            f"The window may span at most {settings.HISTORY_MAX_RANGE_DAYS} days.",
            details={
                "requested_days": round(span.total_seconds() / 86400.0, 2),
                "max_days": settings.HISTORY_MAX_RANGE_DAYS,
            },
        )

    radius = min(
        radius_km or settings.HISTORY_DEFAULT_RADIUS_KM, settings.HISTORY_MAX_RADIUS_KM
    )
    limit = settings.HISTORY_MAX_RESULTS

    records = await service.observations(
        HistoryQuery(
            latitude=latitude,
            longitude=longitude,
            start=window_start,
            end=window_end,
            radius_m=radius * 1000.0,
            provider_id=provider,
            limit=limit,
        )
    )

    return HistoricalWeatherResponse(
        requested={"latitude": latitude, "longitude": longitude},
        range=TimeRange(start=window_start, end=window_end),
        search_radius_km=radius,
        count=len(records),
        truncated=len(records) >= limit,
        observations=[HistoricalObservation.from_record(r) for r in records],
    )
