"""Forecast endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, status

from app.api.v1.params import (
    WEATHER_ERROR_RESPONSES,
    LocationQueryDep,
    ProviderQuery,
)
from app.core.dependencies import SettingsDep, WeatherServiceDep
from app.core.exceptions import WeatherGPTError
from app.domain.forecast import Forecast

router = APIRouter(prefix="/forecast", tags=["Forecast"])

#: Absolute ceiling advertised in the schema. The effective limit is the lower
#: of MAX_FORECAST_DAYS and what the serving provider supports.
_SCHEMA_MAX_DAYS = 16


class EmptySeriesSelectionError(WeatherGPTError):
    """The caller switched off both forecast series, leaving nothing to return."""

    code = "EMPTY_SERIES_SELECTION"
    status_code = 422
    message = "Enable at least one of 'hourly' or 'daily'."


@router.get(
    "",
    response_model=Forecast,
    status_code=status.HTTP_200_OK,
    summary="Weather forecast",
    description=(
        "Forecast for a point, given either coordinates or a place name.\n\n"
        "Two resolutions are available and both are opt-in, so a caller pays "
        "only for the series it needs: `daily` aggregates each local calendar "
        "day, `hourly` gives per-hour detail. Requesting neither is rejected.\n\n"
        "As with current conditions, values are normalised to canonical units, "
        "every point is validated before it is served, and the response carries "
        "full provenance."
    ),
    response_description="Validated forecast series with provenance.",
    responses=WEATHER_ERROR_RESPONSES,
)
async def weather_forecast(
    query: LocationQueryDep,
    service: WeatherServiceDep,
    settings: SettingsDep,
    days: Annotated[
        int | None,
        Query(
            ge=1,
            le=_SCHEMA_MAX_DAYS,
            description="Forecast horizon in days. Defaults to the configured horizon.",
            examples=[7],
        ),
    ] = None,
    hourly: Annotated[
        bool, Query(description="Include the hourly series.")
    ] = False,
    daily: Annotated[
        bool, Query(description="Include the daily series.")
    ] = True,
    provider: ProviderQuery = None,
) -> Forecast:
    if not hourly and not daily:
        raise EmptySeriesSelectionError()

    horizon = min(days or settings.DEFAULT_FORECAST_DAYS, settings.MAX_FORECAST_DAYS)
    return await service.get_forecast(
        query,
        days=horizon,
        include_hourly=hourly,
        include_daily=daily,
        provider_id=provider,
    )
