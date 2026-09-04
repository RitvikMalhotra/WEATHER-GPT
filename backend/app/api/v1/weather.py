"""Current-conditions endpoints."""

from __future__ import annotations

from fastapi import APIRouter, status

from app.api.v1.params import (
    WEATHER_ERROR_RESPONSES,
    LocationQueryDep,
    ProviderQuery,
)
from app.core.dependencies import WeatherServiceDep
from app.domain.weather import WeatherReport

router = APIRouter(prefix="/weather", tags=["Weather"])


@router.get(
    "/current",
    response_model=WeatherReport,
    status_code=status.HTTP_200_OK,
    summary="Current conditions",
    description=(
        "Current conditions for a point, given either coordinates or a place "
        "name.\n\n"
        "Values are normalised to the canonical unit system — Celsius, metres "
        "per second, hectopascals, millimetres — regardless of what the "
        "upstream source reported, and every response carries the provenance "
        "of the data: which source served it, which model produced it, when it "
        "was fetched, and whether it came from cache.\n\n"
        "Data that fails meteorological validation is never returned. If a "
        "source reports implausible or self-inconsistent values, the request "
        "falls through to the next source rather than serving the number."
    ),
    response_description="Validated current conditions with provenance.",
    responses=WEATHER_ERROR_RESPONSES,
)
async def current_weather(
    query: LocationQueryDep,
    service: WeatherServiceDep,
    provider: ProviderQuery = None,
) -> WeatherReport:
    return await service.get_current(query, provider_id=provider)
