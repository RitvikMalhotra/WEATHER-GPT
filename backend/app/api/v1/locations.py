"""Location resolution endpoints.

Exposed separately from the weather endpoints because clients need it on its
own: a search box resolves a place before any weather is requested, and the
conversational layer will need to disambiguate "Springfield" by asking rather
than guessing.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, status
from pydantic import BaseModel, ConfigDict, Field

from app.api.v1.params import WEATHER_ERROR_RESPONSES
from app.core.dependencies import GeocodingServiceDep
from app.domain.location import Location

router = APIRouter(prefix="/locations", tags=["Locations"])


class LocationSearchResponse(BaseModel):
    """Candidate locations for a search query."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "delhi",
                "count": 1,
                "results": [
                    {
                        "coordinates": {"latitude": 28.6519, "longitude": 77.2315},
                        "name": "Delhi",
                        "country": "India",
                        "country_code": "IN",
                        "admin1": "Delhi",
                        "timezone": "Asia/Kolkata",
                    }
                ],
            }
        }
    )

    query: str = Field(description="The query as it was interpreted.")
    count: int = Field(description="Number of candidates returned.")
    results: list[Location] = Field(
        description="Candidates ordered by match quality, best first."
    )


@router.get(
    "/search",
    response_model=LocationSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Search for a location",
    description=(
        "Resolve a place name to coordinates. Returns ranked candidates rather "
        "than a single guess, so an ambiguous name can be disambiguated by the "
        "caller instead of silently resolving to the wrong city.\n\n"
        "An unmatched query returns an empty list with a 200, not an error — "
        "'no such place' is a valid answer to a search."
    ),
    response_description="Ranked location candidates.",
    responses=WEATHER_ERROR_RESPONSES,
)
async def search_locations(
    service: GeocodingServiceDep,
    q: Annotated[
        str,
        Query(
            min_length=1,
            max_length=200,
            description="Place name to search for.",
            examples=["Delhi"],
        ),
    ],
    limit: Annotated[
        int, Query(ge=1, le=20, description="Maximum candidates to return.")
    ] = 5,
) -> LocationSearchResponse:
    results = await service.search(q, limit=limit)
    return LocationSearchResponse(query=q.strip(), count=len(results), results=results)
