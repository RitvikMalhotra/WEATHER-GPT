"""Shared query parameters for the meteorological endpoints.

Location can be given two ways — coordinates or a place name — and every weather
endpoint accepts both. Parsing and validating that choice once, here, keeps the
rule in a single place and keeps the routes free of argument juggling.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Query, status

from app.core.exceptions import ErrorResponse, WeatherGPTError
from app.domain.location import Coordinates
from app.services.weather_service import LocationQuery

#: Documented failure modes shared by every meteorological endpoint.
WEATHER_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    status.HTTP_404_NOT_FOUND: {
        "model": ErrorResponse,
        "description": "The place name could not be resolved to coordinates.",
    },
    422: {
        "model": ErrorResponse,
        "description": "The request arguments were missing, incomplete or contradictory.",
    },
    status.HTTP_502_BAD_GATEWAY: {
        "model": ErrorResponse,
        "description": "An upstream source failed or returned unusable data.",
    },
    status.HTTP_503_SERVICE_UNAVAILABLE: {
        "model": ErrorResponse,
        "description": "No source produced data that passed validation.",
    },
}


class InvalidLocationQueryError(WeatherGPTError):
    """The caller gave neither coordinates nor a place name, or gave half a pair."""

    code = "INVALID_LOCATION_QUERY"
    status_code = 422
    message = (
        "Provide either both 'latitude' and 'longitude', or a 'location' place name."
    )


def location_query(
    latitude: Annotated[
        float | None,
        Query(
            ge=-90.0,
            le=90.0,
            description="Latitude in decimal degrees. Requires `longitude`.",
            examples=[28.6139],
        ),
    ] = None,
    longitude: Annotated[
        float | None,
        Query(
            ge=-180.0,
            le=180.0,
            description="Longitude in decimal degrees. Requires `latitude`.",
            examples=[77.209],
        ),
    ] = None,
    location: Annotated[
        str | None,
        Query(
            min_length=1,
            max_length=200,
            description=(
                "Place name to geocode, e.g. `Mumbai`. Ignored when coordinates "
                "are supplied."
            ),
            examples=["New Delhi"],
        ),
    ] = None,
) -> LocationQuery:
    """Resolve the location arguments into a single query object.

    Coordinates take precedence: they are unambiguous and need no geocoding.

    Raises:
        InvalidLocationQueryError: neither form was supplied, or only one half
            of a coordinate pair was.
    """
    if latitude is not None and longitude is not None:
        return LocationQuery(coordinates=Coordinates(latitude=latitude, longitude=longitude))

    if (latitude is None) != (longitude is None):
        raise InvalidLocationQueryError(
            "'latitude' and 'longitude' must be supplied together.",
            details={"latitude": latitude, "longitude": longitude},
        )

    if location:
        return LocationQuery(place=location)

    raise InvalidLocationQueryError()


LocationQueryDep = Annotated[LocationQuery, Depends(location_query)]

ProviderQuery = Annotated[
    str | None,
    Query(
        alias="provider",
        description=(
            "Force a specific source, e.g. `open-meteo`. Omit to use the default "
            "chain with fallback. An explicit choice is never silently "
            "substituted, so the response provenance stays truthful."
        ),
        examples=["open-meteo"],
    ),
]
