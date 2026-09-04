"""Open-Meteo geocoding: place names to coordinates.

Geocoding is a separate upstream service with its own schema, so it gets its
own client rather than being folded into the weather provider. It produces
:class:`~app.domain.location.Location` objects — the same type the weather
layer consumes — so the rest of the system never sees a raw gazetteer record.
"""

from __future__ import annotations

from typing import Any

from app.domain.location import Coordinates, Location
from app.providers.http import UpstreamHttpClient

GEOCODER_ID = "open-meteo-geocoding"


def _to_location(entry: Any) -> Location | None:
    """Map one gazetteer record onto a Location, skipping unusable entries."""
    if not isinstance(entry, dict):
        return None
    latitude, longitude = entry.get("latitude"), entry.get("longitude")
    if not isinstance(latitude, (int, float)) or not isinstance(longitude, (int, float)):
        return None
    if not (-90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0):
        return None

    elevation = entry.get("elevation")
    return Location(
        coordinates=Coordinates(
            latitude=float(latitude),
            longitude=float(longitude),
            # Gazetteer elevations occasionally fall outside the domain bounds;
            # a bad elevation should not cost us an otherwise valid place.
            elevation_m=(
                float(elevation)
                if isinstance(elevation, (int, float)) and -500.0 <= elevation <= 9000.0
                else None
            ),
        ),
        name=entry.get("name"),
        country=entry.get("country"),
        country_code=entry.get("country_code"),
        admin1=entry.get("admin1"),
        timezone=entry.get("timezone"),
        population=entry.get("population"),
    )


class OpenMeteoGeocodingClient:
    """Search the Open-Meteo gazetteer for places matching a name."""

    def __init__(self, http: UpstreamHttpClient, *, search_url: str) -> None:
        self._http = http
        self._search_url = search_url

    async def search(self, query: str, *, limit: int = 5, language: str = "en") -> list[Location]:
        """Return candidate locations, best match first.

        An unmatched query yields an empty list rather than an error — deciding
        whether "no results" is a failure belongs to the caller.

        Raises:
            WeatherProviderError: the gazetteer was unreachable or errored.
        """
        body = await self._http.get_json(
            self._search_url,
            params={
                "name": query,
                "count": limit,
                "language": language,
                "format": "json",
            },
        )
        # Open-Meteo omits "results" entirely when nothing matches.
        results = body.get("results") or []
        if not isinstance(results, list):
            return []

        locations = (_to_location(entry) for entry in results)
        return [location for location in locations if location is not None]
