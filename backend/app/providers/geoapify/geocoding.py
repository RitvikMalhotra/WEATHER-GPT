"""Geoapify geocoding: place names to coordinates, at neighbourhood resolution.

Why a second gazetteer exists at all: Open-Meteo's gazetteer only carries
populated places above a population threshold, so an Indian locality such as
Miyapur in Hyderabad is absent from it entirely and the nearest same-named
village in another state wins the query. That is not a ranking problem a caller
can fix — the right answer is not in the list. Geoapify indexes localities,
suburbs and districts, so the right answer is there to be chosen.

It resolves locations and nothing else. No weather value ever comes from here;
the coordinates it returns are fed to the existing provider chain exactly as
Open-Meteo's are. Like the other client, it emits
:class:`~app.domain.location.Location` objects, so nothing downstream can tell
which gazetteer answered.
"""

from __future__ import annotations

from typing import Any

from app.domain.location import Coordinates, Location
from app.providers.http import UpstreamHttpClient

GEOAPIFY_GEOCODER_ID = "geoapify-geocoding"

#: Result kinds that name a *place*. A street, a building or a shop is a
#: precise address, not a location a forecast is issued for, and letting one
#: through fills a disambiguation list with four entrances to the same suburb.
_PLACE_TYPES = frozenset(
    {"city", "town", "village", "hamlet", "suburb", "district", "county", "state", "locality"}
)


def _first_text(entry: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _to_location(entry: Any) -> Location | None:
    """Map one Geoapify record onto a Location, skipping unusable entries."""
    if not isinstance(entry, dict):
        return None

    latitude, longitude = entry.get("lat"), entry.get("lon")
    if not isinstance(latitude, (int, float)) or not isinstance(longitude, (int, float)):
        return None
    if not (-90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0):
        return None

    result_type = entry.get("result_type")
    if isinstance(result_type, str) and result_type not in _PLACE_TYPES:
        return None

    timezone = entry.get("timezone")
    country_code = entry.get("country_code")

    return Location(
        coordinates=Coordinates(latitude=float(latitude), longitude=float(longitude)),
        # Most specific place label first: a suburb is what the person named.
        name=_first_text(entry, "name", "suburb", "district", "city", "county", "address_line1"),
        country=_first_text(entry, "country"),
        country_code=country_code.upper() if isinstance(country_code, str) else None,
        # `state` is the first-level administrative area, matching admin1 in the
        # existing gazetteer, so both sources produce the same shaped label.
        admin1=_first_text(entry, "state", "county"),
        timezone=timezone.get("name") if isinstance(timezone, dict) else None,
    )


def _dedupe_key(location: Location) -> tuple[str, str, str]:
    return (
        (location.name or "").casefold(),
        (location.admin1 or "").casefold(),
        (location.country or "").casefold(),
    )


class GeoapifyGeocodingClient:
    """Search the Geoapify gazetteer for places matching a name."""

    def __init__(self, http: UpstreamHttpClient, *, search_url: str, api_key: str) -> None:
        self._http = http
        self._search_url = search_url
        self._api_key = api_key

    async def search(self, query: str, *, limit: int = 5, language: str = "en") -> list[Location]:
        """Return candidate locations, best match first.

        An unmatched query yields an empty list rather than an error, so the
        caller can fall through to another gazetteer.

        Raises:
            WeatherProviderError: the gazetteer was unreachable or errored.
        """
        body = await self._http.get_json(
            self._search_url,
            params={
                "text": query,
                # Over-fetch: the place filter and the deduplication below both
                # discard rows, and a short list is the point of the feature.
                "limit": min(limit * 3, 20),
                "lang": language,
                "format": "json",
                "apiKey": self._api_key,
            },
        )
        results = body.get("results") or []
        if not isinstance(results, list):
            return []

        seen: set[tuple[str, str, str]] = set()
        locations: list[Location] = []
        for entry in results:
            location = _to_location(entry)
            if location is None or not location.name:
                continue
            key = _dedupe_key(location)
            if key in seen:
                continue
            seen.add(key)
            locations.append(location)
            if len(locations) >= limit:
                break
        return locations
