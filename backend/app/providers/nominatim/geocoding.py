"""OpenStreetMap geocoding: place names to coordinates, at locality resolution.

Why this exists. The gazetteer WeatherGPT started with indexes populated places
above a population threshold, which is a reasonable rule that produces one
unreasonable result: an Indian locality such as Miyapur in Hyderabad is *absent
from it entirely*, so the query lands on a same-named village in another state
and answers confidently with the wrong city's weather. That is not a ranking
problem — asking for more candidates cannot return a row that does not exist.

OpenStreetMap indexes suburbs, villages and neighbourhoods, and it carries
``name:hi``, so "दिल्ली" resolves to Delhi without a transliteration table or a
dictionary of Indian cities. Both of those are the same fix: put the right rows
in the list, then rank them.

It needs no API key, which is what makes it the default. In exchange it asks
for one request per second and an honest User-Agent, so this client holds a
minimum interval between calls and the service above it caches for a day.

Location resolution only. No weather value ever comes from here.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from app.domain.location import Coordinates, Location
from app.providers.http import UpstreamHttpClient

NOMINATIM_GEOCODER_ID = "nominatim"

#: OSM place classes that name somewhere weather is reported for. A road or a
#: shop is a precise address, not a location a forecast is issued at, and
#: letting one through fills a disambiguation list with four entrances to the
#: same suburb.
_PLACE_TYPES = frozenset(
    {
        "city", "town", "village", "hamlet", "suburb", "neighbourhood", "quarter",
        "borough", "municipality", "district", "county", "state", "province",
        "region", "island", "locality", "city_district", "administrative",
    }
)

#: Administrative fields, most specific first, used to name a result and to
#: place it in its hierarchy.
_LOCALITY_FIELDS = (
    "village", "hamlet", "suburb", "neighbourhood", "quarter", "town",
    "city_district", "municipality", "city",
)
_REGION_FIELDS = ("state", "province", "region", "state_district", "county")


def _text(source: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _to_location(entry: Any) -> Location | None:
    """Map one Nominatim record onto a Location, skipping unusable entries."""
    if not isinstance(entry, dict):
        return None

    try:
        latitude = float(entry["lat"])
        longitude = float(entry["lon"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (-90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0):
        return None

    kind = entry.get("addresstype") or entry.get("type") or entry.get("class")
    if isinstance(kind, str) and kind not in _PLACE_TYPES:
        return None

    address = entry.get("address") if isinstance(entry.get("address"), dict) else {}
    name = _text(entry, "name") or _text(address, *_LOCALITY_FIELDS)
    if not name:
        return None

    country_code = _text(address, "country_code")
    return Location(
        coordinates=Coordinates(latitude=latitude, longitude=longitude),
        name=name,
        country=_text(address, "country"),
        country_code=country_code.upper() if country_code else None,
        # `admin1` is the first-level area in the canonical model, which lines
        # up with OSM's `state`, so both gazetteers produce the same label.
        admin1=_text(address, *_REGION_FIELDS),
    )


def _city_of(entry: Any) -> str | None:
    """The settlement a locality sits inside, when it is not the locality itself."""
    if not isinstance(entry, dict):
        return None
    address = entry.get("address") if isinstance(entry.get("address"), dict) else {}
    city = _text(address, "city", "town", "municipality")
    name = _text(entry, "name")
    return city if city and city != name else None


class NominatimGeocodingClient:
    """Search OpenStreetMap for places matching a name.

    The rate limit is honoured here rather than left to the caller: it is a
    property of this upstream, and forgetting it gets a deployment blocked.
    """

    def __init__(
        self,
        http: UpstreamHttpClient,
        *,
        search_url: str,
        reverse_url: str | None = None,
        min_interval_seconds: float = 1.0,
    ) -> None:
        self._http = http
        self._search_url = search_url
        self._reverse_url = reverse_url or search_url.rsplit("/", 1)[0] + "/reverse"
        self._min_interval = min_interval_seconds
        self._last_call = 0.0
        self._gate = asyncio.Lock()

    async def _wait_turn(self) -> None:
        """Hold OpenStreetMap's one-request-per-second courtesy limit."""
        async with self._gate:
            elapsed = time.monotonic() - self._last_call
            if elapsed < self._min_interval:
                await asyncio.sleep(self._min_interval - elapsed)
            self._last_call = time.monotonic()

    async def reverse(self, latitude: float, longitude: float) -> Location | None:
        """Name the place a pair of coordinates falls in.

        Used for browser geolocation: a device reports a point, and a person
        needs to see somewhere they recognise before they trust the answer.

        Raises:
            WeatherProviderError: the gazetteer was unreachable or errored.
        """
        await self._wait_turn()
        body = await self._http.get_json(
            self._reverse_url,
            params={
                "lat": f"{latitude:.6f}",
                "lon": f"{longitude:.6f}",
                "format": "jsonv2",
                "addressdetails": 1,
                # Zoom 14 is the district/suburb level: specific enough to
                # recognise, coarse enough not to read as surveillance.
                "zoom": 14,
                "accept-language": "en",
            },
        )
        if not isinstance(body, dict):
            return None
        location = _to_location({**body, "addresstype": body.get("addresstype") or "city"})
        if location is None:
            return None
        city = _city_of(body)
        if city:
            location = location.model_copy(update={"name": f"{location.name}, {city}"})
        # The point that was asked about, not the centroid of the area it fell in.
        return location.model_copy(
            update={"coordinates": Coordinates(latitude=latitude, longitude=longitude)}
        )

    async def search(self, query: str, *, limit: int = 5, language: str = "en") -> list[Location]:
        """Return candidate locations, best match first.

        An unmatched query yields an empty list rather than an error, so the
        caller can fall through to another gazetteer.

        Raises:
            WeatherProviderError: the gazetteer was unreachable or errored.
        """
        await self._wait_turn()
        body = await self._http.get_json(
            self._search_url,
            # Nominatim answers with a top-level array of records.
            allow_array=True,
            params={
                "q": query,
                "format": "jsonv2",
                # Over-fetch: the place filter below discards rows, and a short
                # ranked list is the point of the feature.
                "limit": min(max(limit * 3, 10), 40),
                "addressdetails": 1,
                # Names come back in English even when the query was in
                # Devanagari, which is what makes a Hindi place name usable
                # everywhere downstream without transliterating anything.
                "accept-language": "en",
            },
        )
        if not isinstance(body, list):
            return []

        seen: set[tuple[str, str, str]] = set()
        locations: list[Location] = []
        for entry in body:
            location = _to_location(entry)
            if location is None:
                continue
            city = _city_of(entry)
            if city:
                # "Miyapur, Hyderabad, Telangana" says which Miyapur far better
                # than "Miyapur, Telangana" does, and the canonical model has
                # one name field to say it in.
                location = location.model_copy(update={"name": f"{location.name}, {city}"})
            key = (
                (location.name or "").casefold(),
                (location.admin1 or "").casefold(),
                (location.country or "").casefold(),
            )
            if key in seen:
                continue
            seen.add(key)
            locations.append(location)
            if len(locations) >= limit:
                break
        return locations
