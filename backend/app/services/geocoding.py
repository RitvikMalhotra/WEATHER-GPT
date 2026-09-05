"""Place-name resolution.

Turns "Delhi" into coordinates so the rest of the system can work in latitude
and longitude. Results are cached aggressively — a city does not move — which
keeps the gazetteer out of the hot path for repeat queries.

Two things happen here that a single gazetteer call cannot do.

**A chain, not a source.** Gazetteers differ in what they *index*, not only in
how they rank. A locality below a population threshold is absent from one and
present in another, and "absent" is not a ranking problem: asking for more
candidates cannot return a row that does not exist. So the first source
configured answers, and a second catches what it cannot see.

**Ranking with context.** "Miyapur" is a suburb of Hyderabad and a village in
Rajasthan and a village elsewhere in Telangana. Which one a person means is
usually settled by the conversation they are already having, so a caller can
pass the place they were last talking about and candidates near it rise. When
the question is genuinely open, the ranked list goes back to the caller to ask
about — never quietly resolved to the top row.
"""

from __future__ import annotations

import math
import unicodedata
from typing import Protocol, Sequence

from app.config.logging import get_logger
from app.core.exceptions import LocationNotFoundError, WeatherProviderError
from app.domain.location import Coordinates, Location
from app.services.cache import TTLCache

logger = get_logger(__name__)


class GeocodingClient(Protocol):
    """The one call a gazetteer has to answer."""

    async def search(
        self, query: str, *, limit: int = 5, language: str = "en"
    ) -> list[Location]: ...


def fold(value: str | None) -> str:
    """Compare names without being defeated by an accent.

    "Miyāpur" and "Miyapur" are the same name written twice; a gazetteer
    returns either, and a comparison that tells them apart hides the ambiguity
    it was meant to surface.
    """
    text = unicodedata.normalize("NFD", value or "")
    stripped = "".join(ch for ch in text if not unicodedata.combining(ch))
    return stripped.strip().casefold()


def distance_km(a: Coordinates, b: Coordinates) -> float:
    """Great-circle distance between two points, in kilometres."""
    radius = 6371.0
    lat1, lat2 = math.radians(a.latitude), math.radians(b.latitude)
    dlat = lat2 - lat1
    dlon = math.radians(b.longitude - a.longitude)
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * radius * math.asin(min(1.0, math.sqrt(h)))


def rank(
    results: Sequence[Location], query: str, near: Coordinates | None = None
) -> list[Location]:
    """Order candidates by how likely each is to be the one that was meant.

    The signals, in the order they matter: the name is exactly what was asked
    for; the place is near what the conversation is already about; it is a
    place many people live in. Source order breaks the remaining ties, which
    preserves whatever relevance the gazetteer itself computed.
    """
    wanted = fold(query)
    mentioned_countries = {
        fold(location.country)
        for location in results
        if location.country and fold(location.country) in wanted
    }
    explicit_country = bool(mentioned_countries)

    def score(item: tuple[int, Location]) -> tuple[float, int]:
        index, location = item
        points = 0.0

        # A gazetteer may return "Miyapur, Hyderabad"; the name asked for is
        # the first segment of it.
        head = fold((location.name or "").split(",")[0])
        if head == wanted:
            points += 100.0
        elif wanted and wanted in fold(location.name):
            points += 40.0

        if not explicit_country and fold(location.country) == "india":
            points += 200.0

        if near is not None:
            separation = distance_km(near, location.coordinates)
            if separation < 60.0:
                points += 90.0
            elif separation < 250.0:
                points += 45.0
            elif separation < 700.0:
                points += 15.0

        if location.population:
            points += min(location.population / 100_000.0, 12.0)

        return (points, -index)

    ordered = sorted(enumerate(results), key=score, reverse=True)
    return [location for _, location in ordered]


class GeocodingService:
    """Resolves place names to :class:`Location` objects."""

    def __init__(
        self,
        client: GeocodingClient,
        *,
        cache: TTLCache[list[Location]],
        default_limit: int = 5,
        fallback: GeocodingClient | None = None,
    ) -> None:
        self._client = client
        self._cache = cache
        self._default_limit = default_limit
        self._fallback = fallback

    async def _lookup(self, query: str, count: int) -> list[Location]:
        """Ask the primary gazetteer, then the fallback if it has nothing."""
        try:
            results = await self._client.search(query, limit=count)
        except WeatherProviderError:
            if self._fallback is None:
                raise
            logger.warning("geocoding.primary_unavailable", extra={"query": query})
            return await self._fallback.search(query, limit=count)
        if results or self._fallback is None:
            return results
        logger.info("geocoding.fallback", extra={"query": query})
        return await self._fallback.search(query, limit=count)

    async def search(
        self,
        query: str,
        *,
        limit: int | None = None,
        near: Coordinates | None = None,
    ) -> list[Location]:
        """Return candidate locations for a query, best match first.

        ``near`` is the place the conversation is already about. It only
        reorders candidates; it never adds, removes or invents one, so a
        question about a place on the other side of the country still finds it.

        An unmatched query returns an empty list.

        Raises:
            LocationNotFoundError: the query is blank.
            WeatherProviderError: every gazetteer was unreachable.
        """
        normalised = query.strip()
        if not normalised:
            raise LocationNotFoundError("A location query must not be empty.")

        count = limit or self._default_limit
        # The cache key deliberately excludes the bias: the *candidates* for a
        # name do not depend on where you are asking from, only their order
        # does, and ordering is cheap.
        key = f"{normalised.casefold()}|{count}"
        results, cached = await self._cache.get_or_load(
            key, lambda: self._lookup(normalised, count)
        )
        ordered = rank(results, normalised, near) if results else results
        logger.info(
            "geocoding.search",
            extra={
                "query": normalised,
                "results": len(ordered),
                "cached": cached,
                "biased": near is not None,
            },
        )
        return ordered

    async def resolve(self, query: str, *, near: Coordinates | None = None) -> Location:
        """Resolve a query to exactly one location, the best match.

        Raises:
            LocationNotFoundError: nothing matched the query.
        """
        # Asks for several and takes the best *after* ranking, rather than
        # asking for one and taking whatever the gazetteer put first.
        results = await self.search(query, limit=self._default_limit, near=near)
        if not results:
            raise LocationNotFoundError(
                f"No location matched {query.strip()!r}.",
                details={"query": query.strip()},
            )
        return results[0]

    async def reverse(self, latitude: float, longitude: float) -> Location | None:
        """Name a point, when a configured gazetteer can.

        Returns ``None`` rather than a guess when none can: an unnamed point is
        still a usable location, and a made-up neighbourhood is not.
        """
        for client in (self._client, self._fallback):
            finder = getattr(client, "reverse", None)
            if finder is None:
                continue
            try:
                found = await finder(latitude, longitude)
            except WeatherProviderError:
                logger.warning("geocoding.reverse_unavailable")
                continue
            if found is not None:
                return found
        return None

    @staticmethod
    def ambiguous(results: Sequence[Location]) -> bool:
        """Whether a ranked list holds more than one place of the same name.

        Two rows called Miyapur in different states is a question for the
        person who asked, not a decision for this service.
        """
        names = [fold((item.name or "").split(",")[0]) for item in results[:5]]
        leader = names[0] if names else ""
        contexts = {
            (item.admin1 or "", item.country or "")
            for item, name in zip(results[:5], names)
            if name == leader
        }
        return len(contexts) > 1
