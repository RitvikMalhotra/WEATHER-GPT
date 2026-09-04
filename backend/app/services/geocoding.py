"""Place-name resolution.

Turns "Delhi" into coordinates so the rest of the system can work in latitude
and longitude. Results are cached aggressively — a city does not move — which
keeps the gazetteer out of the hot path for repeat queries.
"""

from __future__ import annotations

from app.config.logging import get_logger
from app.core.exceptions import LocationNotFoundError
from app.domain.location import Location
from app.providers.open_meteo.geocoding import OpenMeteoGeocodingClient
from app.services.cache import TTLCache

logger = get_logger(__name__)


class GeocodingService:
    """Resolves place names to :class:`Location` objects."""

    def __init__(
        self,
        client: OpenMeteoGeocodingClient,
        *,
        cache: TTLCache[list[Location]],
        default_limit: int = 5,
    ) -> None:
        self._client = client
        self._cache = cache
        self._default_limit = default_limit

    async def search(self, query: str, *, limit: int | None = None) -> list[Location]:
        """Return candidate locations for a query, best match first.

        An unmatched query returns an empty list.

        Raises:
            LocationNotFoundError: the query is blank.
            WeatherProviderError: the gazetteer was unreachable.
        """
        normalised = query.strip()
        if not normalised:
            raise LocationNotFoundError("A location query must not be empty.")

        count = limit or self._default_limit
        key = f"{normalised.casefold()}|{count}"
        results, cached = await self._cache.get_or_load(
            key, lambda: self._client.search(normalised, limit=count)
        )
        logger.info(
            "geocoding.search",
            extra={"query": normalised, "results": len(results), "cached": cached},
        )
        return results

    async def resolve(self, query: str) -> Location:
        """Resolve a query to exactly one location, the best match.

        Raises:
            LocationNotFoundError: nothing matched the query.
        """
        results = await self.search(query, limit=1)
        if not results:
            raise LocationNotFoundError(
                f"No location matched {query.strip()!r}.",
                details={"query": query.strip()},
            )
        return results[0]
