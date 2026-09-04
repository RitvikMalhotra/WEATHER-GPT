"""Geoapify geocoding and the two-gazetteer chain behind it.

The chain exists for one reason: a locality below Open-Meteo's population
threshold is *absent* from its gazetteer, not merely ranked low, so no amount
of asking for more candidates finds it. These tests pin the mapping, the
place-only filter, and the fallback behaviour that keeps a missing key or a
dead upstream from changing today's answers.
"""

from __future__ import annotations

import pytest

from app.core.exceptions import WeatherProviderError
from app.domain.location import Coordinates, Location
from app.providers.geoapify.geocoding import GeoapifyGeocodingClient
from app.services.cache import TTLCache
from app.services.geocoding import GeocodingService

SEARCH_URL = "https://api.geoapify.test/v1/geocode/search"


class StubHttp:
    """Stands in for the shared upstream client."""

    def __init__(self, body: dict | None = None, error: Exception | None = None) -> None:
        self._body = body or {}
        self._error = error
        self.calls: list[tuple[str, dict]] = []

    async def get_json(self, url: str, *, params: dict | None = None) -> dict:
        self.calls.append((url, dict(params or {})))
        if self._error is not None:
            raise self._error
        return self._body


class StubGazetteer:
    """A geocoding client with a scripted answer."""

    def __init__(self, results: list[Location] | None = None, error: Exception | None = None) -> None:
        self._results = results or []
        self._error = error
        self.queries: list[str] = []

    async def search(self, query: str, *, limit: int = 5, language: str = "en") -> list[Location]:
        self.queries.append(query)
        if self._error is not None:
            raise self._error
        return self._results[:limit]


def place(name: str, admin1: str = "Telangana") -> Location:
    return Location(
        coordinates=Coordinates(latitude=17.5, longitude=78.4),
        name=name,
        admin1=admin1,
        country="India",
    )


def entry(**overrides) -> dict:
    base = {
        "name": "Miyapur",
        "city": "Hyderabad",
        "state": "Telangana",
        "country": "India",
        "country_code": "in",
        "lat": 17.4948,
        "lon": 78.3908,
        "result_type": "suburb",
        "timezone": {"name": "Asia/Kolkata"},
    }
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_maps_a_record_onto_a_location() -> None:
    http = StubHttp({"results": [entry()]})
    client = GeoapifyGeocodingClient(http, search_url=SEARCH_URL, api_key="k")

    [result] = await client.search("Miyapur")

    assert result.name == "Miyapur"
    assert result.admin1 == "Telangana"
    assert result.country == "India"
    assert result.country_code == "IN"
    assert result.timezone == "Asia/Kolkata"
    assert result.coordinates.latitude == pytest.approx(17.4948)
    assert result.display_name == "Miyapur, Telangana, India"


@pytest.mark.asyncio
async def test_key_travels_in_the_query_and_never_in_the_result() -> None:
    http = StubHttp({"results": [entry()]})
    client = GeoapifyGeocodingClient(http, search_url=SEARCH_URL, api_key="secret-key")

    await client.search("Miyapur", limit=4)

    url, params = http.calls[0]
    assert url == SEARCH_URL
    assert params["apiKey"] == "secret-key"
    assert params["text"] == "Miyapur"
    # Over-fetching is what makes room for the place filter below.
    assert params["limit"] > 4


@pytest.mark.asyncio
async def test_drops_results_that_are_not_places() -> None:
    """A street or a shop is an address, not somewhere a forecast is issued."""
    http = StubHttp(
        {
            "results": [
                entry(name="Miyapur Road", result_type="street"),
                entry(name="Miyapur Metro", result_type="building"),
                entry(),
            ]
        }
    )
    client = GeoapifyGeocodingClient(http, search_url=SEARCH_URL, api_key="k")

    results = await client.search("Miyapur")

    assert [r.name for r in results] == ["Miyapur"]


@pytest.mark.asyncio
async def test_collapses_duplicates_of_the_same_place() -> None:
    http = StubHttp(
        {
            "results": [
                entry(),
                entry(lat=17.4949, lon=78.3909),
                entry(name="Miyapur", state="Karnataka"),
            ]
        }
    )
    client = GeoapifyGeocodingClient(http, search_url=SEARCH_URL, api_key="k")

    results = await client.search("Miyapur")

    # Two genuinely different Miyapurs survive; the repeat of the first does not.
    assert [(r.name, r.admin1) for r in results] == [
        ("Miyapur", "Telangana"),
        ("Miyapur", "Karnataka"),
    ]


@pytest.mark.asyncio
async def test_unusable_records_are_skipped_not_fatal() -> None:
    http = StubHttp({"results": ["nonsense", {"lat": 17.5}, entry()]})
    client = GeoapifyGeocodingClient(http, search_url=SEARCH_URL, api_key="k")

    assert len(await client.search("Miyapur")) == 1


@pytest.mark.asyncio
async def test_no_match_is_an_empty_list_not_an_error() -> None:
    client = GeoapifyGeocodingClient(StubHttp({}), search_url=SEARCH_URL, api_key="k")

    assert await client.search("qqqqqq") == []


def build_service(primary: StubGazetteer, fallback: StubGazetteer | None) -> GeocodingService:
    return GeocodingService(
        primary,
        cache=TTLCache[list[Location]](ttl_seconds=0.0),
        fallback=fallback,
    )


@pytest.mark.asyncio
async def test_primary_answer_wins_and_the_fallback_is_never_asked() -> None:
    primary = StubGazetteer([place("Miyapur")])
    fallback = StubGazetteer([place("Miyapur", "Karnataka")])

    results = await build_service(primary, fallback).search("Miyapur")

    assert [r.admin1 for r in results] == ["Telangana"]
    assert fallback.queries == []


@pytest.mark.asyncio
async def test_empty_primary_falls_through() -> None:
    primary = StubGazetteer([])
    fallback = StubGazetteer([place("Delhi", "Delhi")])

    results = await build_service(primary, fallback).search("Delhi")

    assert [r.name for r in results] == ["Delhi"]
    assert fallback.queries == ["Delhi"]


@pytest.mark.asyncio
async def test_unreachable_primary_falls_through() -> None:
    primary = StubGazetteer(error=WeatherProviderError("down"))
    fallback = StubGazetteer([place("Delhi", "Delhi")])

    results = await build_service(primary, fallback).search("Delhi")

    assert [r.name for r in results] == ["Delhi"]


@pytest.mark.asyncio
async def test_without_a_fallback_a_failure_still_surfaces() -> None:
    primary = StubGazetteer(error=WeatherProviderError("down"))

    with pytest.raises(WeatherProviderError):
        await build_service(primary, None).search("Delhi")


@pytest.mark.asyncio
async def test_without_a_key_openstreetmap_leads_and_open_meteo_backs_it() -> None:
    """The keyless default still reaches localities and Hindi place names."""
    from app.config.settings import Settings
    from app.core.container import build_container

    container = build_container(Settings(DATABASE_URL=None, GEOAPIFY_API_KEY=None))
    try:
        assert type(container.geocoding._client).__name__ == "NominatimGeocodingClient"  # noqa: SLF001
        assert type(container.geocoding._fallback).__name__ == "OpenMeteoGeocodingClient"  # noqa: SLF001
    finally:
        await container.aclose()


@pytest.mark.asyncio
async def test_with_every_extra_gazetteer_off_open_meteo_answers_alone() -> None:
    """The original single-source behaviour is still one setting away."""
    from app.config.settings import Settings
    from app.core.container import build_container

    container = build_container(
        Settings(DATABASE_URL=None, GEOAPIFY_API_KEY=None, NOMINATIM_ENABLED=False)
    )
    try:
        assert type(container.geocoding._client).__name__ == "OpenMeteoGeocodingClient"  # noqa: SLF001
        assert container.geocoding._fallback is None  # noqa: SLF001
    finally:
        await container.aclose()


@pytest.mark.asyncio
async def test_geoapify_leads_when_a_key_is_present() -> None:
    from app.config.settings import Settings
    from app.core.container import build_container

    container = build_container(Settings(DATABASE_URL=None, GEOAPIFY_API_KEY="k"))
    try:
        assert type(container.geocoding._client).__name__ == "GeoapifyGeocodingClient"  # noqa: SLF001
        assert type(container.geocoding._fallback).__name__ == "OpenMeteoGeocodingClient"  # noqa: SLF001
    finally:
        await container.aclose()
