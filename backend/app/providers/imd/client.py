"""India Meteorological Department API access.

Endpoints and field names come from IMD's own published reference at
https://api.imd.gov.in/public/api_reference.html — nothing here is inferred
from a third-party mirror, and no IMD website page is scraped.

Products used:

    /api/v1/cityforecast_mapping  station catalogue (for location mapping)
    /api/v1/current_wx?id=...     current observation at a station
    /api/v1/cityforecast?id=...   seven-day city forecast

Access requires a key issued by IMD; every endpoint answers
``401 {"error": "API key missing"}`` without one. The key is read from the
environment and sent per request. IMD's documentation does not publish the
header name, so it is configurable rather than guessed, and the key can be
sent as a query parameter instead where a deployment's registration requires
that.
"""

from __future__ import annotations

from typing import Any

from app.core.exceptions import WeatherProviderError
from app.providers.http import UpstreamHttpClient

PROVIDER_ID = "imd"

#: Documented units. IMD fixes these in its API reference rather than declaring
#: them per response, so they are asserted here, in the one module that speaks
#: IMD's wire format, instead of being assumed downstream.
WIND_SPEED_UNIT = "km/h"
TEMPERATURE_UNIT = "°C"


class ImdClient:
    """Typed access to the IMD public API."""

    def __init__(
        self,
        http: UpstreamHttpClient,
        *,
        base_url: str,
        api_key: str,
        api_key_header: str = "x-api-key",
        api_key_param: str | None = None,
    ) -> None:
        self._http = http
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._api_key_header = api_key_header
        self._api_key_param = api_key_param

    async def station_catalogue(self) -> list[dict[str, Any]]:
        """Every city-forecast station IMD publishes, with its coordinates."""
        return _as_records(await self._get("/cityforecast_mapping"))

    async def current_weather(self, station_id: str) -> dict[str, Any]:
        """The latest observation for one station."""
        records = _as_records(await self._get("/current_wx", {"id": station_id}))
        if not records:
            raise WeatherProviderError(
                "IMD returned no current observation for the station.",
                provider_id=PROVIDER_ID,
                details={"station": station_id},
            )
        return records[0]

    async def city_forecast(self, station_id: str) -> dict[str, Any]:
        """The seven-day forecast for one station."""
        records = _as_records(await self._get("/cityforecast", {"id": station_id}))
        if not records:
            raise WeatherProviderError(
                "IMD returned no forecast for the station.",
                provider_id=PROVIDER_ID,
                details={"station": station_id},
            )
        return records[0]

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        query = dict(params or {})
        headers = {}
        # Registration decides how the key travels; support both rather than
        # betting on one, since sending it the wrong way reads as "key missing".
        if self._api_key_param:
            query[self._api_key_param] = self._api_key
        else:
            headers[self._api_key_header] = self._api_key
        return await self._http.get_json(
            f"{self._base_url}{path}",
            params=query,
            headers=headers or None,
            allow_array=True,
        )


def _as_records(body: Any) -> list[dict[str, Any]]:
    """IMD returns either one object or a list of them, depending on the product."""
    if isinstance(body, list):
        return [item for item in body if isinstance(item, dict)]
    if isinstance(body, dict):
        # An error envelope must never be mistaken for data.
        if "error" in body and len(body) == 1:
            raise WeatherProviderError(
                f"IMD rejected the request: {body['error']}",
                provider_id=PROVIDER_ID,
            )
        return [body]
    raise WeatherProviderError(
        "IMD returned an unrecognised response shape.", provider_id=PROVIDER_ID
    )
