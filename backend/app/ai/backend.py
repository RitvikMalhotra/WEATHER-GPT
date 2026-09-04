"""HTTP client for WeatherGPT's own public API.

This is the AI layer's only path to weather information.  Keeping the boundary
at the published FastAPI contract means validation, provider fallback,
persistence and the deterministic Alert Engine remain owned by the backend.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.ai.models import BackendError
from app.api.v1.alerts import AlertListResponse
from app.api.v1.historical import HistoricalWeatherResponse
from app.api.v1.locations import LocationSearchResponse
from app.domain.forecast import Forecast
from app.domain.location import Coordinates
from app.domain.weather import WeatherReport

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class BackendAPIError(RuntimeError):
    """A safe error returned by, or while reaching, the WeatherGPT API."""

    def __init__(self, error: BackendError, *, status_code: int | None = None) -> None:
        super().__init__(error.message)
        self.error = error
        self.status_code = status_code


class BackendAPIClient:
    """Typed consumer of existing FastAPI routes; never an upstream provider client."""

    def __init__(
        self,
        *,
        base_url: str,
        api_prefix: str,
        timeout_seconds: float = 10.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_prefix = "/" + api_prefix.strip("/")
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=timeout_seconds
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def current(
        self,
        *,
        latitude: float | None = None,
        longitude: float | None = None,
        location: str | None = None,
        provider: str | None = None,
    ) -> WeatherReport:
        data = await self._get(
            "/weather/current",
            self._location_params(latitude, longitude, location, provider),
        )
        return _validated(WeatherReport, data)

    async def forecast(
        self,
        *,
        latitude: float | None = None,
        longitude: float | None = None,
        location: str | None = None,
        days: int = 7,
        hourly: bool = False,
        daily: bool = True,
        provider: str | None = None,
    ) -> Forecast:
        params = self._location_params(latitude, longitude, location, provider)
        params.update({"days": days, "hourly": hourly, "daily": daily})
        data = await self._get("/forecast", params)
        return _validated(Forecast, data)

    async def historical(
        self,
        *,
        latitude: float,
        longitude: float,
        start: str,
        end: str,
        radius_km: float | None = None,
        hour_from: int | None = None,
        hour_to: int | None = None,
        provider: str | None = None,
    ) -> HistoricalWeatherResponse:
        params: dict[str, Any] = {
            "latitude": latitude,
            "longitude": longitude,
            "start": start,
            "end": end,
        }
        if radius_km is not None:
            params["radius_km"] = radius_km
        if hour_from is not None and hour_to is not None:
            params["hour_from"] = hour_from
            params["hour_to"] = hour_to
        if provider:
            params["provider"] = provider
        data = await self._get("/weather/historical", params)
        return _validated(HistoricalWeatherResponse, data)

    async def alerts(
        self,
        *,
        latitude: float,
        longitude: float,
        radius_km: float | None = None,
        provider: str | None = None,
    ) -> AlertListResponse:
        params: dict[str, Any] = {"latitude": latitude, "longitude": longitude}
        if radius_km is not None:
            params["radius_km"] = radius_km
        if provider:
            params["provider"] = provider
        data = await self._get("/weather/alerts", params)
        return _validated(AlertListResponse, data)

    async def locations(
        self,
        *,
        query: str,
        limit: int = 5,
        near: Coordinates | None = None,
    ) -> LocationSearchResponse:
        params: dict[str, Any] = {"q": query, "limit": limit}
        # The place the conversation is already about. It reorders candidates
        # and nothing else.
        if near is not None:
            params["near_lat"] = near.latitude
            params["near_lon"] = near.longitude
        data = await self._get("/locations/search", params)
        return _validated(LocationSearchResponse, data)

    async def _get(self, path: str, params: Mapping[str, Any]) -> dict[str, Any]:
        try:
            response = await self._client.get(f"{self._api_prefix}{path}", params=params)
        except httpx.HTTPError as exc:
            raise BackendAPIError(
                BackendError(message="Weather backend data is temporarily unavailable.")
            ) from exc

        try:
            body = response.json()
        except ValueError:
            body = {}

        if response.is_success:
            if not isinstance(body, dict):
                raise BackendAPIError(
                    BackendError(message="Weather backend returned an invalid response."),
                    status_code=response.status_code,
                )
            return body

        envelope = body.get("error", {}) if isinstance(body, dict) else {}
        error = BackendError(
            code=str(envelope.get("code", "BACKEND_UNAVAILABLE")),
            message=str(envelope.get("message", "Weather backend data is unavailable.")),
            request_id=envelope.get("request_id"),
        )
        raise BackendAPIError(error, status_code=response.status_code)

    @staticmethod
    def _location_params(
        latitude: float | None,
        longitude: float | None,
        location: str | None,
        provider: str | None,
    ) -> dict[str, Any]:
        params: dict[str, Any]
        if latitude is not None and longitude is not None:
            params = {"latitude": latitude, "longitude": longitude}
        elif location:
            params = {"location": location}
        else:
            raise ValueError("a complete coordinate pair or location is required")
        if provider:
            params["provider"] = provider
        return params


def _validated(model: type[ResponseModel], data: dict[str, Any]) -> ResponseModel:
    """Treat malformed successful responses as unavailable rather than improvise."""
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise BackendAPIError(
            BackendError(message="Weather backend returned an invalid response.")
        ) from exc
