"""Meteorological data sources.

Each source implements :class:`~app.providers.base.WeatherProvider` and is
responsible for exactly two things: talking to its upstream API, and mapping the
response onto the canonical domain model. Validation, caching, fallback and
provider selection all live above this layer.
"""

from app.providers.base import ProviderCapability, ProviderMetadata, WeatherProvider
from app.providers.http import UpstreamHttpClient, build_http_client
from app.providers.registry import ProviderRegistry

__all__ = [
    "ProviderCapability",
    "ProviderMetadata",
    "ProviderRegistry",
    "UpstreamHttpClient",
    "WeatherProvider",
    "build_http_client",
]
