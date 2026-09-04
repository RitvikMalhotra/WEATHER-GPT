"""Open-Meteo integration.

Split three ways so each concern is testable on its own:

    client.py      wire format and HTTP access
    normalizer.py  pure mapping from that wire format to the domain model
    provider.py    the WeatherProvider implementation that composes them
    geocoding.py   the separate place-name lookup service
"""

from app.providers.open_meteo.client import PROVIDER_ID, OpenMeteoClient, OpenMeteoPayload
from app.providers.open_meteo.geocoding import GEOCODER_ID, OpenMeteoGeocodingClient
from app.providers.open_meteo.provider import METADATA, OpenMeteoProvider

__all__ = [
    "GEOCODER_ID",
    "METADATA",
    "PROVIDER_ID",
    "OpenMeteoClient",
    "OpenMeteoGeocodingClient",
    "OpenMeteoPayload",
    "OpenMeteoProvider",
]
