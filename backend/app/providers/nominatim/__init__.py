"""OpenStreetMap place-name resolution.

Location resolution only. This package never retrieves weather.
"""

from app.providers.nominatim.geocoding import NOMINATIM_GEOCODER_ID, NominatimGeocodingClient

__all__ = ["NOMINATIM_GEOCODER_ID", "NominatimGeocodingClient"]
