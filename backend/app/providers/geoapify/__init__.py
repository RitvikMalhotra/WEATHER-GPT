"""Geoapify geocoding.

Location resolution only. This package never retrieves weather: it turns a
place name into coordinates, and the existing provider chain does the rest.
"""

from app.providers.geoapify.geocoding import GEOAPIFY_GEOCODER_ID, GeoapifyGeocodingClient

__all__ = ["GEOAPIFY_GEOCODER_ID", "GeoapifyGeocodingClient"]
