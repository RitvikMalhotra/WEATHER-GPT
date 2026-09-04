"""Geographic location model.

Coordinates are the system's primary key for weather: place names are resolved
to coordinates once, at the edge, and every layer below works in latitude and
longitude. That keeps the provider layer free of geocoding concerns and makes
results cacheable by a stable key.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Coordinates(BaseModel):
    """A WGS-84 point, optionally with elevation."""

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "example": {"latitude": 28.6139, "longitude": 77.209, "elevation_m": 216.0}
        },
    )

    latitude: float = Field(ge=-90.0, le=90.0, description="Degrees north of the equator.")
    longitude: float = Field(
        ge=-180.0, le=180.0, description="Degrees east of the prime meridian."
    )
    elevation_m: float | None = Field(
        default=None,
        ge=-500.0,
        le=9000.0,
        description="Ground elevation above mean sea level, in metres.",
    )

    @property
    def cache_key(self) -> str:
        """Coordinates rounded to ~11 m, used to group nearby lookups."""
        return f"{self.latitude:.4f},{self.longitude:.4f}"


class Location(BaseModel):
    """A resolved place: coordinates plus the administrative labels around them."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "coordinates": {
                    "latitude": 28.6139,
                    "longitude": 77.209,
                    "elevation_m": 216.0,
                },
                "name": "New Delhi",
                "country": "India",
                "country_code": "IN",
                "admin1": "Delhi",
                "timezone": "Asia/Kolkata",
            }
        }
    )

    coordinates: Coordinates = Field(description="Resolved geographic position.")
    name: str | None = Field(default=None, description="Populated place name.")
    country: str | None = Field(default=None, description="Country name.")
    country_code: str | None = Field(
        default=None, description="ISO 3166-1 alpha-2 country code.", examples=["IN"]
    )
    admin1: str | None = Field(
        default=None, description="First-level administrative area (state or region)."
    )
    timezone: str | None = Field(
        default=None, description="IANA timezone name.", examples=["Asia/Kolkata"]
    )
    population: int | None = Field(
        default=None, ge=0, description="Population, when the gazetteer reports it."
    )

    @property
    def display_name(self) -> str:
        """Human-facing label, e.g. ``New Delhi, Delhi, India``."""
        parts = [part for part in (self.name, self.admin1, self.country) if part]
        return ", ".join(parts) if parts else self.coordinates.cache_key
