"""Data provenance.

Every meteorological value the system emits carries a record of where it came
from. This is not decoration: a decision-support platform that cannot say which
model produced a number, when it was fetched, and under what licence it may be
redistributed is not usable for anything consequential.

Provenance also gives the future AI layer something honest to cite. The model
explains and contextualises values; it never originates them.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DataProvenance(BaseModel):
    """Origin and freshness of a meteorological record."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "provider_id": "open-meteo",
                "provider_name": "Open-Meteo",
                "model": "best_match",
                "fetched_at": "2026-09-04T07:30:00Z",
                "source_url": "https://api.open-meteo.com/v1/forecast",
                "license": "CC-BY-4.0",
                "attribution": "Weather data by Open-Meteo.com",
            }
        }
    )

    provider_id: str = Field(
        description="Stable identifier of the provider that served this data.",
        examples=["open-meteo"],
    )
    provider_name: str = Field(
        description="Human-readable provider name, for display and attribution.",
        examples=["Open-Meteo"],
    )
    model: str | None = Field(
        default=None,
        description="Numerical weather model behind the values, when disclosed.",
        examples=["best_match"],
    )
    fetched_at: datetime = Field(
        description="UTC instant the upstream response was received."
    )
    source_url: str | None = Field(
        default=None, description="Upstream endpoint the data was read from."
    )
    license: str | None = Field(
        default=None, description="Licence governing redistribution of the data."
    )
    attribution: str | None = Field(
        default=None,
        description="Attribution string that must accompany public display.",
    )
    cached: bool = Field(
        default=False,
        description="True when served from the response cache rather than a live fetch.",
    )
