"""Provider discovery endpoint.

Makes the source layer inspectable: which meteorological sources are wired in,
what each can answer, in what order they are tried, and how their data must be
attributed. Useful to clients choosing a source explicitly, and to anyone
auditing where a number came from.
"""

from __future__ import annotations

from fastapi import APIRouter, status
from pydantic import BaseModel, ConfigDict, Field

from app.core.dependencies import ProviderRegistryDep, SettingsDep
from app.providers.base import ProviderCapability, ProviderMetadata

router = APIRouter(prefix="/providers", tags=["Providers"])


class ProviderInfo(BaseModel):
    """A registered meteorological source."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "provider_id": "open-meteo",
                "name": "Open-Meteo",
                "capabilities": ["current", "daily_forecast", "hourly_forecast"],
                "model": "best_match",
                "source_url": "https://open-meteo.com",
                "license": "CC-BY-4.0",
                "attribution": "Weather data by Open-Meteo.com",
                "max_forecast_days": 16,
                "priority": 10,
                "is_default": True,
            }
        }
    )

    provider_id: str = Field(description="Value to pass as the `provider` query parameter.")
    name: str = Field(description="Human-readable source name.")
    capabilities: list[ProviderCapability] = Field(
        description="What this source is able to answer."
    )
    model: str | None = Field(default=None, description="Numerical model behind the data.")
    source_url: str | None = Field(default=None, description="Upstream service.")
    license: str | None = Field(default=None, description="Redistribution licence.")
    attribution: str | None = Field(
        default=None, description="Attribution required on public display."
    )
    max_forecast_days: int = Field(description="Longest forecast horizon offered.")
    priority: int = Field(description="Fallback order; lower is tried first.")
    is_default: bool = Field(description="Whether this source leads the fallback chain.")
    notes: list[str] = Field(default_factory=list, description="Source-specific notes.")

    @classmethod
    def from_metadata(cls, metadata: ProviderMetadata, *, is_default: bool) -> "ProviderInfo":
        return cls(
            provider_id=metadata.provider_id,
            name=metadata.name,
            capabilities=sorted(metadata.capabilities, key=lambda c: c.value),
            model=metadata.model,
            source_url=metadata.source_url,
            license=metadata.license,
            attribution=metadata.attribution,
            max_forecast_days=metadata.max_forecast_days,
            priority=metadata.priority,
            is_default=is_default,
            notes=list(metadata.notes),
        )


class ProviderListResponse(BaseModel):
    """Every registered source, in fallback order."""

    count: int = Field(description="Number of registered sources.")
    default_provider: str = Field(description="Source that leads the fallback chain.")
    providers: list[ProviderInfo] = Field(
        description="Sources in the order the pipeline tries them."
    )


@router.get(
    "",
    response_model=ProviderListResponse,
    status_code=status.HTTP_200_OK,
    summary="List meteorological sources",
    description=(
        "The sources wired into this deployment, in the order the ingestion "
        "pipeline tries them. Any `provider_id` listed here may be passed as "
        "the `provider` query parameter on the weather and forecast endpoints "
        "to pin a request to one source."
    ),
    response_description="Registered sources with capabilities and attribution.",
)
async def list_providers(
    registry: ProviderRegistryDep, settings: SettingsDep
) -> ProviderListResponse:
    providers = [
        ProviderInfo.from_metadata(
            provider.metadata,
            is_default=provider.metadata.provider_id == settings.DEFAULT_PROVIDER,
        )
        for provider in registry.all()
    ]
    return ProviderListResponse(
        count=len(providers),
        default_provider=settings.DEFAULT_PROVIDER,
        providers=providers,
    )
