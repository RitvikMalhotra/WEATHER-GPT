"""Provider registry and fallback ordering.

Holds the registered sources and answers one question for the ingestion
pipeline: *given a capability, which sources should I try, and in what order?*

Ordering is by declared priority, with the configured default provider promoted
to the front. That is the whole policy — deliberately simple, because the
interesting resilience (retries, validation, fallback) lives in the pipeline.
"""

from __future__ import annotations

from app.core.exceptions import ProviderNotFoundError
from app.providers.base import ProviderCapability, WeatherProvider


class ProviderRegistry:
    """Ordered collection of the meteorological sources available to the app."""

    def __init__(self, *, default_provider_id: str | None = None) -> None:
        self._providers: dict[str, WeatherProvider] = {}
        self._default_provider_id = default_provider_id

    def register(self, provider: WeatherProvider) -> None:
        """Add a provider, replacing any previous registration under its id."""
        self._providers[provider.metadata.provider_id] = provider

    def get(self, provider_id: str) -> WeatherProvider:
        """Return one provider by id.

        Raises:
            ProviderNotFoundError: no provider is registered under that id.
        """
        try:
            return self._providers[provider_id]
        except KeyError:
            raise ProviderNotFoundError(
                f"Unknown weather provider: {provider_id!r}.",
                details={"available": sorted(self._providers)},
            ) from None

    def all(self) -> list[WeatherProvider]:
        """Every registered provider, in fallback order."""
        return self._sorted(self._providers.values())

    def for_capability(
        self,
        capability: ProviderCapability,
        *,
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> list[WeatherProvider]:
        """Providers able to serve ``capability``, in the order to try them.

        Given a point, a source whose declared coverage excludes it is dropped,
        and a source that covers it regionally is tried first. That is what puts
        a national service ahead of a global model inside its own country
        without the caller, or a language model, choosing a provider.
        """
        candidates = [
            provider
            for provider in self._providers.values()
            if provider.metadata.supports(capability)
        ]
        if latitude is not None and longitude is not None:
            candidates = [
                provider
                for provider in candidates
                if provider.metadata.covers(latitude, longitude)
            ]
        return self._sorted(candidates, latitude=latitude, longitude=longitude)

    def _sorted(
        self,
        providers,
        *,
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> list[WeatherProvider]:
        located = latitude is not None and longitude is not None

        def sort_key(provider: WeatherProvider) -> tuple[int, int, int, str]:
            metadata = provider.metadata
            # A regional source that covers the point leads. This outranks the
            # configured default deliberately: the default is the global
            # fallback, not the best source for a country that runs its own.
            regional_first = 0 if (located and metadata.coverage is not None) else 1
            is_default = metadata.provider_id != self._default_provider_id
            return (regional_first, int(is_default), metadata.priority, metadata.provider_id)

        return sorted(providers, key=sort_key)

    def __len__(self) -> int:
        return len(self._providers)

    def __contains__(self, provider_id: object) -> bool:
        return provider_id in self._providers
