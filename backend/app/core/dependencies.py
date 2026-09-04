"""FastAPI dependency boundary.

Routes depend on the annotated aliases exported here rather than on concrete
implementations. That keeps the API layer stable while the layers underneath it
are built out: when the datastore, the cache and the weather providers land,
they are introduced as new dependencies here and injected into routes without
rewriting them.

Concrete objects are built once by the composition root and reached through
``request.app.state.container``, so a test can swap the whole graph — or one
FastAPI dependency — without patching module internals.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Annotated, Awaitable, Callable, Union

from fastapi import Depends, Request

from app.config.logging import NO_REQUEST_ID
from app.config.settings import Settings, get_settings
from app.core.container import ApplicationContainer
from app.db.engine import Database
from app.providers.base import ProviderCapability
from app.providers.registry import ProviderRegistry
from app.services.geocoding import GeocodingService
from app.services.alerts import AlertService
from app.services.history import HistoryService
from app.services.subscriptions import SubscriptionService
from app.services.persistence import PersistenceService
from app.services.weather_service import WeatherService

SettingsDep = Annotated[Settings, Depends(get_settings)]
"""Validated application configuration."""


def get_request_id(request: Request) -> str:
    """Correlation id assigned to the current request by the HTTP middleware."""
    return getattr(request.state, "request_id", NO_REQUEST_ID)


RequestIdDep = Annotated[str, Depends(get_request_id)]
"""Correlation id, for handlers that need to reference it explicitly."""


def get_container(request: Request) -> ApplicationContainer:
    """The application graph built at startup by the composition root."""
    return request.app.state.container


ContainerDep = Annotated[ApplicationContainer, Depends(get_container)]


def get_weather_service(container: ContainerDep) -> WeatherService:
    """Application service backing the weather and forecast endpoints."""
    return container.weather


WeatherServiceDep = Annotated[WeatherService, Depends(get_weather_service)]


def get_geocoding_service(container: ContainerDep) -> GeocodingService:
    """Place-name resolution service."""
    return container.geocoding


GeocodingServiceDep = Annotated[GeocodingService, Depends(get_geocoding_service)]


def get_history_service(container: ContainerDep) -> HistoryService:
    """Read access to stored observations."""
    return container.history


HistoryServiceDep = Annotated[HistoryService, Depends(get_history_service)]


def get_subscription_service(container: ContainerDep) -> SubscriptionService:
    """The watched-location service owned by the lifespan container."""
    return container.subscriptions


SubscriptionServiceDep = Annotated[
    SubscriptionService, Depends(get_subscription_service)
]


def get_persistence_service(container: ContainerDep) -> PersistenceService:
    """Write access to durable storage."""
    return container.persistence


PersistenceServiceDep = Annotated[PersistenceService, Depends(get_persistence_service)]


def get_alert_service(container: ContainerDep) -> AlertService:
    """Deterministic alert evaluation and retrieval."""
    return container.alerts


AlertServiceDep = Annotated[AlertService, Depends(get_alert_service)]


def get_provider_registry(container: ContainerDep) -> ProviderRegistry:
    """Registry of the meteorological sources wired in at startup."""
    return container.registry


ProviderRegistryDep = Annotated[ProviderRegistry, Depends(get_provider_registry)]


@dataclass(frozen=True, slots=True)
class ComponentStatus:
    """Outcome of a single readiness check."""

    name: str
    healthy: bool
    detail: str | None = None


ReadinessCheck = Callable[[], Union[ComponentStatus, Awaitable[ComponentStatus]]]
"""A zero-argument callable, sync or async, reporting one component's state."""


class ReadinessRegistry:
    """Collects the checks that decide whether the service can take traffic.

    Phase 1 registers nothing, so the service is ready as soon as the process
    is up. Later phases register their own probes at startup, for example::

        registry.register("postgres", postgres_probe)
        registry.register("redis", redis_probe)
        registry.register("open-meteo", provider_probe)

    The readiness endpoint needs no change when they do.
    """

    def __init__(self) -> None:
        self._checks: dict[str, ReadinessCheck] = {}

    def register(self, name: str, check: ReadinessCheck) -> None:
        """Add (or replace) a named readiness check."""
        self._checks[name] = check

    def clear(self) -> None:
        """Drop all registered checks. Primarily useful in tests."""
        self._checks.clear()

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._checks)

    async def evaluate(self) -> list[ComponentStatus]:
        """Run every check. A raising check is reported as unhealthy, not fatal."""
        results: list[ComponentStatus] = []
        for name, check in self._checks.items():
            try:
                outcome = check()
                if inspect.isawaitable(outcome):
                    outcome = await outcome
                results.append(outcome)
            except Exception as exc:  # noqa: BLE001 - a probe must never 500
                results.append(
                    ComponentStatus(name=name, healthy=False, detail=str(exc))
                )
        return results


def register_provider_probes(
    providers: ProviderRegistry, readiness: ReadinessRegistry
) -> None:
    """Install the readiness probe covering the meteorological source layer.

    The probe is deliberately *local*: it asserts that this instance has at
    least one source able to answer current conditions. A misconfigured
    deployment with an empty registry genuinely cannot serve weather and should
    not receive traffic, and that is exactly what readiness is for.

    It does **not** call the upstream APIs. Gating readiness on a third party
    would take every instance out of the load balancer simultaneously the
    moment that third party had an outage — turning a degraded service into a
    total one, with nowhere healthier to route to. An upstream failure is
    already reported per-request, with provenance, by the ingestion pipeline.
    Active upstream monitoring belongs on a metrics path, not on a probe an
    orchestrator runs every few seconds.
    """

    def weather_sources_available() -> ComponentStatus:
        capable = providers.for_capability(ProviderCapability.CURRENT)
        return ComponentStatus(
            name="weather-providers",
            healthy=bool(capable),
            detail=(
                None
                if capable
                else "No registered provider can serve current conditions."
            ),
        )

    readiness.register("weather-providers", weather_sources_available)


def register_database_probe(database: Database, readiness: ReadinessRegistry) -> None:
    """Install the readiness probe for the datastore.

    Unlike the upstream weather sources — which are deliberately *not* probed,
    because a third-party outage would empty the whole fleet at once with
    nowhere healthier to route — the database is infrastructure this deployment
    owns. An instance that cannot reach it is commonly broken in a way that
    rerouting or restarting fixes: exhausted pool, stale credentials, lost
    network. That makes it a legitimate readiness signal.

    The trade-off is acknowledged: current conditions and forecasts still work
    without the database, so failing readiness on a shared outage costs more
    than it recovers. Writes are best-effort for exactly that reason, and only
    the historical endpoint hard-fails.
    """

    async def database_reachable() -> ComponentStatus:
        healthy = await database.ping()
        return ComponentStatus(
            name="database",
            healthy=healthy,
            detail=None if healthy else "The database is not reachable.",
        )

    readiness.register("database", database_reachable)


_readiness_registry = ReadinessRegistry()


def get_readiness_registry() -> ReadinessRegistry:
    """Return the process-wide readiness registry."""
    return _readiness_registry


ReadinessRegistryDep = Annotated[ReadinessRegistry, Depends(get_readiness_registry)]
"""Registry of dependency probes backing ``GET /health/ready``."""
