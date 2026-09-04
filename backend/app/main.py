"""WeatherGPT API — application entry point.

Assembly only. Configuration, observability, the error contract and the routing
tree each live in their own module; this file wires them together so the
composition of the service is readable in one screen.

Architectural rule the rest of the system depends on: meteorological values are
produced by validated data sources, never by a language model. The provider,
normalisation and validation layers land in later phases and plug in beneath
this API without changing it.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from time import perf_counter
from typing import AsyncIterator, Callable, Awaitable
from uuid import uuid4

from fastapi import FastAPI, Request, Response

from app.api.router import build_api_router
from app.config.logging import (
    REQUEST_ID_HEADER,
    configure_logging,
    get_logger,
    request_id_ctx,
)
from app.config.settings import Settings, get_settings
from app.core.container import build_container
from app.core.dependencies import (
    get_readiness_registry,
    register_database_probe,
    register_provider_probes,
)
from app.core.exceptions import register_exception_handlers

logger = get_logger(__name__)

DESCRIPTION = """
Backend for **WeatherGPT**, a conversational meteorological intelligence and
decision-support platform.

The service is built as a pipeline: meteorological sources are ingested,
normalised and validated into a single weather model before anything else in
the system may read them. The API, the alert engine and the AI layer all sit
*downstream* of that validation — a language model is never the source of truth
for a weather value.

Two guarantees hold for every meteorological value this API returns:

* **Normalised units.** Celsius, metres per second, hectopascals, millimetres —
  whatever the upstream source reported. Units are read from the source's own
  declaration and converted, never assumed.
* **Validated before served.** Values that are physically implausible or
  internally inconsistent are rejected and the request falls through to the
  next source. A wrong number is worse than no number.

Every response carries provenance: which source served it, which model produced
it, when it was fetched, and how it must be attributed.

**Phase 4 (current)** — providers, ingestion, normalisation and validation;
PostgreSQL/PostGIS persistence with historical and spatial queries; and a
deterministic alert and risk engine.

Alerts are produced by configurable thresholds applied to validated data, and
each one records the rule, the value, the threshold and the source behind it.
They are **not** official meteorological warnings, and no language model
participates in deciding that an alert exists — that boundary is deliberate.

**Next** — the AI explanation layer, which explains alerts it did not create.
"""

OPENAPI_TAGS = [
    {
        "name": "System",
        "description": (
            "Operational endpoints for health, liveness and readiness. Consumed "
            "by orchestrators, load balancers and uptime monitoring."
        ),
    },
    {
        "name": "Weather",
        "description": "Current conditions, normalised and validated.",
    },
    {
        "name": "Alerts",
        "description": (
            "Deterministic threshold alerts produced by WeatherGPT's rule "
            "engine. Not official meteorological warnings."
        ),
    },
    {
        "name": "Forecast",
        "description": "Hourly and daily forecasts, normalised and validated.",
    },
    {
        "name": "Locations",
        "description": "Resolve place names to coordinates.",
    },
    {
        "name": "Providers",
        "description": (
            "Discover the meteorological sources wired into this deployment, "
            "their capabilities and their attribution requirements."
        ),
    },
]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Own the application's long-lived resources.

    The composition root builds the object graph — HTTP connection pool,
    provider registry, ingestion pipeline, services — and it is torn down here
    on the way out. Readiness probes for the registered sources are installed
    at the same time, so ``/health/ready`` reflects the real dependency set
    without the health endpoint knowing what a weather provider is.
    """
    settings: Settings = app.state.settings
    container = build_container(settings)
    app.state.container = container
    readiness = get_readiness_registry()
    register_provider_probes(container.registry, readiness)
    if container.database is not None:
        register_database_probe(container.database, readiness)

    logger.info(
        "application.startup",
        extra={
            "service": settings.SERVICE_NAME,
            "app_version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT.value,
            "api_prefix": settings.API_V1_PREFIX,
            "debug": settings.DEBUG,
            "providers": [p.metadata.provider_id for p in container.registry.all()],
            "persistence": container.persistence.enabled,
        },
    )
    try:
        yield
    finally:
        await container.aclose()
        logger.info("application.shutdown", extra={"service": settings.SERVICE_NAME})


async def request_context_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Assign a correlation id to every request and emit one access log line.

    An inbound ``X-Request-ID`` is honoured so a caller's trace survives the hop;
    otherwise one is minted. The id is exposed to handlers via ``request.state``,
    to loggers via a context var, and to the client via the response header.
    """
    request_id = request.headers.get(REQUEST_ID_HEADER) or uuid4().hex
    request.state.request_id = request_id
    token = request_id_ctx.set(request_id)
    started = perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        # The error contract turns this into a 500 envelope; log it with timing
        # while the correlation id is still bound.
        logger.exception(
            "request.failed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "duration_ms": round((perf_counter() - started) * 1000, 2),
            },
        )
        raise
    else:
        response.headers[REQUEST_ID_HEADER] = request_id
        logger.info(
            "request.completed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round((perf_counter() - started) * 1000, 2),
            },
        )
        return response
    finally:
        request_id_ctx.reset(token)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a fully wired application instance.

    Exposed as a factory so tests (and future workers) can construct an app with
    explicit settings instead of relying on import-time global state.
    """
    settings = settings or get_settings()
    configure_logging(settings)

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=DESCRIPTION,
        openapi_tags=OPENAPI_TAGS,
        openapi_url="/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    app.state.settings = settings

    app.middleware("http")(request_context_middleware)
    register_exception_handlers(app)
    app.include_router(build_api_router(settings))

    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover - local convenience entry point
    import uvicorn

    _settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=_settings.DEBUG,
        log_config=None,  # logging is configured by the application itself
    )
