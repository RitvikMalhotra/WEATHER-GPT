"""System status endpoints.

Three endpoints with three distinct jobs — the distinction is what makes them
usable by an orchestrator:

``/health/live``   the process is running. Never touches a dependency, so a
                   slow database can never cause a restart loop.
``/health/ready``  the service can serve traffic. Consults every probe in the
                   readiness registry, so an instance with a dead datastore is
                   pulled from the load balancer without being killed.
``/health``        overall application status, including build identity.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, status
from pydantic import BaseModel, ConfigDict, Field

from app.config.settings import Environment
from app.core.dependencies import ReadinessRegistryDep, SettingsDep
from app.core.exceptions import ErrorResponse, ServiceUnavailableError

router = APIRouter(prefix="/health", tags=["System"])


class HealthResponse(BaseModel):
    """Overall application status and build identity."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "service": "weathergpt-backend",
                "version": "1.0.0",
                "environment": "development",
            }
        }
    )

    status: Literal["healthy"] = Field(description="Aggregate application status.")
    service: str = Field(description="Logical service name used in logs and metrics.")
    version: str = Field(description="Deployed application version.")
    environment: Environment = Field(description="Environment the process runs in.")


class LivenessResponse(BaseModel):
    """Process-level liveness."""

    model_config = ConfigDict(json_schema_extra={"example": {"status": "alive"}})

    status: Literal["alive"] = Field(description="Always 'alive' if the process answers.")


class ReadinessResponse(BaseModel):
    """Traffic readiness."""

    model_config = ConfigDict(json_schema_extra={"example": {"status": "ready"}})

    status: Literal["ready"] = Field(
        description="'ready' when every registered dependency probe passes."
    )


@router.get(
    "",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Application health",
    description=(
        "Reports the aggregate status of the service together with the build "
        "identity it is running. Use this for dashboards and smoke tests; use "
        "`/health/live` and `/health/ready` for orchestrator probes."
    ),
    response_description="The service is healthy.",
)
async def health(settings: SettingsDep) -> HealthResponse:
    return HealthResponse(
        status="healthy",
        service=settings.SERVICE_NAME,
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
    )


@router.get(
    "/live",
    response_model=LivenessResponse,
    status_code=status.HTTP_200_OK,
    summary="Liveness probe",
    description=(
        "Returns as long as the process can serve a request. Intentionally free "
        "of dependency checks so a degraded dependency never triggers a restart."
    ),
    response_description="The process is alive.",
)
async def liveness() -> LivenessResponse:
    return LivenessResponse(status="alive")


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    status_code=status.HTTP_200_OK,
    summary="Readiness probe",
    description=(
        "Returns 200 when every registered dependency probe passes, and 503 "
        "with the failing components otherwise. No probes are registered in "
        "this phase; the datastore, cache and weather providers register theirs "
        "at startup as they are introduced."
    ),
    response_description="The service is ready to accept traffic.",
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ErrorResponse,
            "description": "One or more dependencies are unavailable.",
        }
    },
)
async def readiness(registry: ReadinessRegistryDep) -> ReadinessResponse:
    components = await registry.evaluate()
    failed = [component for component in components if not component.healthy]
    if failed:
        raise ServiceUnavailableError(
            details={
                "components": [
                    {"name": component.name, "detail": component.detail}
                    for component in failed
                ]
            }
        )
    return ReadinessResponse(status="ready")
