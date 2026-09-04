"""Root API router.

Versioning is applied here and nowhere else. Every v1 endpoint is mounted under
``settings.API_V1_PREFIX`` (``/api/v1`` by default), so the surface grows as::

    /api/v1/health      <- this phase
    /api/v1/weather
    /api/v1/forecast
    /api/v1/alerts

A future ``/api/v2`` is added by mounting a second version router beside the
first; existing clients keep working because no route hard-codes its prefix.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import v1_router
from app.config.settings import Settings


def build_api_router(settings: Settings) -> APIRouter:
    """Compose the versioned routers into a single mountable router."""
    # Imported lazily to keep the public API schemas usable as typed contracts
    # by the downstream AI client without an import cycle through app.api.
    from app.ai.router import router as ai_router
    from app.voice.router import router as voice_router

    router = APIRouter()
    router.include_router(v1_router, prefix=settings.API_V1_PREFIX)
    router.include_router(ai_router, prefix=settings.API_V1_PREFIX)
    # Unversioned: a page a person opens, not an API contract.
    router.include_router(voice_router)
    return router
