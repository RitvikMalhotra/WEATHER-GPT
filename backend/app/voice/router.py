"""Serves the WeatherGPT client.

The page is a thin shell: speech recognition and synthesis run in the browser,
and every weather question goes to ``/api/v1/ai/chat`` like any other client.
There is no client-specific weather path, and no audio reaches this service.

The stylesheet and script are served beside the page rather than inlined into
it, so the browser can cache them independently and the three concerns stay in
three readable files. This router is presentation delivery only: it holds no
weather, provider, alert or AI logic.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, Response

router = APIRouter(tags=["Voice"])

_HERE = Path(__file__).parent


def _asset(name: str) -> str:
    return (_HERE / name).read_text(encoding="utf-8")


@router.get(
    "/",
    response_class=HTMLResponse,
    include_in_schema=False,
    summary="WeatherGPT home",
)
@router.get(
    "/voice",
    response_class=HTMLResponse,
    include_in_schema=False,
    summary="WeatherGPT client",
)
async def voice_page() -> HTMLResponse:
    return HTMLResponse(_asset("page.html"))


@router.get("/voice/styles.css", include_in_schema=False)
async def voice_styles() -> Response:
    return Response(_asset("styles.css"), media_type="text/css; charset=utf-8")


@router.get("/voice/app.js", include_in_schema=False)
async def voice_script() -> Response:
    return Response(_asset("app.js"), media_type="text/javascript; charset=utf-8")
