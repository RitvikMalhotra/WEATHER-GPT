"""HTTP surface for the isolated conversational layer."""

from __future__ import annotations

from fastapi import APIRouter, Request, status

from app.ai.models import ChatRequest, ChatResponse
from app.ai.service import AIService

router = APIRouter(prefix="/ai", tags=["AI"])


def get_ai_service(request: Request) -> AIService:
    """Read the lifespan-owned service without coupling AI to backend internals."""
    return request.app.state.ai_service


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Grounded weather conversation",
    description=(
        "Interprets a weather question, invokes read-only tools against the existing "
        "WeatherGPT API, and renders only the data returned in that request. The "
        "optional LLM selects tools but cannot author weather facts or alerts. "
        "WeatherGPT alerts returned here are deterministic rule results, not official warnings."
    ),
)
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    return await get_ai_service(request).chat(payload)
