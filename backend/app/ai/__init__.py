"""Conversational layer for WeatherGPT.

This package is deliberately downstream-only.  Its tools talk to WeatherGPT's
versioned HTTP API and never to a meteorological provider, the database, or the
alert engine.  An LLM may select a tool, but it never originates weather facts
or alert decisions.
"""

from app.ai.service import AIService, build_ai_service

__all__ = ["AIService", "build_ai_service"]
