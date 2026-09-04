"""Ingestion, normalisation and validation.

The layer between raw sources and the rest of the application. Nothing above it
sees a provider payload, and nothing above it sees data that has not passed the
validation gate.
"""

from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.validation import (
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
    WeatherValidator,
)

__all__ = [
    "IngestionPipeline",
    "ValidationIssue",
    "ValidationResult",
    "ValidationSeverity",
    "WeatherValidator",
]
