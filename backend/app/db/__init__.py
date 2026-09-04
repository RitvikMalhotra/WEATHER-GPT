"""Persistence layer.

PostgreSQL with PostGIS, reached through SQLAlchemy's async engine. The layer
is deliberately thin: models describe the schema, mappers translate to and from
the canonical domain model, repositories hold the queries, and the engine owns
the connection lifecycle and stops driver exceptions from escaping.

The database stores validated meteorological data. It never produces it — that
remains the providers' job, upstream of validation.
"""

from app.db.base import Base
from app.db.engine import Database, translate_database_errors
from app.db.models import WeatherForecast, WeatherObservation

__all__ = [
    "Base",
    "Database",
    "WeatherForecast",
    "WeatherObservation",
    "translate_database_errors",
]
