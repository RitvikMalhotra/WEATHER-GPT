"""Repositories: the only place that writes SQL.

Each repository takes a session and exposes the queries this application
actually runs. Services depend on these rather than on SQLAlchemy directly, so
query shape stays reviewable in one place and the service layer stays testable
against a fake.
"""

from app.db.repositories.alerts import AlertFilter, AlertRepository, NearbyAlert
from app.db.repositories.forecasts import ForecastRepository
from app.db.repositories.observations import NearbyObservation, ObservationRepository
from app.db.repositories.subscriptions import (
    SubscriptionRepository,
    row_to_location,
    subscription_values,
)

__all__ = [
    "AlertFilter",
    "AlertRepository",
    "ForecastRepository",
    "NearbyAlert",
    "NearbyObservation",
    "ObservationRepository",
    "SubscriptionRepository",
    "row_to_location",
    "subscription_values",
]
