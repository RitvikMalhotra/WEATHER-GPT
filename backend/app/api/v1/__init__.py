"""Version 1 of the WeatherGPT API."""

from fastapi import APIRouter

from app.api.v1.alerts import router as alerts_router
from app.api.v1.forecast import router as forecast_router
from app.api.v1.health import router as health_router
from app.api.v1.historical import router as historical_router
from app.api.v1.locations import router as locations_router
from app.api.v1.providers import router as providers_router
from app.api.v1.subscriptions import router as subscriptions_router
from app.api.v1.weather import router as weather_router

v1_router = APIRouter()
v1_router.include_router(health_router)
v1_router.include_router(weather_router)
v1_router.include_router(historical_router)
v1_router.include_router(alerts_router)
# Mounted after the alert routes so /alerts/subscriptions is not
# shadowed by a path parameter on /alerts.
v1_router.include_router(subscriptions_router)
v1_router.include_router(forecast_router)
v1_router.include_router(locations_router)
v1_router.include_router(providers_router)

__all__ = ["v1_router"]
