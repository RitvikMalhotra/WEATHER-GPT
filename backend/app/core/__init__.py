"""Cross-cutting application primitives: errors and dependency boundaries."""

from app.core.exceptions import (
    ErrorDetail,
    ErrorResponse,
    ForecastUnavailableError,
    LocationNotFoundError,
    ProviderNotFoundError,
    ServiceUnavailableError,
    WeatherDataUnavailableError,
    WeatherDataValidationError,
    WeatherGPTError,
    WeatherProviderError,
    WeatherProviderTimeoutError,
    register_exception_handlers,
)

__all__ = [
    "ErrorDetail",
    "ErrorResponse",
    "ForecastUnavailableError",
    "LocationNotFoundError",
    "ProviderNotFoundError",
    "ServiceUnavailableError",
    "WeatherDataUnavailableError",
    "WeatherDataValidationError",
    "WeatherGPTError",
    "WeatherProviderError",
    "WeatherProviderTimeoutError",
    "register_exception_handlers",
]
