"""Application errors and the single API error contract.

Every failure leaving the API — raised by us, by FastAPI's validation layer, or
by an unhandled bug — is rendered in one shape::

    {
      "error": {
        "code": "WEATHER_DATA_UNAVAILABLE",
        "message": "Weather data is temporarily unavailable.",
        "request_id": "0f1c...",
        "details": {...}
      }
    }

Clients can therefore branch on a stable ``code`` instead of parsing prose, and
every error is tied back to a log line through ``request_id``.

The meteorological hierarchy subclasses :class:`WeatherGPTError` and needs no
handler of its own — the base handler serialises every one of them.
"""

from __future__ import annotations

from http import HTTPStatus
from typing import Any, Mapping

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config.logging import (
    NO_REQUEST_ID,
    REQUEST_ID_HEADER,
    get_logger,
    request_id_ctx,
)

logger = get_logger(__name__)


class WeatherGPTError(Exception):
    """Base class for every error the application raises deliberately.

    Subclasses override the three class attributes to describe themselves; the
    API layer needs no knowledge of the specific subclass to serialise them.
    """

    code: str = "INTERNAL_SERVER_ERROR"
    status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR
    message: str = "An unexpected error occurred."

    def __init__(
        self,
        message: str | None = None,
        *,
        code: str | None = None,
        status_code: int | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.message = message or type(self).message
        self.code = code or type(self).code
        self.status_code = status_code or type(self).status_code
        self.details: dict[str, Any] | None = dict(details) if details else None
        super().__init__(self.message)


class ServiceUnavailableError(WeatherGPTError):
    """A dependency the service needs is not currently usable."""

    code = "SERVICE_UNAVAILABLE"
    status_code = HTTPStatus.SERVICE_UNAVAILABLE
    message = "The service is not ready to handle requests."


class WeatherProviderError(WeatherGPTError):
    """An upstream meteorological source failed, timed out or misbehaved.

    Carries the provider id so the ingestion pipeline can log which source
    failed and move on to the next one in the fallback chain.
    """

    code = "WEATHER_PROVIDER_ERROR"
    status_code = HTTPStatus.BAD_GATEWAY
    message = "An upstream weather provider failed."

    def __init__(
        self,
        message: str | None = None,
        *,
        provider_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.provider_id = provider_id
        details = dict(kwargs.pop("details", None) or {})
        if provider_id:
            details.setdefault("provider", provider_id)
        super().__init__(message, details=details or None, **kwargs)


class WeatherProviderTimeoutError(WeatherProviderError):
    """An upstream source did not answer inside its deadline."""

    code = "WEATHER_PROVIDER_TIMEOUT"
    status_code = HTTPStatus.GATEWAY_TIMEOUT
    message = "An upstream weather provider timed out."


class WeatherDataValidationError(WeatherGPTError):
    """Data was retrieved but failed meteorological validation.

    Raised by the validation engine when a record is physically implausible or
    internally inconsistent. Such data is rejected rather than served: a wrong
    number is worse than no number in a decision-support system.
    """

    code = "WEATHER_DATA_VALIDATION_FAILED"
    status_code = HTTPStatus.BAD_GATEWAY
    message = "Weather data failed validation and was rejected."


class WeatherDataUnavailableError(WeatherGPTError):
    """No configured source could supply usable data for the request."""

    code = "WEATHER_DATA_UNAVAILABLE"
    status_code = HTTPStatus.SERVICE_UNAVAILABLE
    message = "Weather data is temporarily unavailable."


class ForecastUnavailableError(WeatherDataUnavailableError):
    """No configured source could supply a forecast for the request."""

    code = "FORECAST_UNAVAILABLE"
    message = "Forecast data is temporarily unavailable."


class LocationNotFoundError(WeatherGPTError):
    """A place name could not be resolved to coordinates."""

    code = "LOCATION_NOT_FOUND"
    status_code = HTTPStatus.NOT_FOUND
    message = "The requested location could not be resolved."


class DatabaseUnavailableError(WeatherGPTError):
    """Persistence could not be reached, or is not configured.

    Deliberately carries no driver detail. SQLAlchemy embeds the failing
    statement — and depending on the driver, its parameters and the connection
    string — in its exception messages; the real cause is logged server-side
    under the request's correlation id instead.
    """

    code = "DATABASE_UNAVAILABLE"
    status_code = HTTPStatus.SERVICE_UNAVAILABLE
    message = "Weather persistence is temporarily unavailable."


class ProviderNotFoundError(WeatherGPTError):
    """A caller asked for a provider that is not registered."""

    code = "PROVIDER_NOT_FOUND"
    status_code = HTTPStatus.BAD_REQUEST
    message = "The requested weather provider is not available."


class ErrorDetail(BaseModel):
    """Machine-readable description of a single failure."""

    code: str = Field(
        description="Stable, screaming-snake-case identifier for the failure.",
        examples=["SERVICE_UNAVAILABLE"],
    )
    message: str = Field(
        description="Human-readable explanation. Safe to surface to end users.",
        examples=["The service is not ready to handle requests."],
    )
    request_id: str = Field(
        description="Correlation id, also returned in the X-Request-ID header.",
        examples=["3f1a9c2e4b7d4f0a8c1e6d5b2a90f7c3"],
    )
    details: dict[str, Any] | None = Field(
        default=None,
        description="Optional structured context, such as field-level errors.",
    )


class ErrorResponse(BaseModel):
    """Envelope returned for every non-2xx response."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": {
                    "code": "SERVICE_UNAVAILABLE",
                    "message": "The service is not ready to handle requests.",
                    "request_id": "3f1a9c2e4b7d4f0a8c1e6d5b2a90f7c3",
                }
            }
        }
    )

    error: ErrorDetail


def _request_id(request: Request) -> str:
    """Best-effort correlation id: request state first, context var second."""
    return (
        getattr(request.state, "request_id", None)
        or request_id_ctx.get()
        or NO_REQUEST_ID
    )


def _error_response(
    request: Request,
    *,
    status_code: int,
    code: str,
    message: str,
    details: Mapping[str, Any] | None = None,
) -> JSONResponse:
    request_id = _request_id(request)
    payload = ErrorResponse(
        error=ErrorDetail(
            code=code,
            message=message,
            request_id=request_id,
            details=dict(details) if details else None,
        )
    )
    return JSONResponse(
        status_code=status_code,
        content=payload.model_dump(mode="json", exclude_none=True),
        headers={REQUEST_ID_HEADER: request_id},
    )


def _code_for_status(status_code: int) -> str:
    """Derive a stable error code from an HTTP status (404 -> NOT_FOUND)."""
    try:
        return HTTPStatus(status_code).name
    except ValueError:
        return "HTTP_ERROR"


async def _handle_application_error(
    request: Request, exc: WeatherGPTError
) -> JSONResponse:
    logger.warning(
        "request.application_error",
        extra={
            "error_code": exc.code,
            "status_code": exc.status_code,
            "path": request.url.path,
        },
    )
    return _error_response(
        request,
        status_code=exc.status_code,
        code=exc.code,
        message=exc.message,
        details=exc.details,
    )


async def _handle_http_exception(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    return _error_response(
        request,
        status_code=exc.status_code,
        code=_code_for_status(exc.status_code),
        message=detail,
    )


async def _handle_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return _error_response(
        request,
        status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
        code="VALIDATION_ERROR",
        message="The request payload failed validation.",
        details={"errors": jsonable_encoder(exc.errors())},
    )


async def _handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(
        "request.unhandled_error",
        extra={"path": request.url.path, "error_type": type(exc).__name__},
    )
    return _error_response(
        request,
        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        code="INTERNAL_SERVER_ERROR",
        message="An unexpected error occurred.",
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Wire the error contract onto an application instance."""
    app.add_exception_handler(WeatherGPTError, _handle_application_error)  # type: ignore[arg-type]
    app.add_exception_handler(StarletteHTTPException, _handle_http_exception)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, _handle_validation_error)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, _handle_unexpected_error)
