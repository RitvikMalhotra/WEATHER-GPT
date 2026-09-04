"""Environment-driven application configuration.

Every runtime knob lives here so the same artifact can be promoted across
environments without a code change. Application code must depend on
:func:`get_settings` (directly, or via the FastAPI dependency in
``app.core.dependencies``) instead of reading ``os.environ`` ad hoc.

Later phases will extend this module with provider credentials, database URLs
and cache endpoints. Those values belong in the environment, never in source.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_VALID_LOG_LEVELS = frozenset(
    {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
)


class Environment(str, Enum):
    """Deployment environment the process believes it is running in."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogFormat(str, Enum):
    """Log rendering strategy.

    ``console`` is human readable for local work; ``json`` emits one structured
    object per line for log aggregation in deployed environments.
    """

    CONSOLE = "console"
    JSON = "json"


class Settings(BaseSettings):
    """Validated application settings.

    Values are read from the process environment first and fall back to a local
    ``.env`` file, then to the defaults below. The defaults are deliberately
    complete so a fresh checkout runs with zero configuration.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Identity -----------------------------------------------------------
    APP_NAME: str = "WeatherGPT API"
    APP_VERSION: str = "1.0.0"
    SERVICE_NAME: str = "weathergpt-backend"

    # --- Runtime ------------------------------------------------------------
    ENVIRONMENT: Environment = Environment.DEVELOPMENT
    DEBUG: bool = True

    # --- HTTP ---------------------------------------------------------------
    API_V1_PREFIX: str = "/api/v1"

    # --- Observability ------------------------------------------------------
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: LogFormat = LogFormat.CONSOLE

    # --- Meteorological providers -------------------------------------------
    #: Promoted to the front of the fallback chain.
    DEFAULT_PROVIDER: str = "open-meteo"
    #: Per-request deadline for an upstream call, in seconds.
    PROVIDER_TIMEOUT_SECONDS: float = 8.0
    #: Extra attempts after the first, for transient upstream failures.
    PROVIDER_MAX_RETRIES: int = 2
    #: Identifies this service to upstream APIs, as fair-use policies expect.
    HTTP_USER_AGENT: str = "WeatherGPT/1.0 (+https://github.com/weathergpt)"

    OPEN_METEO_FORECAST_URL: str = "https://api.open-meteo.com/v1/forecast"
    OPEN_METEO_GEOCODING_URL: str = "https://geocoding-api.open-meteo.com/v1/search"

    # --- Response caching ---------------------------------------------------
    #: Seconds to retain a current-conditions response. 0 disables the cache.
    CURRENT_CACHE_TTL_SECONDS: float = 300.0
    #: Seconds to retain a forecast response.
    FORECAST_CACHE_TTL_SECONDS: float = 900.0
    #: Seconds to retain a geocoding result. Places rarely move.
    GEOCODING_CACHE_TTL_SECONDS: float = 86_400.0

    # --- Persistence --------------------------------------------------------
    #: SQLAlchemy async URL, e.g.
    #: postgresql+asyncpg://weathergpt:weathergpt@localhost:5432/weathergpt
    #: Unset means the service runs without persistence: current conditions and
    #: forecasts still work, historical queries return 503.
    DATABASE_URL: str | None = None
    #: Escape hatch to run against a configured database with writes disabled.
    PERSISTENCE_ENABLED: bool = True
    DATABASE_ECHO: bool = False
    DATABASE_POOL_SIZE: int = 5
    DATABASE_MAX_OVERFLOW: int = 10
    DATABASE_POOL_TIMEOUT_SECONDS: float = 10.0
    #: Window used to group forecast points into one generation. See
    #: app.services.persistence.generation_bucket for why this exists.
    FORECAST_GENERATION_BUCKET_MINUTES: int = 60

    # --- Historical queries -------------------------------------------------
    #: Default radius for the historical search, in kilometres.
    HISTORY_DEFAULT_RADIUS_KM: float = 25.0
    HISTORY_MAX_RADIUS_KM: float = 500.0
    #: Longest window a single historical request may span, in days.
    HISTORY_MAX_RANGE_DAYS: int = 366
    HISTORY_MAX_RESULTS: int = 1000

    # --- Alert engine -------------------------------------------------------
    # Every threshold below is an ENGINEERING DEFAULT for a prototype. None of
    # them is an official safety threshold, and an alert produced by them is a
    # WeatherGPT rule result — never an official meteorological warning. A
    # deployment used for real decisions must set its own values from a sourced
    # specification. See app/alerts/rules.py and the backend README.
    ALERT_EVALUATION_ENABLED: bool = True
    #: How long a current-conditions reading is treated as describing the
    #: present. Also an observed alert's validity window, so a condition that
    #: stops being reported ages out instead of lingering as active.
    ALERT_OBSERVATION_VALIDITY_MINUTES: int = 60
    #: Forecast points beyond this horizon are not evaluated.
    ALERT_FORECAST_LOOKAHEAD_HOURS: int = 48
    #: Cap on alerts written from one forecast evaluation.
    ALERT_MAX_PER_EVALUATION: int = 200

    # Rainfall accumulated within one forecast hour, in millimetres.
    HEAVY_RAINFALL_THRESHOLD: float = 20.0
    HEAVY_RAINFALL_WARNING_THRESHOLD: float = 50.0
    HEAVY_RAINFALL_SEVERE_THRESHOLD: float = 100.0

    # Rainfall accumulated across one forecast day, in millimetres. Set at the
    # boundaries of commonly published 24-hour rainfall categories so the
    # numbers are familiar; reusing a number does not borrow the authority of
    # whoever published it.
    HEAVY_RAINFALL_DAILY_THRESHOLD: float = 64.5
    HEAVY_RAINFALL_DAILY_WARNING_THRESHOLD: float = 115.6
    HEAVY_RAINFALL_DAILY_SEVERE_THRESHOLD: float = 204.5
    HEAVY_RAINFALL_DAILY_EXTREME_THRESHOLD: float = 350.0

    # Dry-bulb air temperature, in degrees Celsius. A temperature threshold
    # alone is NOT a heat-health model: humidity, acclimatisation, exposure and
    # duration all matter. Humidity and apparent temperature are recorded as
    # evidence context so a later rule can use them properly.
    EXTREME_HEAT_THRESHOLD: float = 38.0
    EXTREME_HEAT_WARNING_THRESHOLD: float = 42.0
    EXTREME_HEAT_SEVERE_THRESHOLD: float = 46.0

    # Sustained wind at 10 m, in metres per second (~50, 75 and 100 km/h).
    HIGH_WIND_THRESHOLD: float = 13.9
    HIGH_WIND_WARNING_THRESHOLD: float = 20.8
    HIGH_WIND_SEVERE_THRESHOLD: float = 27.8

    # Peak gust at 10 m, in metres per second (~65, 90 and 120 km/h).
    HIGH_WIND_GUST_THRESHOLD: float = 18.0
    HIGH_WIND_GUST_WARNING_THRESHOLD: float = 25.0
    HIGH_WIND_GUST_SEVERE_THRESHOLD: float = 33.3

    # Forecast probability of measurable precipitation, as a percentage.
    SEVERE_PRECIPITATION_PROBABILITY: float = 70.0
    SEVERE_PRECIPITATION_PROBABILITY_WARNING: float = 90.0

    # --- Alert queries ------------------------------------------------------
    ALERT_DEFAULT_RADIUS_KM: float = 25.0
    ALERT_MAX_RADIUS_KM: float = 500.0
    ALERT_MAX_RESULTS: int = 200

    # --- Forecast and validation limits -------------------------------------
    #: Upper bound a caller may request, independent of provider capability.
    MAX_FORECAST_DAYS: int = 16
    DEFAULT_FORECAST_DAYS: int = 7
    #: An observation older than this is flagged as stale by the validator.
    MAX_OBSERVATION_AGE_MINUTES: int = 180
    #: How far ahead of us an observation timestamp may sit before it is rejected.
    MAX_CLOCK_SKEW_MINUTES: int = 90

    @field_validator("API_V1_PREFIX")
    @classmethod
    def _normalise_prefix(cls, value: str) -> str:
        """Accept ``v1``, ``/api/v1`` or ``/api/v1/`` and normalise the shape."""
        prefix = value.strip().strip("/")
        if not prefix:
            raise ValueError("API_V1_PREFIX must not be empty")
        return f"/{prefix}"

    @field_validator("LOG_LEVEL")
    @classmethod
    def _normalise_log_level(cls, value: str) -> str:
        level = value.strip().upper()
        if level not in _VALID_LOG_LEVELS:
            raise ValueError(
                f"LOG_LEVEL must be one of {sorted(_VALID_LOG_LEVELS)}, got {value!r}"
            )
        return level

    @field_validator("DEFAULT_FORECAST_DAYS")
    @classmethod
    def _validate_forecast_days(cls, value: int) -> int:
        if value < 1:
            raise ValueError("DEFAULT_FORECAST_DAYS must be at least 1")
        return value

    @field_validator("DATABASE_URL")
    @classmethod
    def _require_async_driver(cls, value: str | None) -> str | None:
        """Reject a synchronous driver URL rather than failing at first query."""
        if value is None or not value.strip():
            return None
        url = value.strip()
        if "+" not in url.split("://", 1)[0]:
            raise ValueError(
                "DATABASE_URL must name an async driver, e.g. "
                "postgresql+asyncpg://user:password@host:5432/database"
            )
        return url

    @property
    def persistence_active(self) -> bool:
        """True when writes and historical queries have somewhere to go."""
        return bool(self.DATABASE_URL) and self.PERSISTENCE_ENABLED

    @property
    def is_production(self) -> bool:
        """True when the process is running in the production environment."""
        return self.ENVIRONMENT is Environment.PRODUCTION


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so configuration is parsed and validated exactly once. Tests that
    need to vary configuration should call ``get_settings.cache_clear()`` or
    override the FastAPI dependency.
    """
    return Settings()
