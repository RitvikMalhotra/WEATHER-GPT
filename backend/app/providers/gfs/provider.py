"""NOAA GFS, the numerical weather prediction model, as a first-class provider.

This is real GFS output: NOAA's Global Forecast System, run four times a day at
00/06/12/18 UTC, served here through Open-Meteo's dedicated ``/v1/gfs``
endpoint. That endpoint returns GFS alone — not the blended "best available
model" the general endpoint selects — so a response from this provider is
attributable to GFS and to a specific model cycle.

Why the hosted endpoint rather than NOMADS: the authoritative alternative is
GRIB2 files off NOMADS, which means a GRIB decoder, a subsetting step and
gigabytes of global grids to extract one point. The requirement is
point-extracted NWP values with honest provenance, and this delivers exactly
that with no grid storage. Swapping the transport later changes this module
only, because everything downstream sees the canonical model.

The wire format is identical to the general Open-Meteo endpoint, so the client
and normaliser are reused rather than duplicated. What differs is provenance:
this provider also reads the model's metadata document to record which GFS run
produced the numbers.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.config.logging import get_logger
from app.domain.forecast import Forecast
from app.domain.location import Coordinates
from app.domain.provenance import DataProvenance
from app.domain.weather import WeatherReport
from app.providers.base import ProviderCapability, ProviderMetadata, WeatherProvider
from app.providers.http import UpstreamHttpClient
from app.providers.open_meteo.client import OpenMeteoClient
from app.providers.open_meteo.normalizer import normalize_current, normalize_forecast

logger = get_logger(__name__)

GFS_PROVIDER_ID = "noaa-gfs"

METADATA = ProviderMetadata(
    provider_id=GFS_PROVIDER_ID,
    name="NOAA GFS",
    capabilities=frozenset(
        {
            ProviderCapability.CURRENT,
            ProviderCapability.HOURLY_FORECAST,
            ProviderCapability.DAILY_FORECAST,
        }
    ),
    source_url="https://www.nco.ncep.noaa.gov/pmb/products/gfs/",
    license="Public domain (U.S. Government work)",
    attribution="NOAA NCEP Global Forecast System (GFS) via Open-Meteo",
    model="gfs_seamless",
    # Sorts after Open-Meteo's blend, which is generally better for a single
    # point. GFS is selected explicitly, by name, when a caller wants the raw
    # numerical model rather than a blend.
    priority=20,
    max_forecast_days=16,
    notes=(
        "NOAA's Global Forecast System: a physics-based numerical weather "
        "prediction model run at 00/06/12/18 UTC.",
        "Values are point extractions from the GFS grid, not a station "
        "observation.",
        "Responses carry the initialisation time of the GFS cycle behind them.",
    ),
)

#: How long a fetched run time is trusted before it is looked up again. GFS
#: publishes every six hours, so a few minutes of staleness is harmless, and
#: this keeps one metadata request from riding on every forecast.
_RUN_METADATA_TTL = timedelta(minutes=10)


class GfsProvider(WeatherProvider):
    """Point forecasts extracted from the NOAA GFS model."""

    def __init__(
        self,
        client: OpenMeteoClient,
        *,
        metadata_http: UpstreamHttpClient | None = None,
        metadata_url: str | None = None,
    ) -> None:
        self._client = client
        self._metadata_http = metadata_http
        self._metadata_url = metadata_url
        self._run_at: datetime | None = None
        self._run_checked_at: datetime | None = None

    @property
    def metadata(self) -> ProviderMetadata:
        return METADATA

    async def fetch_current(self, coordinates: Coordinates) -> WeatherReport:
        payload = await self._client.fetch(
            latitude=coordinates.latitude,
            longitude=coordinates.longitude,
            current=True,
        )
        return normalize_current(payload, provenance=await self._provenance())

    async def fetch_forecast(
        self,
        coordinates: Coordinates,
        *,
        days: int,
        include_hourly: bool,
        include_daily: bool,
    ) -> Forecast:
        payload = await self._client.fetch(
            latitude=coordinates.latitude,
            longitude=coordinates.longitude,
            hourly=include_hourly,
            daily=include_daily,
            forecast_days=min(days, METADATA.max_forecast_days),
        )
        return normalize_forecast(payload, provenance=await self._provenance())

    async def _provenance(self) -> DataProvenance:
        return DataProvenance(
            provider_id=METADATA.provider_id,
            provider_name=METADATA.name,
            model=METADATA.model,
            fetched_at=datetime.now(timezone.utc),
            model_run_at=await self._model_run_at(),
            source_url=self._client.forecast_url,
            license=METADATA.license,
            attribution=METADATA.attribution,
        )

    async def _model_run_at(self) -> datetime | None:
        """Initialisation time of the GFS cycle behind the current data.

        Returns None rather than guessing. The cycle time could be derived from
        the clock — GFS runs on a fixed six-hour schedule — but a derived value
        would be indistinguishable from a real one while being wrong whenever
        publication is late, which is exactly when it matters.
        """
        if self._metadata_http is None or self._metadata_url is None:
            return None

        now = datetime.now(timezone.utc)
        if (
            self._run_at is not None
            and self._run_checked_at is not None
            and now - self._run_checked_at < _RUN_METADATA_TTL
        ):
            return self._run_at

        try:
            body = await self._metadata_http.get_json(self._metadata_url)
            initialised = body.get("last_run_initialisation_time")
            if isinstance(initialised, (int, float)):
                self._run_at = datetime.fromtimestamp(float(initialised), tz=timezone.utc)
                self._run_checked_at = now
        except Exception:
            # Run time is provenance detail, not the forecast. Losing it must
            # never cost the caller their data.
            logger.warning("gfs.run_metadata_unavailable", extra={"url": self._metadata_url})
            self._run_checked_at = now

        return self._run_at
