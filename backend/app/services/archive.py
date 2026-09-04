"""Past weather that was never stored here.

The database holds what this deployment happened to observe, which on a fresh
install is nothing. "Stored observations: 0 returned" is a true statement about
our storage and a useless answer to "how much rain fell yesterday?", so a
question the database cannot answer falls through to the source that can.

Nothing about the guarantee changes on the way: the archive is read through the
same provider interface, normalised by the same normaliser, and stamped with
the same provenance as a live request. An archived value is a retrieved value,
never an interpolated one — a gap in the record stays a gap.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone

from app.config.logging import get_logger
from app.core.exceptions import WeatherProviderError
from app.domain.forecast import HourlyForecastPoint
from app.domain.location import Coordinates
from app.domain.weather import CurrentWeather
from app.providers.base import ProviderCapability
from app.providers.registry import ProviderRegistry
from app.services.history import HistoricalRecord, HistoryQuery

logger = get_logger(__name__)

#: How far back the archive can be asked to reach in one request. Open-Meteo
#: serves 92 days of recent past on the forecast endpoint, which covers every
#: question a conversation realistically asks.
MAX_PAST_DAYS = 92


def _as_observation(point: HourlyForecastPoint) -> CurrentWeather:
    """One archived hour, read as the measurement record it is.

    Every field is copied across; none is derived. An hour the source did not
    report stays missing rather than becoming a zero.
    """
    return CurrentWeather(
        observed_at=point.valid_at,
        temperature_c=point.temperature_c,
        apparent_temperature_c=point.apparent_temperature_c,
        dew_point_c=point.dew_point_c,
        relative_humidity_pct=point.relative_humidity_pct,
        pressure_msl_hpa=point.pressure_msl_hpa,
        wind_speed_ms=point.wind_speed_ms,
        wind_gust_ms=point.wind_gust_ms,
        wind_direction_deg=point.wind_direction_deg,
        precipitation_mm=point.precipitation_mm,
        cloud_cover_pct=point.cloud_cover_pct,
        visibility_m=point.visibility_m,
        uv_index=point.uv_index,
        condition=point.condition,
        condition_description=point.condition_description,
        wmo_code=point.wmo_code,
        is_day=point.is_day,
    )


class ArchiveReader:
    """Reads recent past weather from a provider that keeps it."""

    def __init__(self, registry: ProviderRegistry, *, max_past_days: int = MAX_PAST_DAYS) -> None:
        self._registry = registry
        self._max_past_days = max_past_days

    @property
    def enabled(self) -> bool:
        return bool(self._registry.for_capability(ProviderCapability.HISTORICAL))

    def _providers(self, latitude: float, longitude: float):
        # The registry already drops sources that do not cover the point and
        # puts a regional service ahead of a global model.
        return self._registry.for_capability(
            ProviderCapability.HISTORICAL, latitude=latitude, longitude=longitude
        )

    async def observations(self, query: HistoryQuery) -> list[HistoricalRecord]:
        """Archived hourly observations inside the query window.

        Returns an empty list when no registered source keeps history for that
        point, or when the window is older than the archive reaches. An empty
        result is "we do not have it", never a filled-in guess.

        Raises:
            WeatherProviderError: every capable source failed.
        """
        providers = self._providers(query.latitude, query.longitude)
        if not providers:
            return []

        now = datetime.now(timezone.utc)
        # One extra day either way: a local day can start before the UTC one.
        reach_days = math.ceil((now - query.start).total_seconds() / 86_400.0) + 2
        if reach_days > self._max_past_days:
            logger.info(
                "archive.window_too_old",
                extra={"requested_days": reach_days, "max_days": self._max_past_days},
            )
            return []

        coordinates = Coordinates(latitude=query.latitude, longitude=query.longitude)
        failures: list[Exception] = []
        for provider in providers:
            if query.provider_id and provider.metadata.provider_id != query.provider_id:
                continue
            try:
                archive = await provider.fetch_archive(
                    coordinates, past_days=max(1, min(reach_days, self._max_past_days))
                )
            except WeatherProviderError as error:  # try the next capable source
                failures.append(error)
                logger.warning(
                    "archive.provider_failed",
                    extra={"provider": provider.metadata.provider_id},
                )
                continue

            offset = timedelta(seconds=archive.utc_offset_seconds or 0)
            records = [
                HistoricalRecord(
                    weather=_as_observation(point),
                    provenance=archive.provenance,
                    latitude=archive.location.coordinates.latitude,
                    longitude=archive.location.coordinates.longitude,
                    distance_m=0.0,
                )
                for point in archive.hourly
                if _inside(point.valid_at, query, offset)
            ]
            logger.info(
                "archive.observations",
                extra={
                    "provider": provider.metadata.provider_id,
                    "location": f"{query.latitude:.4f},{query.longitude:.4f}",
                    "results": len(records),
                },
            )
            return records

        if failures:
            raise failures[0]
        return []


def _inside(moment: datetime, query: HistoryQuery, offset: timedelta = timedelta(0)) -> bool:
    """Whether an archived hour falls in the window that was asked for.

    A window written as calendar dates is judged against the *location's* own
    day. Instants are stored in UTC, so the offset is applied here rather than
    anywhere the value is displayed: a UTC midnight is the wrong midnight for
    someone in India asking about yesterday, and using it would answer with
    five and a half hours of the wrong day at each end.
    """
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)

    if query.calendar_start is not None and query.calendar_end is not None:
        local = moment + offset
        if not (query.calendar_start <= local.date() <= query.calendar_end):
            return False
        if query.local_hours is not None:
            start_hour, end_hour = query.local_hours
            if not (start_hour <= local.hour < end_hour):
                return False
        return True

    return query.start <= moment <= query.end


def default_window(days_back: int = 1) -> tuple[date, date]:
    """The calendar window "that many days ago", for callers without one."""
    today = datetime.now(timezone.utc).date()
    day = today - timedelta(days=days_back)
    return day, day
