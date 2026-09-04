"""Integration tests against a real PostgreSQL/PostGIS.

These are the tests that cannot honestly be faked: ``ST_DWithin`` semantics,
``ON CONFLICT`` behaviour against a live constraint, and whether the migration
actually produces a working schema. A SQLite stand-in would prove none of it.

They are skipped unless ``TEST_DATABASE_URL`` points at a PostGIS server, so the
default suite stays offline and infrastructure-free::

    docker compose up -d --wait postgres
    TEST_DATABASE_URL=postgresql+asyncpg://weathergpt:weathergpt@localhost:5432/postgres \\
        pytest -m integration

Each run creates its own throwaway database, migrates it with Alembic, and drops
it afterwards — so the suite also proves that a fresh database can be built from
migrations alone.
"""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.db.engine import Database
from app.db.repositories import ForecastRepository, ObservationRepository
from app.domain.forecast import DailyForecastPoint, Forecast, HourlyForecastPoint
from app.domain.location import Coordinates, Location
from app.domain.provenance import DataProvenance
from app.domain.weather import CurrentWeather, WeatherCondition, WeatherReport
from app.services.history import HistoryQuery, HistoryService
from app.services.persistence import PersistenceService

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("TEST_DATABASE_URL"),
        reason="set TEST_DATABASE_URL to a PostGIS server to run integration tests",
    ),
]

BACKEND_ROOT = Path(__file__).resolve().parents[1]

NOW = datetime(2026, 9, 4, 7, 30, tzinfo=timezone.utc)
# Hyderabad, and three points at known distances from it.
HYDERABAD = Coordinates(latitude=17.385, longitude=78.4867, elevation_m=505.0)
NEARBY = Coordinates(latitude=17.4400, longitude=78.4867)  # ~6.1 km north
FAR_AWAY = Coordinates(latitude=28.6139, longitude=77.2090)  # Delhi, ~1250 km


# --- Throwaway database ------------------------------------------------------


def _admin_url() -> str:
    return os.environ["TEST_DATABASE_URL"]


def _swap_database(url: str, name: str) -> str:
    base, _, _ = url.rpartition("/")
    return f"{base}/{name}"


@pytest.fixture(scope="session")
def migrated_database_url() -> str:
    """Create a fresh database, migrate it with Alembic, drop it afterwards.

    Running the real migration rather than ``metadata.create_all`` is the point:
    it verifies that a brand-new database can be built from the migration chain,
    PostGIS extension included.
    """
    name = f"weathergpt_test_{uuid.uuid4().hex[:12]}"
    admin = _admin_url()
    target = _swap_database(admin, name)

    import asyncio

    async def _run_ddl(statement: str) -> None:
        engine = create_async_engine(admin, isolation_level="AUTOCOMMIT")
        try:
            async with engine.connect() as connection:
                await connection.execute(text(statement))
        finally:
            await engine.dispose()

    asyncio.run(_run_ddl(f'CREATE DATABASE "{name}"'))
    try:
        result = subprocess.run(
            [sys.executable, "-m", "alembic", "-x", f"url={target}", "upgrade", "head"],
            cwd=BACKEND_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"alembic upgrade head failed:\n{result.stdout}\n{result.stderr}"
        )
        yield target
    finally:
        asyncio.run(
            _run_ddl(
                f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'
            )
        )


@pytest.fixture
async def database(migrated_database_url: str):
    """A Database on the migrated schema, emptied before each test."""
    db = Database(migrated_database_url, pool_size=2, max_overflow=2)
    async with db.session() as session:
        await session.execute(
            text("TRUNCATE weather_observations, weather_forecasts")
        )
    try:
        yield db
    finally:
        await db.dispose()


# --- Fixtures ----------------------------------------------------------------


def _provenance(**overrides) -> DataProvenance:
    return DataProvenance(
        **{
            "provider_id": "open-meteo",
            "provider_name": "Open-Meteo",
            "model": "best_match",
            "fetched_at": NOW,
            "source_url": "https://api.open-meteo.com/v1/forecast",
            "license": "CC-BY-4.0",
            "attribution": "Weather data by Open-Meteo.com",
            **overrides,
        }
    )


def _report(
    *,
    coordinates: Coordinates = HYDERABAD,
    observed_at: datetime = NOW,
    temperature_c: float = 28.4,
    provenance: DataProvenance | None = None,
) -> WeatherReport:
    return WeatherReport(
        location=Location(coordinates=coordinates, timezone="Asia/Kolkata"),
        current=CurrentWeather(
            observed_at=observed_at,
            temperature_c=temperature_c,
            relative_humidity_pct=81.0,
            wind_speed_ms=3.2,
            precipitation_mm=0.4,
            condition=WeatherCondition.RAIN,
            condition_description="Moderate rain",
            wmo_code=63,
            is_day=True,
        ),
        provenance=provenance or _provenance(),
    )


def _forecast(*, fetched_at: datetime = NOW) -> Forecast:
    return Forecast(
        location=Location(coordinates=HYDERABAD, timezone="Asia/Kolkata"),
        hourly=[
            HourlyForecastPoint(
                valid_at=NOW + timedelta(hours=offset),
                temperature_c=28.0 + offset,
                precipitation_probability_pct=40.0,
                wind_speed_ms=3.0,
            )
            for offset in range(3)
        ],
        daily=[
            DailyForecastPoint(
                date=date(2026, 9, 4),
                temperature_min_c=24.1,
                temperature_max_c=31.8,
                precipitation_sum_mm=6.4,
                wind_speed_max_ms=7.0,
            )
        ],
        provenance=_provenance(fetched_at=fetched_at),
    )


async def _store(database: Database, *reports: WeatherReport) -> None:
    async with database.session() as session:
        repository = ObservationRepository(session)
        for report in reports:
            await repository.upsert(report)


# --- Schema ------------------------------------------------------------------


async def test_the_migration_produces_a_working_postgis_schema(database: Database):
    async with database.session() as session:
        version = await session.execute(text("SELECT postgis_version()"))
        tables = await session.execute(
            text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public'"
            )
        )
        geometry = await session.execute(
            text(
                "SELECT type, srid FROM geography_columns "
                "WHERE f_table_name = 'weather_observations'"
            )
        )

    assert version.scalar_one()
    names = {row[0] for row in tables}
    assert {"weather_observations", "weather_forecasts", "alembic_version"} <= names
    assert geometry.one() == ("Point", 4326)


async def test_the_expected_indexes_exist(database: Database):
    async with database.session() as session:
        result = await session.execute(
            text("SELECT indexname FROM pg_indexes WHERE schemaname = 'public'")
        )
    indexes = {row[0] for row in result}

    assert "ix_weather_observations_geom" in indexes
    assert "ix_weather_observations_location_key_observed_at" in indexes
    assert "ix_weather_forecasts_forecast_created_at_forecast_for" in indexes
    assert "uq_weather_observations_identity" in indexes


# --- Observation persistence -------------------------------------------------


async def test_a_valid_observation_is_persisted(database: Database):
    assert await PersistenceService(database).record_observation(_report()) is True

    async with database.session() as session:
        assert await ObservationRepository(session).count() == 1


async def test_provenance_survives_the_round_trip(database: Database):
    await _store(database, _report())

    async with database.session() as session:
        found = await ObservationRepository(session).find_nearest(
            latitude=HYDERABAD.latitude, longitude=HYDERABAD.longitude, radius_m=1000
        )

    row = found.observation
    assert row.provider_id == "open-meteo"
    assert row.provider_name == "Open-Meteo"
    assert row.model == "best_match"
    assert row.fetched_at == NOW
    assert row.license == "CC-BY-4.0"
    assert row.attribution == "Weather data by Open-Meteo.com"
    assert row.source_url == "https://api.open-meteo.com/v1/forecast"


async def test_measurements_and_condition_survive_the_round_trip(database: Database):
    await _store(database, _report())

    async with database.session() as session:
        found = await ObservationRepository(session).find_nearest(
            latitude=HYDERABAD.latitude, longitude=HYDERABAD.longitude, radius_m=1000
        )

    row = found.observation
    assert row.temperature_c == pytest.approx(28.4)
    assert row.relative_humidity_pct == pytest.approx(81.0)
    assert row.precipitation_mm == pytest.approx(0.4)
    assert row.condition == "rain"
    assert row.wmo_code == 63
    assert row.is_day is True
    assert row.elevation_m == pytest.approx(505.0)


async def test_an_undisclosed_model_is_stored_and_still_unique(database: Database):
    report = _report(provenance=_provenance(model=None))

    await _store(database, report, report)

    async with database.session() as session:
        assert await ObservationRepository(session).count() == 1


# --- Idempotency -------------------------------------------------------------


async def test_re_ingesting_the_same_observation_does_not_duplicate(database: Database):
    await _store(database, _report(), _report(), _report())

    async with database.session() as session:
        assert await ObservationRepository(session).count() == 1


async def test_re_ingestion_refreshes_the_stored_values(database: Database):
    await _store(database, _report(temperature_c=28.4))
    await _store(database, _report(temperature_c=29.9))

    async with database.session() as session:
        repository = ObservationRepository(session)
        assert await repository.count() == 1
        found = await repository.find_nearest(
            latitude=HYDERABAD.latitude, longitude=HYDERABAD.longitude, radius_m=1000
        )
    assert found.observation.temperature_c == pytest.approx(29.9)


async def test_a_different_instant_is_a_different_observation(database: Database):
    await _store(
        database,
        _report(observed_at=NOW),
        _report(observed_at=NOW + timedelta(hours=1)),
    )

    async with database.session() as session:
        assert await ObservationRepository(session).count() == 2


async def test_a_different_provider_is_a_different_observation(database: Database):
    await _store(
        database,
        _report(),
        _report(provenance=_provenance(provider_id="imd", provider_name="IMD")),
    )

    async with database.session() as session:
        assert await ObservationRepository(session).count() == 2


# --- Spatial queries ---------------------------------------------------------


async def test_a_nearby_observation_is_found(database: Database):
    await _store(database, _report(coordinates=NEARBY))

    async with database.session() as session:
        found = await ObservationRepository(session).find_in_range(
            latitude=HYDERABAD.latitude,
            longitude=HYDERABAD.longitude,
            start=NOW - timedelta(days=1),
            end=NOW + timedelta(days=1),
            radius_m=10_000,
        )

    assert len(found) == 1
    # ~6.1 km north; PostGIS measures on the spheroid.
    assert 5_500 < found[0].distance_m < 6_500


async def test_a_far_away_observation_is_excluded(database: Database):
    await _store(database, _report(coordinates=FAR_AWAY))

    async with database.session() as session:
        found = await ObservationRepository(session).find_in_range(
            latitude=HYDERABAD.latitude,
            longitude=HYDERABAD.longitude,
            start=NOW - timedelta(days=1),
            end=NOW + timedelta(days=1),
            radius_m=25_000,
        )

    assert found == []


async def test_the_radius_is_what_decides_inclusion(database: Database):
    """Same data, same query, different radius — the boundary is real."""
    await _store(database, _report(coordinates=NEARBY))
    window = {"start": NOW - timedelta(days=1), "end": NOW + timedelta(days=1)}

    async with database.session() as session:
        repository = ObservationRepository(session)
        tight = await repository.find_in_range(
            latitude=HYDERABAD.latitude,
            longitude=HYDERABAD.longitude,
            radius_m=3_000,
            **window,
        )
        wide = await repository.find_in_range(
            latitude=HYDERABAD.latitude,
            longitude=HYDERABAD.longitude,
            radius_m=20_000,
            **window,
        )

    assert tight == []
    assert len(wide) == 1


async def test_the_nearest_observation_is_returned_first(database: Database):
    await _store(
        database,
        _report(coordinates=NEARBY, temperature_c=20.0),
        _report(coordinates=HYDERABAD, temperature_c=30.0),
    )

    async with database.session() as session:
        nearest = await ObservationRepository(session).find_nearest(
            latitude=HYDERABAD.latitude, longitude=HYDERABAD.longitude, radius_m=50_000
        )

    assert nearest.observation.temperature_c == pytest.approx(30.0)
    assert nearest.distance_m < 100


async def test_the_spatial_query_uses_the_gist_index(database: Database):
    """Proves the work happens in PostGIS on an index, not row-by-row in Python."""
    await _store(database, _report())

    async with database.session() as session:
        # Without this the planner prefers a sequential scan on a tiny table.
        await session.execute(text("SET LOCAL enable_seqscan = off"))
        plan = await session.execute(
            text(
                "EXPLAIN SELECT id FROM weather_observations "
                "WHERE ST_DWithin(geom, ST_GeogFromText('SRID=4326;POINT(78.4867 17.385)'), 25000)"
            )
        )
    rendered = "\n".join(row[0] for row in plan)

    assert "ix_weather_observations_geom" in rendered, rendered


# --- Time filtering ----------------------------------------------------------


async def test_observations_outside_the_window_are_excluded(database: Database):
    await _store(
        database,
        _report(observed_at=NOW - timedelta(days=10)),
        _report(observed_at=NOW),
    )

    async with database.session() as session:
        found = await ObservationRepository(session).find_in_range(
            latitude=HYDERABAD.latitude,
            longitude=HYDERABAD.longitude,
            start=NOW - timedelta(hours=1),
            end=NOW + timedelta(hours=1),
            radius_m=25_000,
        )

    assert len(found) == 1
    assert found[0].observation.observed_at == NOW


async def test_a_provider_filter_narrows_the_result(database: Database):
    await _store(
        database,
        _report(),
        _report(provenance=_provenance(provider_id="imd", provider_name="IMD")),
    )

    async with database.session() as session:
        found = await ObservationRepository(session).find_in_range(
            latitude=HYDERABAD.latitude,
            longitude=HYDERABAD.longitude,
            start=NOW - timedelta(days=1),
            end=NOW + timedelta(days=1),
            radius_m=25_000,
            provider_id="imd",
        )

    assert [row.observation.provider_id for row in found] == ["imd"]


# --- Forecast persistence ----------------------------------------------------


async def test_a_valid_forecast_is_persisted(database: Database):
    written = await PersistenceService(database).record_forecast(_forecast())

    assert written == 4  # three hourly points plus one daily
    async with database.session() as session:
        assert await ForecastRepository(session).count() == 4


async def test_forecast_rows_record_both_timestamps(database: Database):
    await PersistenceService(database).record_forecast(_forecast())

    async with database.session() as session:
        rows = await ForecastRepository(session).find_for_location(
            location_key=HYDERABAD.cache_key,
            start=NOW - timedelta(days=1),
            end=NOW + timedelta(days=2),
            resolution="hourly",
        )

    assert rows
    # One generation, bucketed to the hour...
    assert {row.forecast_created_at for row in rows} == {NOW.replace(minute=0)}
    # ...covering three distinct target instants.
    assert {row.forecast_for for row in rows} == {
        NOW + timedelta(hours=offset) for offset in range(3)
    }


async def test_repeated_forecast_requests_do_not_duplicate(database: Database):
    service = PersistenceService(database)

    await service.record_forecast(_forecast(fetched_at=NOW))
    await service.record_forecast(_forecast(fetched_at=NOW + timedelta(minutes=20)))

    async with database.session() as session:
        assert await ForecastRepository(session).count() == 4


async def test_a_new_generation_appends_rather_than_overwriting(database: Database):
    """Keeping successive generations is what makes skill analysis possible."""
    service = PersistenceService(database)

    await service.record_forecast(_forecast(fetched_at=NOW))
    await service.record_forecast(_forecast(fetched_at=NOW + timedelta(hours=2)))

    async with database.session() as session:
        assert await ForecastRepository(session).count() == 8


async def test_forecast_provenance_is_preserved(database: Database):
    await PersistenceService(database).record_forecast(_forecast())

    async with database.session() as session:
        rows = await ForecastRepository(session).find_for_location(
            location_key=HYDERABAD.cache_key,
            start=NOW - timedelta(days=1),
            end=NOW + timedelta(days=2),
        )

    assert rows
    assert all(row.provider_id == "open-meteo" for row in rows)
    assert all(row.model == "best_match" for row in rows)
    assert all(row.attribution == "Weather data by Open-Meteo.com" for row in rows)


async def test_daily_and_hourly_points_are_stored_separately(database: Database):
    await PersistenceService(database).record_forecast(_forecast())

    async with database.session() as session:
        repository = ForecastRepository(session)
        window = {
            "location_key": HYDERABAD.cache_key,
            "start": NOW - timedelta(days=2),
            "end": NOW + timedelta(days=2),
        }
        hourly = await repository.find_for_location(**window, resolution="hourly")
        daily = await repository.find_for_location(**window, resolution="daily")

    assert len(hourly) == 3
    assert len(daily) == 1
    assert daily[0].temperature_max_c == pytest.approx(31.8)
    assert daily[0].precipitation_mm == pytest.approx(6.4)


# --- History service ---------------------------------------------------------


async def test_the_history_service_returns_canonical_records(database: Database):
    await _store(database, _report())

    records = await HistoryService(database).observations(
        HistoryQuery(
            latitude=HYDERABAD.latitude,
            longitude=HYDERABAD.longitude,
            start=NOW - timedelta(days=1),
            end=NOW + timedelta(days=1),
            radius_m=25_000,
        )
    )

    assert len(records) == 1
    record = records[0]
    assert record.weather.temperature_c == pytest.approx(28.4)
    assert record.weather.condition is WeatherCondition.RAIN
    assert record.provenance.provider_id == "open-meteo"
    assert record.provenance.model == "best_match"
    assert record.distance_m < 100


async def test_an_empty_history_window_returns_no_records(database: Database):
    await _store(database, _report())

    records = await HistoryService(database).observations(
        HistoryQuery(
            latitude=HYDERABAD.latitude,
            longitude=HYDERABAD.longitude,
            start=NOW + timedelta(days=30),
            end=NOW + timedelta(days=60),
            radius_m=25_000,
        )
    )

    assert records == []


async def test_a_database_outage_surfaces_as_a_structured_error(
    migrated_database_url: str,
):
    """A disposed engine stands in for an unreachable server."""
    from app.core.exceptions import DatabaseUnavailableError

    database = Database(_swap_database(migrated_database_url, "definitely_not_here"))
    try:
        with pytest.raises(DatabaseUnavailableError) as exc_info:
            await HistoryService(database).observations(
                HistoryQuery(
                    latitude=HYDERABAD.latitude,
                    longitude=HYDERABAD.longitude,
                    start=NOW - timedelta(days=1),
                    end=NOW,
                    radius_m=25_000,
                )
            )
    finally:
        await database.dispose()

    error = exc_info.value
    assert error.code == "DATABASE_UNAVAILABLE"
    assert "definitely_not_here" not in f"{error.message} {error.details}"


async def test_ping_reports_reachability(database: Database, migrated_database_url: str):
    assert await database.ping() is True

    unreachable = Database(_swap_database(migrated_database_url, "definitely_not_here"))
    try:
        assert await unreachable.ping() is False
    finally:
        await unreachable.dispose()


# --- Transactions ------------------------------------------------------------


async def test_a_failed_transaction_is_rolled_back(database: Database):
    with pytest.raises(RuntimeError):
        async with database.session() as session:
            await ObservationRepository(session).upsert(_report())
            raise RuntimeError("something went wrong after the write")

    async with database.session() as session:
        assert await ObservationRepository(session).count() == 0


async def test_a_successful_transaction_is_committed(database: Database):
    async with database.session() as session:
        await ObservationRepository(session).upsert(_report())

    async with database.session() as session:
        assert await ObservationRepository(session).count() == 1
