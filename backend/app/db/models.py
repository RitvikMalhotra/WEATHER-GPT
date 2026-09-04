"""Persisted meteorological records.

Two tables, shaped around the questions WeatherGPT actually asks rather than a
generic key/value store:

``weather_observations``
    What were the conditions at this place at this time? One row per
    (source, model, place, instant).

``weather_forecasts``
    What did this source predict, at this time, for that time? One row per
    (source, model, place, resolution, generation, target). The separation of
    ``forecast_created_at`` from ``forecast_for`` is what makes forecast-skill
    analysis possible later — you cannot ask "how good was last Tuesday's
    three-day forecast" if you only keep the newest prediction.

Design notes that apply to both:

* **Coordinates are stored twice, deliberately.** ``latitude``/``longitude`` are
  the values we return to callers; ``geom`` is a PostGIS geography used only for
  spatial predicates. Keeping the plain columns means reads never need to parse
  geometry, and keeping the geography means distance queries run in the database
  on a GIST index rather than over every row in Python.
* **``location_key`` is the identity of a place.** Floating-point coordinates
  make a poor unique key, so a place is identified by its coordinates rounded to
  four decimals (~11 m) — the same key the response cache uses.
* **Provenance travels with the row.** Every record carries the source, model
  and fetch time that produced it. A measurement without that context is not
  usable for decision support.
* **Missing stays missing.** Every measurement is nullable, matching the
  canonical model: an unreported value must not become a zero.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from geoalchemy2 import Geography
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    SmallInteger,
    String,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base

#: WGS-84. The SRID every coordinate in this system uses.
SRID = 4326

#: Sentinel stored when a source does not disclose its model. A NULL here would
#: silently disable the uniqueness constraints, because NULLs never compare
#: equal — so "not disclosed" is spelled explicitly and mapped back to None.
UNDISCLOSED_MODEL = ""


class _SpatialRecord:
    """Columns shared by every geolocated meteorological record."""

    id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    # --- Place --------------------------------------------------------------
    location_key: Mapped[str] = mapped_column(String(32), nullable=False)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    elevation_m: Mapped[float | None] = mapped_column(Float)
    # spatial_index=False: the GIST index is declared explicitly per table so
    # its name is under the naming convention and Alembic can manage it.
    geom: Mapped[object] = mapped_column(
        Geography(geometry_type="POINT", srid=SRID, spatial_index=False),
        nullable=False,
    )
    timezone: Mapped[str | None] = mapped_column(String(64))

    # --- Provenance ---------------------------------------------------------
    provider_id: Mapped[str] = mapped_column(String(64), nullable=False)
    provider_name: Mapped[str] = mapped_column(String(128), nullable=False)
    model: Mapped[str] = mapped_column(
        String(64), nullable=False, server_default=UNDISCLOSED_MODEL
    )
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    #: Initialisation time of the NWP run behind the values. Null for sources
    #: that do not disclose one; never inferred from the fetch time.
    model_run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    source_url: Mapped[str | None] = mapped_column(String(512))
    license: Mapped[str | None] = mapped_column(String(128))
    attribution: Mapped[str | None] = mapped_column(String(256))

    # --- Bookkeeping --------------------------------------------------------
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # --- Condition ----------------------------------------------------------
    condition: Mapped[str] = mapped_column(String(32), nullable=False)
    condition_description: Mapped[str | None] = mapped_column(String(128))
    wmo_code: Mapped[int | None] = mapped_column(SmallInteger)


class WeatherObservation(_SpatialRecord, Base):
    """Conditions observed or analysed at a place and instant."""

    __tablename__ = "weather_observations"

    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    temperature_c: Mapped[float | None] = mapped_column(Float)
    apparent_temperature_c: Mapped[float | None] = mapped_column(Float)
    dew_point_c: Mapped[float | None] = mapped_column(Float)
    relative_humidity_pct: Mapped[float | None] = mapped_column(Float)
    pressure_msl_hpa: Mapped[float | None] = mapped_column(Float)
    surface_pressure_hpa: Mapped[float | None] = mapped_column(Float)
    wind_speed_ms: Mapped[float | None] = mapped_column(Float)
    wind_gust_ms: Mapped[float | None] = mapped_column(Float)
    wind_direction_deg: Mapped[float | None] = mapped_column(Float)
    precipitation_mm: Mapped[float | None] = mapped_column(Float)
    cloud_cover_pct: Mapped[float | None] = mapped_column(Float)
    visibility_m: Mapped[float | None] = mapped_column(Float)
    uv_index: Mapped[float | None] = mapped_column(Float)
    is_day: Mapped[bool | None] = mapped_column(Boolean)

    __table_args__ = (
        # Re-fetching the same 15-minute observation must update the row, not
        # append a duplicate.
        UniqueConstraint(
            "provider_id",
            "model",
            "location_key",
            "observed_at",
            name="uq_weather_observations_identity",
        ),
        # The historical endpoint filters by place and orders by time.
        Index(
            "ix_weather_observations_location_key_observed_at",
            "location_key",
            "observed_at",
        ),
        Index("ix_weather_observations_observed_at", "observed_at"),
        Index("ix_weather_observations_provider_id", "provider_id"),
        # Backs ST_DWithin / ST_Distance in the radius search.
        Index(
            "ix_weather_observations_geom",
            "geom",
            postgresql_using="gist",
        ),
    )


class WeatherForecast(Base):
    """A single predicted point: what a source expected, and when it said so."""

    __tablename__ = "weather_forecasts"

    id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    location_key: Mapped[str] = mapped_column(String(32), nullable=False)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    elevation_m: Mapped[float | None] = mapped_column(Float)
    geom: Mapped[object] = mapped_column(
        Geography(geometry_type="POINT", srid=SRID, spatial_index=False),
        nullable=False,
    )
    timezone: Mapped[str | None] = mapped_column(String(64))

    provider_id: Mapped[str] = mapped_column(String(64), nullable=False)
    provider_name: Mapped[str] = mapped_column(String(128), nullable=False)
    model: Mapped[str] = mapped_column(
        String(64), nullable=False, server_default=UNDISCLOSED_MODEL
    )
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    #: Initialisation time of the NWP run behind the values. Null for sources
    #: that do not disclose one; never inferred from the fetch time.
    model_run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    source_url: Mapped[str | None] = mapped_column(String(512))
    license: Mapped[str | None] = mapped_column(String(128))
    attribution: Mapped[str | None] = mapped_column(String(256))
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    #: When this prediction was produced. Bucketed from the fetch time, because
    #: providers rarely publish their model run — see PersistenceService.
    forecast_created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    #: The instant being predicted. For a daily point, local midnight.
    forecast_for: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    #: "hourly" or "daily" — the two series carry different fields, so mixing
    #: them in one query would be a mistake.
    resolution: Mapped[str] = mapped_column(String(8), nullable=False)

    condition: Mapped[str] = mapped_column(String(32), nullable=False)
    condition_description: Mapped[str | None] = mapped_column(String(128))
    wmo_code: Mapped[int | None] = mapped_column(SmallInteger)

    # Hourly values, and the daily aggregates that share their meaning.
    temperature_c: Mapped[float | None] = mapped_column(Float)
    temperature_min_c: Mapped[float | None] = mapped_column(Float)
    temperature_max_c: Mapped[float | None] = mapped_column(Float)
    apparent_temperature_c: Mapped[float | None] = mapped_column(Float)
    apparent_temperature_min_c: Mapped[float | None] = mapped_column(Float)
    apparent_temperature_max_c: Mapped[float | None] = mapped_column(Float)
    dew_point_c: Mapped[float | None] = mapped_column(Float)
    relative_humidity_pct: Mapped[float | None] = mapped_column(Float)
    pressure_msl_hpa: Mapped[float | None] = mapped_column(Float)
    wind_speed_ms: Mapped[float | None] = mapped_column(Float)
    wind_gust_ms: Mapped[float | None] = mapped_column(Float)
    wind_direction_deg: Mapped[float | None] = mapped_column(Float)
    precipitation_mm: Mapped[float | None] = mapped_column(Float)
    precipitation_probability_pct: Mapped[float | None] = mapped_column(Float)
    precipitation_hours: Mapped[float | None] = mapped_column(Float)
    cloud_cover_pct: Mapped[float | None] = mapped_column(Float)
    visibility_m: Mapped[float | None] = mapped_column(Float)
    uv_index: Mapped[float | None] = mapped_column(Float)
    is_day: Mapped[bool | None] = mapped_column(Boolean)
    sunrise: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    sunset: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        # One row per prediction. Re-requesting the same forecast within a
        # generation window updates it; a new generation appends, preserving
        # the history needed to score forecast skill.
        UniqueConstraint(
            "provider_id",
            "model",
            "location_key",
            "resolution",
            "forecast_created_at",
            "forecast_for",
            name="uq_weather_forecasts_identity",
        ),
        # "What is predicted for this place over this window" — the read path.
        Index(
            "ix_weather_forecasts_location_key_forecast_for",
            "location_key",
            "forecast_for",
        ),
        # "What did we predict, and when did we say it" — skill analysis.
        Index(
            "ix_weather_forecasts_forecast_created_at_forecast_for",
            "forecast_created_at",
            "forecast_for",
        ),
        Index("ix_weather_forecasts_provider_id", "provider_id"),
        Index("ix_weather_forecasts_geom", "geom", postgresql_using="gist"),
    )


class WeatherAlert(_SpatialRecord, Base):
    """A triggered rule, persisted for auditing and for later evaluation.

    Rows are never deleted when an alert stops being active. The history of what
    fired — and what turned out to be a false alarm — is the only basis on which
    the rules can later be assessed.

    Deduplication is enforced by a *partial* unique index on ``dedup_key`` over
    active rows only. That gives exactly the semantics needed: while a condition
    persists, repeated evaluation updates one active alert; once it expires or
    resolves, the key is free again and a genuinely new episode can open.
    """

    __tablename__ = "weather_alerts"

    #: Deterministic identity of the alert episode. Computed by AlertService;
    #: deliberately excludes the weather timestamp for observed alerts, so
    #: repeated evaluation of an ongoing condition lands on the same row.
    dedup_key: Mapped[str] = mapped_column(String(128), nullable=False)

    alert_type: Mapped[str] = mapped_column(String(48), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    #: DETERMINISTIC_RULE or OFFICIAL_WARNING. Never inferred — an official
    #: warning may only be written from an actual authoritative source.
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    #: OBSERVED or FORECAST_RISK. "It is raining" versus "rain is forecast".
    kind: Mapped[str] = mapped_column(String(16), nullable=False)

    rule_id: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str] = mapped_column(String(1024), nullable=False)

    triggered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    valid_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    valid_until: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # --- Evidence -----------------------------------------------------------
    # The measured value, its threshold and its unit are first-class columns
    # rather than JSON: they are queried, compared and audited. Only the
    # open-ended supporting context is JSON, because its shape varies by rule.
    variable: Mapped[str] = mapped_column(String(64), nullable=False)
    observed_value: Mapped[float] = mapped_column(Float, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    unit: Mapped[str] = mapped_column(String(16), nullable=False)
    comparison: Mapped[str] = mapped_column(String(8), nullable=False)
    sample_window: Mapped[str] = mapped_column(String(16), nullable=False)
    evidence_context: Mapped[dict | None] = mapped_column(JSONB)

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        # "What is active near here right now" — the API's main query.
        Index(
            "ix_weather_alerts_status_valid_until",
            "status",
            "valid_until",
        ),
        Index("ix_weather_alerts_location_key_status", "location_key", "status"),
        Index("ix_weather_alerts_alert_type_severity", "alert_type", "severity"),
        Index("ix_weather_alerts_rule_id", "rule_id"),
        Index("ix_weather_alerts_provider_id", "provider_id"),
        Index("ix_weather_alerts_valid_from", "valid_from"),
        Index("ix_weather_alerts_geom", "geom", postgresql_using="gist"),
        # The deduplication guarantee. Partial, so only one alert per identity
        # can be active at a time while history accumulates freely underneath.
        Index(
            "uq_weather_alerts_active_dedup",
            "dedup_key",
            unique=True,
            postgresql_where=text("status = 'active'"),
        ),
    )


class AlertSubscription(Base):
    """A location a person asked WeatherGPT to keep watching.

    This is the only new state the alerts feature needs. It holds *what to
    watch*, never *what was found*: the alerts themselves stay in
    ``weather_alerts``, produced by the deterministic engine exactly as they are
    for any other request. A subscription is a standing instruction to run the
    ordinary pipeline for a point on a schedule, nothing more.

    The place is stored resolved — coordinates plus the administrative labels
    the gazetteer returned — rather than as the text that was typed. Re-resolving
    a name later could quietly land on a different Miyapur.
    """

    __tablename__ = "alert_subscriptions"

    id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    #: Groups subscriptions belonging to one browser. Not an account: the
    #: client mints it and keeps it, so a demo needs no sign-in and no personal
    #: data is stored against a watch.
    owner_key: Mapped[str] = mapped_column(String(64), nullable=False)

    location_key: Mapped[str] = mapped_column(String(32), nullable=False)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    geom: Mapped[object] = mapped_column(
        Geography(geometry_type="POINT", srid=SRID), nullable=False
    )
    label: Mapped[str] = mapped_column(String(200), nullable=False)
    admin1: Mapped[str | None] = mapped_column(String(128))
    country: Mapped[str | None] = mapped_column(String(128))
    timezone: Mapped[str | None] = mapped_column(String(64))

    #: Alert types this watch cares about. Empty means every rule the engine
    #: runs — the sensible default, and the one that cannot silently omit a
    #: hazard the person did not think to tick.
    alert_types: Mapped[list | None] = mapped_column(JSONB)

    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    #: When the monitor last ran the pipeline for this point. Displayed, so a
    #: person can tell "no alerts" from "not checked yet".
    last_evaluated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index("ix_alert_subscriptions_owner_key", "owner_key"),
        Index("ix_alert_subscriptions_enabled", "enabled"),
        Index("ix_alert_subscriptions_geom", "geom", postgresql_using="gist"),
        # One watch per place per browser. Asking twice for the same point is a
        # no-op, not a second row that doubles the evaluation work.
        Index(
            "uq_alert_subscriptions_owner_location",
            "owner_key",
            "location_key",
            unique=True,
        ),
    )


__all__ = [
    "SRID",
    "AlertSubscription",
    "UNDISCLOSED_MODEL",
    "WeatherAlert",
    "WeatherForecast",
    "WeatherObservation",
]
