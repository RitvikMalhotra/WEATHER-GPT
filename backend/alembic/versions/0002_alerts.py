"""Alert table for the deterministic rule engine.

The one non-obvious piece is the deduplication index. It is *partial* — unique
over active rows only — which is what allows one open alert per identity while
history accumulates freely beneath it. A plain unique index would forbid a
condition ever recurring at the same place; no index at all would let repeated
evaluation of one ongoing condition produce a row every few minutes.

Revision ID: 0002_alerts
Revises: 0001_initial_schema
Create Date: 2026-09-04
"""

from __future__ import annotations

from typing import Sequence

import sqlalchemy as sa
from alembic import op
from geoalchemy2 import Geography
from sqlalchemy.dialects import postgresql

revision: str = "0002_alerts"
down_revision: str | None = "0001_initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SRID = 4326


def upgrade() -> None:
    op.create_table(
        "weather_alerts",
        sa.Column("id", sa.UUID(as_uuid=True), nullable=False),
        # Place
        sa.Column("location_key", sa.String(length=32), nullable=False),
        sa.Column("latitude", sa.Float(), nullable=False),
        sa.Column("longitude", sa.Float(), nullable=False),
        sa.Column("elevation_m", sa.Float(), nullable=True),
        sa.Column(
            "geom",
            Geography(geometry_type="POINT", srid=SRID, spatial_index=False),
            nullable=False,
        ),
        sa.Column("timezone", sa.String(length=64), nullable=True),
        # Provenance — which source and model produced the triggering data
        sa.Column("provider_id", sa.String(length=64), nullable=False),
        sa.Column("provider_name", sa.String(length=128), nullable=False),
        sa.Column("model", sa.String(length=64), server_default="", nullable=False),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source_url", sa.String(length=512), nullable=True),
        sa.Column("license", sa.String(length=128), nullable=True),
        sa.Column("attribution", sa.String(length=256), nullable=True),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        # Inherited from the shared spatial record; holds the hazard type.
        sa.Column("condition", sa.String(length=32), nullable=False),
        sa.Column("condition_description", sa.String(length=128), nullable=True),
        sa.Column("wmo_code", sa.SmallInteger(), nullable=True),
        # Identity and classification
        sa.Column("dedup_key", sa.String(length=128), nullable=False),
        sa.Column("alert_type", sa.String(length=48), nullable=False),
        sa.Column("severity", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        # DETERMINISTIC_RULE vs OFFICIAL_WARNING — never inferred.
        sa.Column("source_type", sa.String(length=32), nullable=False),
        # OBSERVED vs FORECAST_RISK — "it is raining" vs "rain is forecast".
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("rule_id", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=128), nullable=False),
        sa.Column("description", sa.String(length=1024), nullable=False),
        # Lifecycle
        sa.Column("triggered_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("valid_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column("valid_until", sa.DateTime(timezone=True), nullable=False),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        # Evidence — columns, not a blob, because these are queried and audited
        sa.Column("variable", sa.String(length=64), nullable=False),
        sa.Column("observed_value", sa.Float(), nullable=False),
        sa.Column("threshold", sa.Float(), nullable=False),
        sa.Column("unit", sa.String(length=16), nullable=False),
        sa.Column("comparison", sa.String(length=8), nullable=False),
        sa.Column("sample_window", sa.String(length=16), nullable=False),
        # Only the open-ended supporting context is JSON.
        sa.Column("evidence_context", postgresql.JSONB(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_weather_alerts"),
    )

    op.create_index(
        "ix_weather_alerts_status_valid_until",
        "weather_alerts",
        ["status", "valid_until"],
    )
    op.create_index(
        "ix_weather_alerts_location_key_status",
        "weather_alerts",
        ["location_key", "status"],
    )
    op.create_index(
        "ix_weather_alerts_alert_type_severity",
        "weather_alerts",
        ["alert_type", "severity"],
    )
    op.create_index("ix_weather_alerts_rule_id", "weather_alerts", ["rule_id"])
    op.create_index("ix_weather_alerts_provider_id", "weather_alerts", ["provider_id"])
    op.create_index("ix_weather_alerts_valid_from", "weather_alerts", ["valid_from"])
    op.create_index(
        "ix_weather_alerts_geom", "weather_alerts", ["geom"], postgresql_using="gist"
    )
    # The deduplication guarantee.
    op.create_index(
        "uq_weather_alerts_active_dedup",
        "weather_alerts",
        ["dedup_key"],
        unique=True,
        postgresql_where=sa.text("status = 'active'"),
    )


def downgrade() -> None:
    op.drop_index("uq_weather_alerts_active_dedup", table_name="weather_alerts")
    op.drop_index("ix_weather_alerts_geom", table_name="weather_alerts")
    op.drop_index("ix_weather_alerts_valid_from", table_name="weather_alerts")
    op.drop_index("ix_weather_alerts_provider_id", table_name="weather_alerts")
    op.drop_index("ix_weather_alerts_rule_id", table_name="weather_alerts")
    op.drop_index("ix_weather_alerts_alert_type_severity", table_name="weather_alerts")
    op.drop_index("ix_weather_alerts_location_key_status", table_name="weather_alerts")
    op.drop_index("ix_weather_alerts_status_valid_until", table_name="weather_alerts")
    op.drop_table("weather_alerts")
