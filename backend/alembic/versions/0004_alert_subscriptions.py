"""Alert subscriptions: the places a person asked WeatherGPT to keep watching.

One table. It records *what to watch* — a resolved point, the rules to care
about, and whether the watch is on. What gets found still lands in
``weather_alerts``, written by the deterministic engine on the ordinary path,
so this migration adds no alert state and changes no existing table.

Revision ID: 0004_alert_subscriptions
Revises: 0003_model_run_at
"""

from __future__ import annotations

import geoalchemy2
import sqlalchemy as sa
from alembic import op

revision: str = "0004_alert_subscriptions"
down_revision: str | None = "0003_model_run_at"
branch_labels: str | None = None
depends_on: str | None = None

SRID = 4326


def upgrade() -> None:
    op.create_table(
        "alert_subscriptions",
        sa.Column(
            "id",
            sa.dialects.postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
        ),
        # Groups watches belonging to one browser. Not an account: the client
        # mints it, so a demo needs no sign-in and stores nothing personal.
        sa.Column("owner_key", sa.String(length=64), nullable=False),
        sa.Column("location_key", sa.String(length=32), nullable=False),
        sa.Column("latitude", sa.Float(), nullable=False),
        sa.Column("longitude", sa.Float(), nullable=False),
        sa.Column(
            "geom",
            geoalchemy2.Geography(geometry_type="POINT", srid=SRID),
            nullable=False,
        ),
        sa.Column("label", sa.String(length=200), nullable=False),
        sa.Column("admin1", sa.String(length=128), nullable=True),
        sa.Column("country", sa.String(length=128), nullable=True),
        sa.Column("timezone", sa.String(length=64), nullable=True),
        # Empty means every rule the engine runs, which is the default that
        # cannot silently omit a hazard nobody thought to tick.
        sa.Column(
            "alert_types",
            sa.dialects.postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("last_evaluated_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_index("ix_alert_subscriptions_owner_key", "alert_subscriptions", ["owner_key"])
    op.create_index("ix_alert_subscriptions_enabled", "alert_subscriptions", ["enabled"])
    op.create_index(
        "ix_alert_subscriptions_geom",
        "alert_subscriptions",
        ["geom"],
        postgresql_using="gist",
    )
    # One watch per place per browser: asking twice is the same instruction.
    op.create_index(
        "uq_alert_subscriptions_owner_location",
        "alert_subscriptions",
        ["owner_key", "location_key"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("uq_alert_subscriptions_owner_location", table_name="alert_subscriptions")
    op.drop_index("ix_alert_subscriptions_geom", table_name="alert_subscriptions")
    op.drop_index("ix_alert_subscriptions_enabled", table_name="alert_subscriptions")
    op.drop_index("ix_alert_subscriptions_owner_key", table_name="alert_subscriptions")
    op.drop_table("alert_subscriptions")
