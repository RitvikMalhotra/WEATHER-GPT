"""Record which NWP model run produced a stored value.

A forecast is only as fresh as the model cycle behind it: two rows fetched
minutes apart can both come from the 06Z GFS run. Nullable, because sources
that do not disclose a run time must store nothing rather than a guess.

Revision ID: 0003_model_run_at
Revises: 0002_alerts
Create Date: 2026-09-04
"""

from __future__ import annotations

from typing import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_model_run_at"
down_revision: str | None = "0002_alerts"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLES = ("weather_observations", "weather_forecasts", "weather_alerts")


def upgrade() -> None:
    for table in _TABLES:
        op.add_column(
            table, sa.Column("model_run_at", sa.DateTime(timezone=True), nullable=True)
        )
    # Forecast-skill work asks "what did the 06Z run predict"; without an index
    # that is a full scan of every forecast row.
    op.create_index(
        "ix_weather_forecasts_model_run_at", "weather_forecasts", ["model_run_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_weather_forecasts_model_run_at", table_name="weather_forecasts")
    for table in _TABLES:
        op.drop_column(table, "model_run_at")
