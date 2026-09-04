"""Alert orchestration: evaluation, deduplication, lifecycle, retrieval.

The layer between the pure engine and the database:

    WeatherService -> AlertService -> AlertEngine -> rules
                            |
                            +-------> AlertRepository -> PostgreSQL/PostGIS

Responsibilities are split deliberately. The engine decides *whether a threshold
was crossed* and knows nothing about storage. This service decides *whether that
is news* — whether an alert already exists, whether one has lifted, when one
expires — and knows nothing about meteorology.

Two rules inherited from earlier phases hold here too:

* **Only validated data reaches evaluation.** This service is called from inside
  the weather service's cache loader, downstream of the ingestion pipeline's
  validation gate. Provider payloads never reach a rule.
* **A write failure never breaks a read.** Evaluation is a side effect of
  serving weather. If the database is unreachable, the caller still gets their
  forecast; the failure is logged.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone

from app.alerts.engine import AlertEngine, Evaluation
from app.alerts.rules import RuleResult
from app.config.logging import get_logger
from app.core.exceptions import DatabaseUnavailableError
from app.db.engine import Database, translate_database_errors
from app.db.mappers import alert_values, row_to_alert
from app.db.repositories import AlertFilter, AlertRepository
from app.domain.alert import (
    Alert,
    AlertEvidence,
    AlertKind,
    AlertSourceType,
    AlertStatus,
)
from app.domain.forecast import Forecast
from app.domain.location import Location
from app.domain.provenance import DataProvenance
from app.domain.weather import WeatherReport

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class AlertMatch:
    """An alert with its distance from a query point."""

    alert: Alert
    distance_m: float


def dedup_key(
    *,
    source_type: AlertSourceType,
    provider_id: str,
    model: str | None,
    rule_id: str,
    location_key: str,
    kind: AlertKind,
    target: datetime | None,
) -> str:
    """Stable identity for an alert episode.

    What is *not* in this key matters as much as what is. For an observed alert
    the weather timestamp is excluded entirely, so evaluating the same ongoing
    condition at 10:00, 10:05 and 10:10 lands on one alert rather than three.
    For a forecast risk the target instant *is* included, because a prediction
    for 14:00 and one for 15:00 are different claims about different moments.

    Combined with the partial unique index over active rows, this gives the
    intended lifecycle: one open alert per identity, and a free key again once
    that episode expires or resolves, so a genuinely new episode can open later.
    """
    parts = [
        source_type.value,
        provider_id,
        model or "",
        rule_id,
        location_key,
        kind.value,
        target.isoformat() if target is not None else "",
    ]
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    # The prefix keeps keys legible in logs and psql without losing uniqueness.
    return f"{rule_id}:{kind.value}:{digest[:32]}"


def build_alert(
    result: RuleResult,
    *,
    location: Location,
    provenance: DataProvenance,
    triggered_at: datetime,
) -> Alert:
    """Turn a fired rule into a canonical alert.

    ``source_type`` is hard-coded to ``DETERMINISTIC_RULE``. There is no code
    path from a threshold comparison to ``OFFICIAL_WARNING``: that value may
    only ever be written by an ingester reading an actual authority's feed.
    """
    return Alert(
        alert_type=result.rule.alert_type,
        severity=result.severity,
        status=AlertStatus.ACTIVE,
        source_type=AlertSourceType.DETERMINISTIC_RULE,
        kind=result.sample.kind,
        rule_id=result.rule.rule_id,
        title=result.rule.title,
        description=result.describe(),
        location=location,
        triggered_at=triggered_at,
        valid_from=result.sample.valid_from,
        valid_until=result.sample.valid_until,
        evidence=AlertEvidence(
            rule_id=result.rule.rule_id,
            variable=result.rule.variable,
            observed_value=result.observed_value,
            threshold=result.threshold,
            unit=result.rule.unit,
            comparison=result.rule.comparison,
            sample_window=result.sample.window.value,
            context=result.context or None,
        ),
        provenance=provenance,
    )


class AlertService:
    """Evaluates validated weather and maintains the alert record."""

    def __init__(
        self,
        engine: AlertEngine,
        database: Database | None,
        *,
        enabled: bool = True,
        max_per_evaluation: int = 200,
    ) -> None:
        self._engine = engine
        self._database = database
        self._enabled = enabled and database is not None
        self._max_per_evaluation = max_per_evaluation

    @property
    def enabled(self) -> bool:
        """False when evaluation is switched off or there is nowhere to store."""
        return self._enabled

    @property
    def engine(self) -> AlertEngine:
        return self._engine

    # --- Evaluation ---------------------------------------------------------

    async def evaluate_observation(
        self, report: WeatherReport, *, now: datetime | None = None
    ) -> list[Alert]:
        """Evaluate validated current conditions and reconcile the alert record.

        Beyond raising new alerts, this closes the loop on old ones: an alert
        still open for this place and source whose rule did not fire again has
        had its condition lift, and is resolved rather than left to expire.

        Never raises. Returns the alerts now active for this evaluation.
        """
        if not self._enabled:
            return []

        now = now or _utcnow()
        evaluation = self._engine.evaluate_observation(report, now=now)
        return await self._persist(
            evaluation,
            location=report.location,
            provenance=report.provenance,
            now=now,
            reconcile_kind=AlertKind.OBSERVED,
        )

    async def evaluate_forecast(
        self, forecast: Forecast, *, now: datetime | None = None
    ) -> list[Alert]:
        """Evaluate a validated forecast for future risk.

        Everything produced is a ``FORECAST_RISK``: a prediction crossed a
        threshold, which is not the same as the event having occurred. Forecast
        alerts are not reconciled against each other — a target dropping out of
        the horizon is handled by expiry, not resolution.

        Never raises.
        """
        if not self._enabled:
            return []

        now = now or _utcnow()
        evaluation = self._engine.evaluate_forecast(forecast, now=now)
        return await self._persist(
            evaluation,
            location=forecast.location,
            provenance=forecast.provenance,
            now=now,
            reconcile_kind=None,
        )

    async def _persist(
        self,
        evaluation: Evaluation,
        *,
        location: Location,
        provenance: DataProvenance,
        now: datetime,
        reconcile_kind: AlertKind | None,
    ) -> list[Alert]:
        assert self._database is not None  # guarded by self._enabled

        results = evaluation.results[: self._max_per_evaluation]
        if len(evaluation.results) > self._max_per_evaluation:
            logger.warning(
                "alerts.evaluation_truncated",
                extra={
                    "location": location.coordinates.cache_key,
                    "produced": len(evaluation.results),
                    "cap": self._max_per_evaluation,
                },
            )

        try:
            async with self._database.session() as session:
                repository = AlertRepository(session)
                # Age out anything whose window has passed before deciding what
                # is new, so a stale row cannot absorb a fresh episode.
                await repository.expire_stale(now=now)

                stored: list[Alert] = []
                raised: list[str] = []
                for result in results:
                    alert = build_alert(
                        result,
                        location=location,
                        provenance=provenance,
                        triggered_at=now,
                    )
                    key = dedup_key(
                        source_type=alert.source_type,
                        provider_id=provenance.provider_id,
                        model=provenance.model,
                        rule_id=alert.rule_id,
                        location_key=location.coordinates.cache_key,
                        kind=alert.kind,
                        # Observed alerts deliberately carry no target: that is
                        # what makes an ongoing condition one alert.
                        target=(
                            None
                            if alert.kind is AlertKind.OBSERVED
                            else alert.valid_from
                        ),
                    )
                    raised.append(key)
                    values = alert_values(alert, dedup_key=key)

                    existing = await repository.find_active_by_dedup_key(key)
                    if existing is None:
                        row = await repository.insert(values)
                    else:
                        await repository.refresh(existing, values)
                        row = existing
                    stored.append(row_to_alert(row))

                if reconcile_kind is not None:
                    resolved = await repository.resolve_except(
                        location_key=location.coordinates.cache_key,
                        provider_id=provenance.provider_id,
                        kind=reconcile_kind.value,
                        keep=raised,
                        now=now,
                    )
                    if resolved:
                        logger.info(
                            "alerts.resolved",
                            extra={
                                "location": location.coordinates.cache_key,
                                "provider": provenance.provider_id,
                                "count": resolved,
                            },
                        )
        except Exception:  # noqa: BLE001 - evaluation must not break the response
            logger.exception(
                "alerts.evaluation_failed",
                extra={
                    "location": location.coordinates.cache_key,
                    "provider": provenance.provider_id,
                    "rules_triggered": len(results),
                },
            )
            return []

        if stored:
            logger.info(
                "alerts.persisted",
                extra={
                    "location": location.coordinates.cache_key,
                    "provider": provenance.provider_id,
                    "count": len(stored),
                    "rules": sorted({alert.rule_id for alert in stored}),
                },
            )
        return stored

    # --- Retrieval ----------------------------------------------------------

    async def search(self, criteria: AlertFilter) -> list[AlertMatch]:
        """Alerts matching a filter.

        Unlike evaluation, this is a read whose whole point is the database, so
        a failure is reported rather than swallowed.

        Raises:
            DatabaseUnavailableError: persistence is unconfigured or unreachable.
        """
        if self._database is None:
            raise DatabaseUnavailableError(
                "Alerts require a configured database.",
                details={"reason": "persistence_disabled"},
            )

        async with translate_database_errors("alerts.search"):
            async with self._database.session() as session:
                found = await AlertRepository(session).search(criteria)

        logger.info(
            "alerts.search",
            extra={
                "spatial": criteria.is_spatial,
                "statuses": list(criteria.statuses),
                "results": len(found),
            },
        )
        return [
            AlertMatch(alert=row_to_alert(entry.alert), distance_m=entry.distance_m)
            for entry in found
        ]

    async def expire_stale(self, *, now: datetime | None = None) -> int:
        """Age out alerts whose validity window has elapsed.

        Raises:
            DatabaseUnavailableError: persistence is unconfigured or unreachable.
        """
        if self._database is None:
            raise DatabaseUnavailableError(
                "Alerts require a configured database.",
                details={"reason": "persistence_disabled"},
            )

        async with translate_database_errors("alerts.expire_stale"):
            async with self._database.session() as session:
                return await AlertRepository(session).expire_stale(
                    now=now or _utcnow()
                )


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


__all__ = ["AlertMatch", "AlertService", "build_alert", "dedup_key"]
