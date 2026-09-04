"""The alert engine.

Turns validated weather into structured rule results. Pure and synchronous: no
database, no HTTP, no clock of its own beyond what it is handed. That is what
makes the safety-critical decision — *did a threshold get crossed* —
exhaustively testable, and what keeps it impossible for anything non-
deterministic to influence the answer.

The engine is fed by :class:`~app.services.alerts.AlertService`, which is the
only component that talks to the database. The engine itself never learns
whether an alert already exists; deduplication and lifecycle are somebody
else's job.

Input is always a canonical model that has already passed the validation gate.
Provider payloads never reach this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Sequence

from app.alerts.rules import (
    EvaluationSample,
    RuleResult,
    ThresholdRule,
    sample_from_daily,
    sample_from_hourly,
    sample_from_observation,
)
from app.config.logging import get_logger
from app.domain.forecast import Forecast
from app.domain.weather import WeatherReport

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Evaluation:
    """Everything one evaluation pass produced."""

    results: tuple[RuleResult, ...]
    evaluated_at: datetime
    samples_examined: int

    @property
    def triggered(self) -> bool:
        return bool(self.results)


class AlertEngine:
    """Applies a rule set to validated weather.

    Args:
        rules: the active rule set, built from configuration.
        observation_validity: how long a current-conditions reading is treated
            as describing the present. Also the alert's validity window, so a
            condition that stops being reported ages out rather than lingering.
        forecast_lookahead: forecast points beyond this horizon are ignored. A
            risk flagged ten days out is noise, and it would keep the alert
            table growing without telling anyone anything actionable.
    """

    def __init__(
        self,
        rules: Sequence[ThresholdRule],
        *,
        observation_validity: timedelta = timedelta(hours=1),
        forecast_lookahead: timedelta = timedelta(hours=48),
    ) -> None:
        self._rules = tuple(rules)
        self._observation_validity = observation_validity
        self._forecast_lookahead = forecast_lookahead

    @property
    def rules(self) -> tuple[ThresholdRule, ...]:
        return self._rules

    def evaluate_sample(self, sample: EvaluationSample) -> list[RuleResult]:
        """Apply every applicable rule to one sample."""
        results = [
            result
            for rule in self._rules
            if (result := rule.evaluate(sample)) is not None
        ]
        return results

    def evaluate_observation(
        self, report: WeatherReport, *, now: datetime | None = None
    ) -> Evaluation:
        """Evaluate validated current conditions.

        The report has already cleared the ingestion pipeline's validation gate,
        so the values here are physically plausible and internally consistent.
        Nothing in this method re-checks that, and nothing bypasses it.
        """
        now = now or _utcnow()
        sample = sample_from_observation(
            report.current, validity=self._observation_validity
        )
        results = tuple(self.evaluate_sample(sample))

        if results:
            logger.info(
                "alerts.observation_triggered",
                extra={
                    "location": report.location.coordinates.cache_key,
                    "provider": report.provenance.provider_id,
                    "rules": [result.rule.rule_id for result in results],
                    "severities": [result.severity.value for result in results],
                },
            )
        return Evaluation(results=results, evaluated_at=now, samples_examined=1)

    def evaluate_forecast(
        self, forecast: Forecast, *, now: datetime | None = None
    ) -> Evaluation:
        """Evaluate a validated forecast for future risk.

        Everything produced here is a *forecast risk*, never an observation: the
        threshold is met by a prediction. The distinction is carried on the
        sample and survives into the persisted alert.
        """
        now = now or _utcnow()
        horizon = now + self._forecast_lookahead

        samples: list[EvaluationSample] = []
        samples.extend(
            sample_from_hourly(point)
            for point in forecast.hourly
            if point.valid_at <= horizon
        )
        samples.extend(
            sample
            for point in forecast.daily
            if (sample := sample_from_daily(point)).valid_from <= horizon
        )

        results: list[RuleResult] = []
        for sample in samples:
            results.extend(self.evaluate_sample(sample))

        if results:
            logger.info(
                "alerts.forecast_triggered",
                extra={
                    "location": forecast.location.coordinates.cache_key,
                    "provider": forecast.provenance.provider_id,
                    "samples": len(samples),
                    "rules": sorted({result.rule.rule_id for result in results}),
                },
            )
        return Evaluation(
            results=tuple(results), evaluated_at=now, samples_examined=len(samples)
        )


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)
