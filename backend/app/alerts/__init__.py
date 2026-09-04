"""The deterministic alert and risk engine.

Pure decision logic: rules, severity bands, and the engine that applies them.
Nothing in this package performs I/O, and nothing in it is probabilistic.

The safety boundary this package exists to hold:

    validated weather -> rules -> severity -> structured alert -> API -> LLM

A language model explains an alert. It never decides that one exists.
"""

from app.alerts.engine import AlertEngine, Evaluation
from app.alerts.rules import (
    EvaluationSample,
    RuleResult,
    SampleWindow,
    ThresholdRule,
    build_rules,
    sample_from_daily,
    sample_from_hourly,
    sample_from_observation,
)
from app.alerts.severity import SeverityBand, build_ladder, classify

__all__ = [
    "AlertEngine",
    "Evaluation",
    "EvaluationSample",
    "RuleResult",
    "SampleWindow",
    "SeverityBand",
    "ThresholdRule",
    "build_ladder",
    "build_rules",
    "classify",
    "sample_from_daily",
    "sample_from_hourly",
    "sample_from_observation",
]
