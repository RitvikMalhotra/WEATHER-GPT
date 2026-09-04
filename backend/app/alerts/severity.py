"""Severity bands.

A band is a threshold paired with the severity that applies once the threshold
is met. A rule holds an ordered ladder of them, and evaluation returns the
highest band the observed value reaches.

Keeping this separate from the rules themselves means the mapping from "how
much" to "how bad" is one small, exhaustively testable function, and the
boundary behaviour — is *exactly* the threshold enough? — is stated in one
place rather than re-decided per rule.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.domain.alert import AlertSeverity


@dataclass(frozen=True, slots=True)
class SeverityBand:
    """A threshold and the severity reaching it implies."""

    threshold: float
    severity: AlertSeverity


def build_ladder(*bands: tuple[float, AlertSeverity]) -> tuple[SeverityBand, ...]:
    """Build an ascending ladder, rejecting one that is out of order.

    An unordered ladder would silently return the wrong severity, so it is a
    construction-time error rather than something to discover in production.
    """
    ladder = tuple(SeverityBand(threshold, severity) for threshold, severity in bands)
    if not ladder:
        raise ValueError("A severity ladder needs at least one band.")

    thresholds = [band.threshold for band in ladder]
    if thresholds != sorted(thresholds):
        raise ValueError(f"Severity bands must ascend by threshold, got {thresholds}.")

    ranks = [band.severity.rank for band in ladder]
    if ranks != sorted(ranks):
        raise ValueError(
            f"Severity bands must ascend by severity, got "
            f"{[band.severity.value for band in ladder]}."
        )
    return ladder


def classify(
    value: float, ladder: tuple[SeverityBand, ...]
) -> SeverityBand | None:
    """Return the highest band ``value`` reaches, or ``None`` if it reaches none.

    The comparison is ``>=``: a value landing exactly on a threshold is *in* that
    band. Thresholds are stated as "this much or more", so excluding the
    boundary would make a rule configured at 50 mm silently ignore 50 mm.
    """
    matched: SeverityBand | None = None
    for band in ladder:
        if value >= band.threshold:
            matched = band
        else:
            break
    return matched
