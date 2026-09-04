"""One line that answers the question that was asked.

Numbers answer "what is the rainfall probability". They do not answer "should I
spray today", and for a long time this module returned the former to people who
had asked the latter — the same reading of the rainfall, whatever the question
was about. A verdict closes that gap, and it is the part of a weather assistant
most likely to invent something, so the pipeline it runs through is narrow:

    backend response ──▶ Brief ──▶ Recommendation ──▶ Verdict
                        (facts)   (the answer)      (one line)

:mod:`app.ai.brief` reduces the response to the figures a decision turns on,
plus the variables it did not carry. :mod:`app.ai.recommend` composes the
recommendation from those figures and the question's domain. This module is the
entry point and the guard between them.

Three rules survive from the original design and still hold:

* **A verdict reads; it never writes.** Every figure in it was returned by the
  backend in this turn. Nothing is estimated, and a missing variable is named
  as missing rather than filled in.
* **It interprets, it does not warn.** Official warnings come from the
  deterministic alert engine and from meteorological services. This layer does
  not create thresholds and does not touch that engine.
* **It stops where its competence stops.** The weather half of "should I spray"
  is answerable from a forecast. Whether the product tolerates the rain that
  follows is a question about the product — raised only if the person raised
  it, because a verdict that opens with rainfast periods has answered a
  question nobody asked.

An optional ``author`` may phrase the recommendation more naturally from the
same brief. It is checked before it is used: any figure it states that the
brief does not account for means the whole sentence is discarded and the
composed one is shown instead. A model may improve the wording; it may never
add a fact.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.ai import brief, recommend
from app.ai.models import AdvisoryPurpose, Intent
from app.config.logging import get_logger

logger = get_logger(__name__)

#: Intents whose results carry measurements. Anything else — a clarification, a
#: location search, a bare alert list — has nothing to interpret.
_READABLE = frozenset(
    {
        Intent.CURRENT_WEATHER,
        Intent.HOURLY_FORECAST,
        Intent.DAILY_FORECAST,
        Intent.HISTORICAL_WEATHER,
        Intent.LOCATION_RISK,
    }
)


@dataclass(frozen=True, slots=True)
class Verdict:
    """A one-line answer to the question, plus anything it cannot speak to."""

    text: str
    icon: str
    #: Set when a variable the question turned on was not in the response.
    caveat: str | None = None

    def as_line(self) -> str:
        return f"{self.icon} {self.text}" if self.icon else self.text


def evidence(
    *,
    intent: Intent,
    data: object,
    question: str = "",
    place: str = "",
    horizon_hours: int = 24,
    purpose: AdvisoryPurpose = AdvisoryPurpose.GENERAL,
    active_alerts: int | None = None,
) -> brief.Brief | None:
    """The facts this answer may be built from, or nothing when there are none.

    Separate from :func:`render` because authoring the sentence may involve an
    awaitable, and the caller needs the brief in hand before it can ask for one.
    """
    if intent not in _READABLE:
        return None
    return brief.build(
        question=question,
        intent=intent,
        domain=purpose,
        data=data,
        place=place,
        window_hours=horizon_hours,
        active_alerts=active_alerts,
    )


def render(
    found: brief.Brief,
    *,
    language: str = "en",
    format_time=None,
    authored: str | None = None,
) -> Verdict | None:
    """Compose the recommendation, preferring a checked phrasing when given one."""
    composed = recommend.compose(found, language=language, format_time=format_time)
    if composed is None:
        return None

    text = composed.text
    if authored:
        invented = brief.unsupported_numbers(authored, found)
        if invented:
            # The sentence stated a figure the backend did not return. It is
            # discarded whole rather than patched: a phrasing that invented one
            # number is not evidence about the rest of it.
            logger.info("ai.verdict.author_rejected", extra={"unsupported": invented})
        else:
            text = authored.strip()

    return Verdict(text, composed.icon, composed.caveat)


def build(
    *,
    intent: Intent,
    data: object,
    question: str = "",
    place: str = "",
    language: str = "en",
    variable: str | None = None,
    horizon_hours: int = 24,
    purpose: AdvisoryPurpose = AdvisoryPurpose.GENERAL,
    active_alerts: int | None = None,
    format_time=None,
    authored: str | None = None,
) -> Verdict | None:
    """Evidence and rendering in one call, for callers with nothing to await.

    Returns ``None`` for a clarification, a location search, an alert list, or
    any result whose fields do not support a reading. A verdict on a message
    that carries no measurement would be an opinion, not an interpretation.
    """
    found = evidence(
        intent=intent,
        data=data,
        question=question,
        place=place,
        horizon_hours=horizon_hours,
        purpose=purpose,
        active_alerts=active_alerts,
    )
    if found is None:
        return None
    return render(found, language=language, format_time=format_time, authored=authored)
