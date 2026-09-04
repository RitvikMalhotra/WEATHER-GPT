"""Live end-to-end check of the conversational layer against a running server.

Run the API first, then point this at it. It exercises the demo scenarios and
asserts the property that matters: every weather value in an answer also
appears in the tool result the backend returned for that same turn.

    python scripts/demo_ai.py                       # default http://127.0.0.1:8000
    python scripts/demo_ai.py --base-url http://127.0.0.1:8000 --hindi
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any

import httpx

#: Numbers as the renderer emits them, so they can be looked for in the data.
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
#: Values that are not weather readings and appear in prose or identifiers.
_IGNORED_CONTEXT = ("fetched", "Observed at", "प्राप्त", "प्रेक्षण समय", "http", "24")


def _numbers(text: str) -> set[str]:
    return {match.group() for match in _NUMBER.finditer(text)}


def _data_numbers(payload: Any) -> set[str]:
    """Every number anywhere in the returned tool data, as rendered strings."""
    found: set[str] = set()
    if isinstance(payload, dict):
        for value in payload.values():
            found |= _data_numbers(value)
    elif isinstance(payload, list):
        for value in payload:
            found |= _data_numbers(value)
    elif isinstance(payload, (int, float)) and not isinstance(payload, bool):
        found.add(f"{payload:g}")
        found.add(str(payload))
    elif isinstance(payload, str):
        found |= _numbers(payload)
    return found


def _check_grounded(answer: str, results: list[dict[str, Any]]) -> list[str]:
    """Report answer numbers with no counterpart in the backend data."""
    if not results:
        return []
    available = set()
    for result in results:
        available |= _data_numbers(result.get("data", {}))
    unexplained = []
    for line in answer.splitlines():
        if any(token in line for token in _IGNORED_CONTEXT):
            continue
        for number in _numbers(line):
            if number not in available:
                unexplained.append(f"{number!r} in {line.strip()!r}")
    return unexplained


def ask(client: httpx.Client, message: str, session: str | None) -> dict[str, Any]:
    payload: dict[str, Any] = {"message": message}
    if session:
        payload["session_id"] = session
    response = client.post("/api/v1/ai/chat", json=payload, timeout=60.0)
    response.raise_for_status()
    return response.json()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--hindi", action="store_true", help="also run a Hindi turn")
    args = parser.parse_args()

    # The fourth field says whether the turn continues the running session.
    # Only the follow-up needs the carried location; a turn that reuses the
    # session also inherits its last intent, which biases the model's choice.
    scenarios: list[tuple[str, str, str, bool]] = [
        ("current weather", "What is the weather in New Delhi?", "current_weather", False),
        ("forecast", "Forecast for New Delhi for 3 days", "daily_forecast", False),
        ("alerts", "Are there any alerts in New Delhi?", "alerts", False),
        ("follow-up", "What about tomorrow?", "daily_forecast", True),
    ]
    if args.hindi:
        scenarios.append(("hindi", "दिल्ली में मौसम कैसा है?", "current_weather", False))

    failures: list[str] = []
    session: str | None = None
    with httpx.Client(base_url=args.base_url) as client:
        for label, message, expected_intent, continues in scenarios:
            print(f"\n=== {label}: {message}")
            try:
                body = ask(client, message, session if continues else None)
            except httpx.HTTPError as exc:
                failures.append(f"{label}: request failed: {exc}")
                print(f"  REQUEST FAILED: {exc}")
                continue

            if not continues:
                session = body["session_id"]
            intent, results = body["intent"], body.get("tool_results", [])
            print(f"  intent={intent} language={body['language']} tools={len(results)}")
            print(f"  sources={[s['provider_id'] for s in body.get('sources', [])]}")
            print("  " + body["answer"].replace("\n", "\n  "))

            if intent != expected_intent:
                failures.append(f"{label}: intent {intent!r}, expected {expected_intent!r}")
            if not results:
                failures.append(f"{label}: no tool result — nothing grounded the answer")
            if results and not body.get("sources"):
                failures.append(f"{label}: data returned without provenance")
            for problem in _check_grounded(body["answer"], results):
                failures.append(f"{label}: ungrounded number {problem}")

    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED ({len(failures)})")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"OK — {len(scenarios)} scenarios grounded in backend data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
