---
version: 1
slug: "backend-app-voice-page-html"
primary_target: "backend/app/voice/page.html"
related_targets: ["backend/app/voice/router.py"]
---

## Scope

Surface: the WeatherGPT client at `GET /voice` (`backend/app/voice/`). Visitor
mode: **Operate** — a person is in a task, asking one weather question and
acting on the answer. Desktop first (1024–1920), then the same document made
responsive to 375–768.

## Audience and job

People in India asking about their own locality in English or Hindi, by voice
or by typing. Job: ask, get a grounded answer, know where the number came from,
pin the exact place when the name is ambiguous, hear it read aloud.

Content and proof come from the frozen FastAPI backend: `/api/v1/ai/chat`
returns the answer plus typed `tool_results` and `sources`;
`/api/v1/locations/search` returns ranked candidates with admin1/country.
No weather value is authored in the client.

## Direction contract

THESIS: the interface is read as an **atmospheric sounding** — a vertical
column of the air above one place, with the conversation ascending through it.
The category default (a pastel weather card over a photo sky, or a plain
chatbot column) is refused: this is an instrument that plots what the
atmosphere is doing and says who measured it.

WORLD: a sounding chart. Ground is a single continuous vertical atmospheric
gradient (cool at the top of the column, warm near the surface) carrying a
hairline skew-T graticule at 2–4% opacity that only becomes legible in the
header band. Ink is near-black on a light column; the one signal colour is the
parcel path — a saturated blue used for the primary action, the active state
and nothing decorative. Data wears the sounding's own vocabulary: hairline
rules, tabular numerals, small uppercase field labels, values that align on
their units.

FIRST VIEWPORT: the header names the place being sounded and the language,
and the column below is already open with a real invitation, not an empty box.
The composer sits at the surface — the bottom of the column — where the air
is warmest and the graticule is densest.

SIGNATURE INTERACTION: the answer does not appear all at once. The reading
rises into the column: the source line settles first as a hairline, then the
measured values latch in place with tabular numerals. Ambiguity is resolved in
the same column — candidate places arrive as selectable readings, not a modal.

CROSS-SURFACE REACH: the graticule, the parcel blue, the tabular data block
and the hairline rule system carry to any future forecast table, alert list or
station page without redrawing the world.

RISK: a gradient ground plus a graticule can drift into decoration. It is held
by keeping both under 5% contrast against the surface, letting the conversation
own the page, and spending the signal colour only on action and state.

SEED: 33db2e6b · assigned index 7 (grounded list, Operate)

## Unresolved

- Geoapify is not adopted: the existing Open-Meteo gazetteer already returns
  ranked candidates with admin1/country, which is what ambiguity resolution
  needs. Revisit only if street-level or landmark resolution is required.
