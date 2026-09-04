# WeatherGPT

## What it is

A conversational meteorological intelligence platform. A person asks a weather
question in plain language — typed or spoken — and gets an answer whose every
number was fetched from a validated meteorological source in that same turn.

## Platform

web

## The unique mechanism

A language model may only *choose which read-only tool to call*. It never
authors a weather value, a forecast, or an alert. Every figure on screen is
copied from a typed backend response, and every answer names the source that
produced it. That boundary is the product: trustworthy AI weather.

## Who uses it

People in India asking about their own locality — commuters, farmers,
travellers, and people planning a day. They ask in English or Hindi, often by
voice, often on a phone, often about a neighbourhood (not a metro) whose name
is shared by several places in the country.

## The scene

A person on a laptop or phone, in daylight, wanting one answer fast. They are
not exploring a dashboard; they are in a task. They re-ask, follow up, and
change locations.

## Jobs

- Ask current conditions, hourly/daily forecast, stored history, or a
  travel/agriculture consideration for a place.
- Pin an exact place when the name is ambiguous ("Miyapur" exists in several
  Indian states) rather than silently getting the wrong one.
- Hear the answer read aloud, and re-hear or stop it.
- See where the number came from, and how fresh it is.

## Truth constraints

- No weather value may be invented, interpolated, or recalled from model
  memory. Backend failure means no answer, never a guess.
- The source is whatever the backend names for that turn (Open-Meteo, NOAA GFS,
  India Meteorological Department). It is never hardcoded.
- WeatherGPT rule alerts are not official meteorological warnings, and the
  backend's own disclaimer text ships verbatim.
- Units are the backend's units (°C, m/s, mm, %); the presentation layer
  reformats time and date only.

## Locale

Indian conventions: 12-hour clock with AM/PM, DD-MM-YYYY dates. English is the
default language; Hindi is fully supported end to end.

## Constraints

- The FastAPI backend is frozen. The client uses `/api/v1/ai/chat` and
  `/api/v1/locations/search` and adds no new weather logic.
- No build step, no framework, no bundler. Plain HTML/CSS/JS served by the
  existing FastAPI route.
- Groq calls cost real credits; the interface must be developable and testable
  without spending them.
