# WeatherGPT — design system

The visual world of the client at `GET /voice`
(`backend/app/voice/page.html`, `styles.css`, `app.js`). Written from the built
interface, not ahead of it.

## The idea

**The page is an atmospheric sounding.** One continuous vertical column of air,
cool at the top and warm at the surface, ruled by a hairline skew-T graticule.
The conversation ascends through that column; the composer sits at the surface,
where the air is warmest and the graticule densest. The instrument reading — a
temperature latched beside its condition — is the moment the page is built
around.

This is an **Operate** surface. Expression lives in precision, not decoration.
A person is in a task, and the interface disappears into it.

## Colour

Strategy: **Restrained** — a light instrument ground, near-black ink, and one
signal colour.

| Token | Value | Used for |
| --- | --- | --- |
| `--parcel` | `#1d63e6` | The parcel path. Primary action, active state, current selection. Nothing decorative. |
| `--parcel-hover` / `--parcel-active` | `#164fbd` / `#123f97` | Pressed and selected-on-tint states |
| `--parcel-soft` / `--parcel-soft-line` | `#e8f0fe` / `#c3d8fb` | Selected surfaces and their hairlines |
| `--stop` | `#c7301f` | Stop, and error only |
| `--warn` / `--ok` | `#8a5a00` / `#17694a` | Advisory note, success status |
| `--ink` / `--ink-2` / `--ink-3` | `#0e1720` / `#45545f` / `#5c6b78` | Body, secondary, label |
| `--line` / `--line-strong` | 10% / 18% ink | Rules and control borders |
| `--surface` / `--surface-quiet` / `--surface-sunk` | `#fff` / `#f7f9fc` / `#f1f4f8` | Cards, footers, inputs |

Light only, declared with `color-scheme: light`. The use scene is daylight and a
one-answer task; a printed instrument does not have a night mode.

Every measured pair clears WCAG AA. The tightest is the selected candidate
chip's text on its own tint at 4.6:1; the rest sit above 5:1.

### The ground

`--air-top` → `--air-mid` → `--air-surface` is a three-stop vertical gradient on
a fixed `.sky` layer. Seven **sky states** swap only those three stops and the
graticule opacity: `clear · clouds · rain · storm · snow · fog · night`. Ink
never moves. The state is set on `<html data-sky>` from the WMO condition
category the backend returned, and from `is_day`.

`repeating-linear-gradient` earns its place here: a sounding chart *is* a
measuring grid. It is held under 10% and masked away below the fold, and becomes
legible only in the masthead band.

## Type

One family: `system-ui` through a Segoe/Roboto/Helvetica stack, with
`"Noto Sans Devanagari"` so a Hindi answer keeps the same voice.

Fixed rem-adjacent scale, ratio ≈ 1.2: 10.5 · 11 · 12 · 12.5 · 13.5 · 14.5 · 15
· 15.5 · 21 · 30 · 40. No fluid headings — users view product UI at a consistent
DPI.

- **Reading** (40px / 600 / `-0.032em`) — the one display moment.
- **Opening title** (30px / 620 / `-0.028em`, `text-wrap: balance`).
- **Field labels** (10.5px / 620 / `0.07em` / uppercase).
- Every figure carries `font-variant-numeric: tabular-nums` so columns of
  measurements read down the page.
- Tracking floor `-0.032em`. Prose stays inside 62–70ch.

## Material and depth

Elevation is declared once per element. Cards carry a hairline **and** a soft
offset shadow only where they float above the ground (`--shadow-raise`);
popovers use `--shadow-float`. Radii: 16px cards, 12px groups, 8px controls,
pills only for chips and candidates.

The agent card's notch (`4px` top-left) points back at the WeatherGPT label
above it; the user bubble's notch (`4px` bottom-right) points at its own side.
That is the whole of the left/right language.

Browser surfaces are themed from the palette: selection, caret, scrollbar
thumb, focus ring.

## Icons

One authored SVG set on a 24px grid at 1.6 stroke, `currentColor`, defined once
as `<symbol>` and referenced with `<use>`. Weather glyphs are drawn in the same
hand as the controls. No emoji, no icon library.

## Components

Every interactive element ships default, hover, focus-visible, active, disabled
— and, where it applies, loading and listening.

- **`.btn`** — one shape, 40px (44px on touch). `--primary` blue, neutral
  default, `--danger` for Stop. Speak / Replay / Stop share height, padding,
  radius, type and icon alignment.
- **`.pill`** — masthead readouts that open a popover. `aria-expanded` rotates
  the chevron.
- **`.composer`** — auto-growing textarea, dictate and send. Focus lifts a 3px
  `--parcel-soft` ring.
- **`.card`** — the answer. Body, then optional disambiguation, then the source
  footer on a hairline. Fixed 704px measure so it does not resize between its
  loading state and its result.
- **`.metrics`** — auto-fit grid of measured values; cells draw their own rules
  via a 1px spread shadow, so an uneven last row leaves surface, not a gap.
- **`.rows`** — the ruled forecast table; figures sit in fixed 80px tracks so
  min, max and rain read as columns.

Empty state teaches the column. Loading shows a working line plus a skeleton —
never a value that has not arrived. Errors name the problem and the recovery,
never the exception.

## Motion

150–320ms, `cubic-bezier(0.16, 1, 0.3, 1)`. State only.

The one authored moment: an answer **rises into the column** — the turn lifts
7px into place, then the measured values latch in on a 45ms stagger. Precipitation
drifts on the ground layer for rain, storm and snow, as a single masked hairline
layer, never a particle system.

`prefers-reduced-motion` collapses every duration to 1ms and stops the drift,
the pulse, the skeleton sweep and the latch.

## Layout

`100dvh` grid: masthead / scrolling column / composer dock. The column is
`min-width: 0` so nothing can widen the application past the viewport.

`--column: 960px` governs the masthead, the conversation and the composer alike,
so the brand, the first answer and the input all hang off one left edge.

Breakpoints are structural: 1120 (padding), 900 (tagline out), 760 (masthead
wraps, forecast figures drop to their own row, chips become a scroller), 430
(control padding), 340 (button labels out, names kept).

## Rules this system keeps

- No weather value is authored in the client. Figures are copied from the typed
  backend response for the turn, or from the answer the backend rendered.
- Time is 12-hour with AM/PM, dates are DD-MM-YYYY, both in the resolved
  location's timezone. Presentation only — nothing is sent back reformatted.
- Provenance is part of the answer, never a separate box, and never carries a
  URL, an id, a request trace or a provider's internal model slug.
- Icons always carry a text label or an `aria-label`; nothing is icon-only above
  340px.
