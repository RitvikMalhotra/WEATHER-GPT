# WeatherGPT Backend

Meteorological intelligence core: ingests weather from external providers,
normalises and validates it into one canonical model, persists it with full
provenance, and serves it over a versioned HTTP API.

```
Weather Provider  →  Adapter  →  Normalisation  →  Validation
                                                       ↓
                                              Canonical Model
                                                       ↓
                              ┌────────────────────────┴──────────┐
                              ↓                                   ↓
                     Persistence Service                    Service Layer
                              ↓                                   ↓
                    PostgreSQL + PostGIS  ────────────────→   FastAPI
```

Two rules the whole design exists to enforce:

1. **A language model is never the source of a weather value.** Numbers come
   from meteorological providers, through validation. The database stores
   validated data; it does not generate predictions.
2. **Invalid data is never stored or served.** The validation engine is a gate,
   not a report.

---

## Quick start

```bash
# 1. Database
docker compose up -d --wait postgres          # from the repository root

# 2. Backend
cd backend
python -m venv .venv && source .venv/bin/activate   # Scripts/activate on Windows
pip install -r requirements.txt
cp .env.example .env

# 3. Schema
alembic upgrade head

# 4. Run
uvicorn app.main:app --reload
```

API docs at `http://127.0.0.1:8000/docs`.

Without `DATABASE_URL` the service still runs: current conditions, forecasts and
location search work as before, and `/api/v1/weather/historical` returns a clean
`503 DATABASE_UNAVAILABLE`.

---

## Database architecture

### Why PostgreSQL

Weather data is relational, heavily queried by time range, and needs real
constraints. The uniqueness rules that make ingestion idempotent
(`ON CONFLICT DO UPDATE` against a composite unique index) are enforced by the
database rather than by application logic that races under concurrency. Native
`timestamptz` handling matters for a system whose records are timestamped by
providers in local time and stored in UTC.

### Why PostGIS

Every record is a point on the Earth, and the questions asked of them are
spatial: *what was recorded near here?* A provider serves the nearest grid point
to a request, and that point shifts slightly between requests — so history
cannot be looked up by coordinate equality, only by proximity.

`ST_DWithin` on a `geography(Point, 4326)` column, backed by a GIST index,
answers that in the database. The alternative — read every row, compute
great-circle distance in Python, discard most of it — reads the whole table for
every request. PostGIS also gives correct spheroid distances for free;
hand-rolled haversine is a recurring source of subtle error.

### Schema

**`weather_observations`** — conditions at a place and instant.

| Group | Columns |
| --- | --- |
| Identity | `id` |
| Place | `location_key`, `latitude`, `longitude`, `elevation_m`, `geom`, `timezone` |
| Time | `observed_at`, `ingested_at` |
| Measurements | `temperature_c`, `apparent_temperature_c`, `dew_point_c`, `relative_humidity_pct`, `pressure_msl_hpa`, `surface_pressure_hpa`, `wind_speed_ms`, `wind_gust_ms`, `wind_direction_deg`, `precipitation_mm`, `cloud_cover_pct`, `visibility_m`, `uv_index`, `is_day` |
| Condition | `condition`, `condition_description`, `wmo_code` |
| Provenance | `provider_id`, `provider_name`, `model`, `fetched_at`, `source_url`, `license`, `attribution` |

**`weather_forecasts`** — one row per predicted point, at hourly or daily
resolution. Carries the same place, condition and provenance groups, plus:

| Column | Meaning |
| --- | --- |
| `forecast_created_at` | when the prediction was produced |
| `forecast_for` | the instant being predicted |
| `resolution` | `hourly` or `daily` |

Keeping those two timestamps separate is what makes forecast-skill analysis
possible later: you cannot ask *how good was last Tuesday's three-day forecast*
if only the newest prediction is retained.

### Design decisions

**Coordinates are stored twice, deliberately.** `latitude`/`longitude` are what
gets returned to callers; `geom` exists only for spatial predicates. Reads never
parse geometry, and distance queries never leave the database.

**`location_key` identifies a place.** Floats make a poor unique key, so a place
is its coordinates rounded to four decimals (~11 m) — the same key the response
cache uses, so both layers agree on what "the same place" means.

**Missing stays missing.** Every measurement is nullable. An unreported value
must not become a zero: "no precipitation reported" and "0 mm of precipitation"
are different facts.

**An undisclosed model is stored as `''`, not `NULL`.** NULLs never compare
equal, so a NULL here would silently disable the uniqueness constraints. It maps
back to `None` on the way out.

### Indexes

| Index | Serves |
| --- | --- |
| `uq_weather_observations_identity` | idempotent ingestion (provider, model, place, instant) |
| `ix_weather_observations_location_key_observed_at` | "what was recorded here, over this window" |
| `ix_weather_observations_observed_at` | time-range scans across places |
| `ix_weather_observations_provider_id` | per-source filtering |
| `ix_weather_observations_geom` (GIST) | `ST_DWithin` / `ST_Distance` |
| `uq_weather_forecasts_identity` | idempotent forecast ingestion |
| `ix_weather_forecasts_location_key_forecast_for` | "what is predicted here" |
| `ix_weather_forecasts_forecast_created_at_forecast_for` | forecast-skill analysis |
| `ix_weather_forecasts_provider_id`, `ix_weather_forecasts_geom` | filtering, spatial |

---

## Persistence flow

```
Request → Provider → Normalise → Validate → Persist → Respond
```

Persistence is invoked by `WeatherService` **inside the cache loader**, which
gives three properties for free:

- **Invalid data cannot be persisted.** The loader only runs the ingestion
  pipeline, which returns nothing that failed validation. There is no code path
  from a provider to a table that skips the gate — it is structural, not a check.
- **Cache hits do not rewrite rows.** A repeat inside the TTL touches neither the
  provider nor the database.
- **A write failure never breaks a read.** Current conditions and forecasts are
  answered from provider data already in hand and already valid. If the database
  is unreachable, the caller still gets their weather and the failure is logged.
  Only `/weather/historical`, where the database *is* the source, hard-fails.

### Cache vs database

| | Purpose | Scope | Lifetime |
| --- | --- | --- | --- |
| `TTLCache` | avoid repeating an upstream request | per instance | seconds–minutes |
| PostgreSQL | durable history and spatial queries | shared | indefinite |

Both sit behind `WeatherService`. The cache is reached only through
`TTLCache.get_or_load`, so replacing it with Redis later changes neither the
service's signature nor its callers.

### Idempotency

Re-ingesting the same observation updates the row rather than appending one.

Forecasts are harder: providers rarely publish which model run produced a
prediction, so the true generation time is unknown. The fetch time is bucketed
(`FORECAST_GENERATION_BUCKET_MINUTES`, default 60) into a stable stand-in.
Repeated requests within a window update one set of rows; a later window appends
a new generation. That stops API traffic from multiplying rows without
collapsing the forecast history.

Verified: four identical round trips with caching disabled produce one
observation row and one forecast generation.

---

## Provenance

Every persisted record answers three questions:

| Question | Column |
| --- | --- |
| Where did this value come from? | `provider_id`, `provider_name`, `source_url` |
| Which model produced it? | `model` |
| When was it obtained? | `fetched_at`, `ingested_at` |

Plus `license` and `attribution`, because openly-licensed data carries
redistribution obligations that must survive into anything built on it.

Persistence never strips provenance — mapping is pure and directly tested.
Reconstructed records report `cached: false`, since history comes from durable
storage, not the response cache.

---

## Historical API

```
GET /api/v1/weather/historical?latitude=&longitude=&start=&end=&radius_km=&provider=
```

```bash
curl "http://127.0.0.1:8000/api/v1/weather/historical\
?latitude=17.385&longitude=78.4867&start=2026-08-01&end=2026-08-31"
```

- `start` / `end` accept a date (`2026-08-01`) or a timestamp
  (`2026-08-01T06:00:00Z`). Both bounds are inclusive; a bare `end` date covers
  the whole day. Naive timestamps are read as UTC.
- `radius_km` defaults to `HISTORY_DEFAULT_RADIUS_KM` and is capped at
  `HISTORY_MAX_RADIUS_KM`. The radius actually used is reported back.
- The response carries the requested coordinates, the interpreted window, the
  search radius, a count, and each record with its own provenance and distance.
- An empty result is a `200` with `count: 0` — "nothing recorded there" is a
  valid answer.

Records are response schemas built from the canonical model. No row ids, column
names or geometry are exposed.

---

## Failure behaviour

Database errors never reach a caller as SQL or a stack trace. SQLAlchemy embeds
the failing statement — and, depending on the driver, its parameters and the
connection string — in its exception messages, so `app/db/engine.py` translates
everything into:

```json
{"error": {"code": "DATABASE_UNAVAILABLE",
           "message": "Weather persistence is temporarily unavailable.",
           "request_id": "…",
           "details": {"operation": "history.observations"}}}
```

The real cause is logged server-side under the request's correlation id. The
translation deliberately catches more than `SQLAlchemyError`: asyncpg raises its
own exceptions at connect time, and one of those escaping would leak a host or
database name.

`/health/ready` includes a database probe. Unlike the upstream weather providers
— which are deliberately *not* probed, because a third-party outage would empty
the whole fleet at once with nowhere healthier to route — the database is
infrastructure this deployment owns, and an instance that cannot reach it is
often broken in a way rerouting fixes.

---

## Migrations

```bash
cd backend
alembic upgrade head                  # apply
alembic downgrade -1                  # roll back one
alembic revision -m "add x"           # new migration
alembic -x url=postgresql+asyncpg://… upgrade head   # explicit target
```

The URL comes from application settings, not `alembic.ini`, so migrations and
the running service can never disagree about which database they mean, and no
credential is committed.

`0001_initial_schema` creates the PostGIS extension, both tables, and all
indexes. It is hand-written: GeoAlchemy2 emits its own DDL for spatial columns,
and autogenerate against that produces migrations that create an index twice or
drop one they cannot recreate.

---

## Tests

```bash
pytest                    # default: offline, no infrastructure
```

The default suite requires no network and no database. Upstream providers are
driven through `httpx.MockTransport` against recorded payloads, so the provider,
retry layer, normaliser, validator, pipeline and routes all execute for real.

```bash
docker compose up -d --wait postgres
TEST_DATABASE_URL=postgresql+asyncpg://weathergpt:weathergpt@localhost:5432/postgres \
  pytest -m integration
```

Integration tests cover what cannot honestly be faked: `ST_DWithin` semantics,
`ON CONFLICT` against live constraints, transaction rollback, and whether the
migration produces a working schema. Each run creates a throwaway database,
migrates it with Alembic and drops it — so the suite also proves a fresh database
can be built from migrations alone. They skip cleanly when
`TEST_DATABASE_URL` is unset.

---

## Configuration

All configuration is environment-based; see `.env.example`. Nothing is
hard-coded and no secret is committed.

| Variable | Default | Purpose |
| --- | --- | --- |
| `DATABASE_URL` | *(unset)* | async PostgreSQL URL; unset disables persistence |
| `PERSISTENCE_ENABLED` | `true` | run against a database with writes off |
| `DATABASE_POOL_SIZE` / `_MAX_OVERFLOW` / `_POOL_TIMEOUT_SECONDS` | `5` / `10` / `10.0` | connection pool |
| `FORECAST_GENERATION_BUCKET_MINUTES` | `60` | forecast generation window |
| `HISTORY_DEFAULT_RADIUS_KM` / `_MAX_RADIUS_KM` | `25` / `500` | historical search radius |
| `HISTORY_MAX_RANGE_DAYS` / `_MAX_RESULTS` | `366` / `1000` | historical query bounds |

---

## Layout

```
backend/
├── app/
│   ├── api/v1/          routes and response schemas
│   ├── config/          settings, structured logging
│   ├── core/            errors, dependencies, composition root
│   ├── db/              models, mappers, repositories, engine
│   ├── domain/          the canonical meteorological model
│   ├── ingestion/       pipeline and validation engine
│   ├── providers/       provider contract and implementations
│   └── services/        weather, geocoding, persistence, history, cache
├── alembic/             migrations
└── tests/
```
