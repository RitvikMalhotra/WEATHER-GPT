# WeatherGPT

WeatherGPT is a conversational weather intelligence platform for India. Users can ask plain-language weather questions in English or Hindi and receive grounded answers based on validated meteorological data from trusted providers, with clear source attribution and no fabricated weather values.

## Why it exists

The core design principle is simple:

- A language model may choose which read-only tool to call.
- It does not invent weather numbers.
- Every value shown to the user is copied from a backend response derived from meteorological providers.
- The app clearly shows the data source, timestamp, and provenance for each answer.

This makes WeatherGPT suitable for trustworthy weather assistance, especially for local and ambiguous place names across India.

## Features

- Natural-language weather Q&A via FastAPI
- English and Hindi support
- Voice-first client experience
- Location search and disambiguation for shared place names
- Forecast, current conditions, historical weather, and alerts
- Source attribution and provider provenance
- Postgres + PostGIS data layer for spatial weather queries
- Optional AI routing layer using OpenAI-compatible or Groq providers

## Architecture

The project is split into a backend API and a lightweight browser client.

- Backend: `backend/`
- API entry point: `backend/app/main.py`
- Client: `backend/app/voice/`
- Database migrations: `backend/alembic/`
- Shared docs: `DESIGN.md`, `PRODUCT.md`

### Stack

- Python 3.x
- FastAPI
- SQLAlchemy
- PostgreSQL + PostGIS
- Alembic
- Geoapify, Open-Meteo, NOAA GFS, IMD integrations
- Optional local or hosted model routing for AI intent selection

## Quick start

From the repository root:

```bash
# 1. Start PostgreSQL
docker compose up -d --wait postgres

# 2. Set up the backend environment
cd backend
python -m venv .venv
# Windows
# .venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env

# 3. Apply schema migrations
alembic upgrade head

# 4. Run the API
uvicorn app.main:app --reload
```

Then open:

- API docs: http://127.0.0.1:8000/docs
- Voice client: http://127.0.0.1:8000/voice

## Environment configuration

The backend reads configuration from environment variables and a local `.env` file. Common options include:

```bash
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/weathergpt
GEOAPIFY_API_KEY=your_key_here
AI_LLM_PROVIDER=groq
GROQ_API_KEY=your_key_here
AI_BACKEND_BASE_URL=http://127.0.0.1:8000
```

You can also run the app without a database URL; the service still supports core weather endpoints, but database-backed features may return a `503 DATABASE_UNAVAILABLE` response.

## AI layer

The AI route is intentionally constrained:

- It may select a read-only tool to answer user questions.
- It may not generate weather values by itself.
- It uses the backend's validated weather data as the source of truth.
- It can work with local OpenAI-compatible endpoints or hosted models such as Groq.

This keeps the assistant grounded and prevents fabricated meteorological responses.

## Project structure

```text
WEATHER-GPT/
├── DESIGN.md
├── PRODUCT.md
├── docker-compose.yml
├── package.json
├── backend/
│   ├── alembic/
│   ├── app/
│   ├── scripts/
│   ├── tests/
│   ├── README.md
│   ├── requirements.txt
│   ├── pyproject.toml
│   └── alembic.ini
├── scripts/
│   └── dev.mjs
└── README.md
```

## Development notes

- The backend is the authoritative source for weather data and validation logic.
- The user-facing client is intentionally thin and does not contain weather logic.
- The project favors deterministic responses, provider traceability, and safe AI usage patterns.
- The repository includes focused tests for weather logic, providers, persistence, alerts, and API contracts.

## License

This project does not currently declare a repository license in the top-level files. If you plan to publish or distribute it, add a license file and update this section accordingly.

## Contributing

To contribute locally:

1. Start the database and backend.
2. Run the relevant tests under `backend/tests/`.
3. Keep provider facts validated and traceable.
4. Avoid introducing AI-generated weather values into the response pipeline.

---

WeatherGPT is designed around trust: the model can route and explain, but the weather itself comes from validated sources, not model memory.
