# LeaseLens

Plain-English answers to California tenant-rights questions, with each answer tied to the Civil Code section it came from.

Legal information, not legal advice.

## Setup

```bash
cp .env.example .env          # then add your GROQ_API_KEY (free at console.groq.com)
docker compose up -d db       # Postgres with pgvector
uv sync
```

## Load the statutes

Download the latest bulk file from the official leginfo site (`pubinfo_YYYY.zip`) into `data/raw/`, then:

```bash
uv run leaselens ingest --zip data/raw/pubinfo_2025.zip
```

## Ask a question

```bash
uv run leaselens ask "My landlord kept my whole deposit. Is that legal?"
```

## Tests

```bash
uv run pytest
uv run ruff check .
```

## Run the web app locally

```bash
uv run uvicorn leaselens.api:app --port 8000   # API
cd web && npm install && npm run dev            # web app at http://localhost:3000
```

By default the app quotes matching statute text and does not call an AI model, so no API key is needed. To turn on AI summaries, set `ANSWER_WITH_LLM=true` and `GROQ_API_KEY` in `.env`.

## Deploy

- **API (Render, or any container host):** build from the repository root with the `Dockerfile`. Set `DATABASE_URL` to your Postgres (with pgvector) and `ALLOWED_ORIGINS` to your web app's address, for example `https://your-app.vercel.app`. The container uses about 200 MB of memory.
- **Web app (Vercel):** deploy the `web/` folder. Set `NEXT_PUBLIC_API_URL` to the API's address.
- **Database:** load the statutes once with `uv run leaselens ingest --zip data/raw/pubinfo_2025.zip`, pointing `DATABASE_URL` at the hosted database.
