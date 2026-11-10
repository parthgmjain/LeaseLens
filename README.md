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
