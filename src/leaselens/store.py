from dataclasses import dataclass
from datetime import date

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector


@dataclass(frozen=True)
class StoredChunk:
    section: str
    heading: str
    text: str
    source_url: str
    as_of: date
    distance: float


@dataclass(frozen=True)
class ChunkRecord:
    section: str
    heading: str
    chunk_index: int
    text: str
    source_url: str
    as_of: date
    embedding: list[float]


def connect(database_url: str) -> psycopg.Connection:
    conn = psycopg.connect(database_url)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def init_schema(conn: psycopg.Connection, embedding_dim: int) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS chunks (
            id          BIGSERIAL PRIMARY KEY,
            section     TEXT NOT NULL,
            heading     TEXT NOT NULL,
            chunk_index INT NOT NULL,
            text        TEXT NOT NULL,
            source_url  TEXT NOT NULL,
            as_of       DATE NOT NULL,
            embedding   vector({embedding_dim}) NOT NULL,
            UNIQUE (section, chunk_index)
        )
        """
    )
    # Full-text column for hybrid search. Added with IF NOT EXISTS so existing tables pick it up.
    conn.execute(
        """
        ALTER TABLE chunks ADD COLUMN IF NOT EXISTS fts tsvector
            GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS chunks_fts_idx ON chunks USING GIN (fts)")
    conn.commit()


def replace_chunks(conn: psycopg.Connection, records: list[ChunkRecord]) -> None:
    """Replace the whole corpus in one transaction so a failed ingest leaves the old data intact."""
    with conn.transaction():
        conn.execute("TRUNCATE chunks")
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO chunks (section, heading, chunk_index, text, source_url, as_of, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    (
                        r.section,
                        r.heading,
                        r.chunk_index,
                        r.text,
                        r.source_url,
                        r.as_of,
                        Vector(r.embedding),
                    )
                    for r in records
                ],
            )


# Reciprocal rank fusion constant: a standard choice that keeps one list from dominating.
RRF_K = 60
# How many candidates each side contributes before fusion.
CANDIDATES = 20


def search(
    conn: psycopg.Connection,
    query_embedding: list[float],
    k: int,
    query_text: str | None = None,
) -> list[StoredChunk]:
    """Vector search, or hybrid (vector + Postgres full-text) when query_text is given.

    Hybrid uses reciprocal rank fusion, so a chunk that matches the query words exactly
    can rank high even when its embedding is not the closest. Distance is always the
    vector distance, so the refusal threshold means the same thing in both modes.
    """
    query = Vector(query_embedding)
    if query_text is None:
        rows = conn.execute(
            """
            SELECT section, heading, text, source_url, as_of, embedding <=> %s AS distance
            FROM chunks
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (query, query, k),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            WITH vec AS (
                SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> %(q)s) AS rank
                FROM chunks
                ORDER BY embedding <=> %(q)s
                LIMIT %(pool)s
            ),
            kw AS (
                SELECT id, ROW_NUMBER() OVER (ORDER BY ts_rank(fts, query) DESC) AS rank
                FROM chunks, websearch_to_tsquery('english', %(text)s) AS query
                WHERE fts @@ query
                ORDER BY ts_rank(fts, query) DESC
                LIMIT %(pool)s
            ),
            fused AS (
                SELECT COALESCE(vec.id, kw.id) AS id,
                       COALESCE(1.0 / (%(rrf)s + vec.rank), 0)
                     + COALESCE(1.0 / (%(rrf)s + kw.rank), 0) AS score
                FROM vec FULL OUTER JOIN kw ON vec.id = kw.id
            )
            SELECT c.section, c.heading, c.text, c.source_url, c.as_of,
                   c.embedding <=> %(q)s AS distance
            FROM fused
            JOIN chunks c ON c.id = fused.id
            ORDER BY fused.score DESC
            LIMIT %(k)s
            """,
            {
                "q": query,
                "text": query_text,
                "pool": CANDIDATES,
                "rrf": RRF_K,
                "k": k,
            },
        ).fetchall()
    return [
        StoredChunk(
            section=section,
            heading=heading,
            text=text,
            source_url=source_url,
            as_of=as_of,
            distance=float(distance),
        )
        for section, heading, text, source_url, as_of, distance in rows
    ]
