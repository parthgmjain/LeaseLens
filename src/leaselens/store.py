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
    chunk_index: int = 0


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
            SELECT section, heading, text, source_url, as_of, embedding <=> %s AS distance, chunk_index
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
                   c.embedding <=> %(q)s AS distance, c.chunk_index
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
            chunk_index=chunk_index,
        )
        for section, heading, text, source_url, as_of, distance, chunk_index in rows
    ]


def neighboring_chunks(conn: psycopg.Connection, hits: list[StoredChunk]) -> list[StoredChunk]:
    """Each hit plus the chunk before and after it in the same section.

    A rule can sit at the edge of a chunk, so the neighbour may hold the sentence the
    question needs. Results keep the sections in the order they were first hit, and
    chunks in statute order within each section.
    """
    section_rank: dict[str, int] = {}
    for h in hits:
        section_rank.setdefault(h.section, len(section_rank))

    found: dict[tuple[str, int], StoredChunk] = {}
    for h in hits:
        for index in (h.chunk_index - 1, h.chunk_index, h.chunk_index + 1):
            if index < 0 or (h.section, index) in found:
                continue
            row = conn.execute(
                "SELECT section, heading, text, source_url, as_of FROM chunks "
                "WHERE section = %s AND chunk_index = %s",
                (h.section, index),
            ).fetchone()
            if row:
                section, heading, text, source_url, as_of = row
                found[(h.section, index)] = StoredChunk(
                    section=section,
                    heading=heading,
                    text=text,
                    source_url=source_url,
                    as_of=as_of,
                    distance=h.distance,
                    chunk_index=index,
                )
    return sorted(
        found.values(),
        key=lambda c: (section_rank[c.section], c.chunk_index),
    )
