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


def search(
    conn: psycopg.Connection, query_embedding: list[float], k: int
) -> list[StoredChunk]:
    query = Vector(query_embedding)
    rows = conn.execute(
        """
        SELECT section, heading, text, source_url, as_of, embedding <=> %s AS distance
        FROM chunks
        ORDER BY embedding <=> %s
        LIMIT %s
        """,
        (query, query, k),
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
