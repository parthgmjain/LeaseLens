import os
from datetime import date

import psycopg
import pytest

from leaselens.store import ChunkRecord, connect, init_schema, replace_chunks, search

# The fixture drops the chunks table, so tests must never point at the real corpus DB.
DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql://leaselens:leaselens@localhost:5433/leaselens_test"
)
DIM = 384


def unit(index: int) -> list[float]:
    vector = [0.0] * DIM
    vector[index] = 1.0
    return vector


def record(section: str, index: int, vector_index: int) -> ChunkRecord:
    return ChunkRecord(
        section=section,
        heading="Security deposits",
        chunk_index=index,
        text=f"text of {section} chunk {index}",
        source_url="https://example.test",
        as_of=date(2025, 1, 1),
        embedding=unit(vector_index),
    )


@pytest.fixture
def conn():
    try:
        connection = connect(DATABASE_URL)
    except psycopg.OperationalError as exc:  # database not running locally
        pytest.skip(f"Postgres not reachable: {exc}")
    init_schema(connection, DIM)
    yield connection
    connection.execute("DROP TABLE IF EXISTS chunks")
    connection.commit()
    connection.close()


def test_search_returns_nearest_chunk_first(conn):
    replace_chunks(conn, [record("1950.5", 0, 0), record("1941", 0, 1)])

    hits = search(conn, unit(1), k=2)

    assert [h.section for h in hits] == ["1941", "1950.5"]
    assert hits[0].distance == pytest.approx(0.0, abs=1e-6)


def test_replace_chunks_overwrites_the_corpus(conn):
    replace_chunks(conn, [record("1950.5", 0, 0)])
    replace_chunks(conn, [record("1941", 0, 1)])

    hits = search(conn, unit(0), k=10)

    assert [h.section for h in hits] == ["1941"]


def test_failed_replace_keeps_old_data(conn):
    replace_chunks(conn, [record("1950.5", 0, 0)])
    bad = record("1941", 0, 1)
    broken = ChunkRecord(**{**bad.__dict__, "embedding": [0.0, 0.0]})

    with pytest.raises(psycopg.DataError):
        replace_chunks(conn, [broken])

    hits = search(conn, unit(0), k=10)
    assert [h.section for h in hits] == ["1950.5"]
