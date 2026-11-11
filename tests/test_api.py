from datetime import date

import psycopg
from fastapi.testclient import TestClient

from leaselens import api
from leaselens.qa import DISCLAIMER, Answer
from leaselens.store import StoredChunk

client = TestClient(api.app)


def source(section: str = "1950.5") -> StoredChunk:
    return StoredChunk(
        section=section,
        heading="Hiring of Real Property",
        text="Within 21 calendar days after the tenant vacates.",
        source_url=f"https://example.test/{section}",
        as_of=date(2026, 1, 1),
        distance=0.2,
    )


def stub(monkeypatch, answer: Answer) -> list[str]:
    seen: list[str] = []

    def fake(question: str) -> Answer:
        seen.append(question)
        return answer

    monkeypatch.setattr(api, "ask_service", fake)
    return seen


def test_answer_is_returned_with_sources_and_disclaimer(monkeypatch):
    stub(
        monkeypatch,
        Answer(text="Return within 21 days [Civ. Code §1950.5].", sources=[source()], refused=False),
    )

    body = client.post("/ask", json={"question": "Is my whole deposit legal to keep?"}).json()

    assert body["answer"].startswith("Return within 21 days")
    assert body["refused"] is False
    assert body["fallback"] is False
    assert body["disclaimer"] == DISCLAIMER
    assert body["sources"][0]["section"] == "1950.5"
    assert body["sources"][0]["as_of"] == "2026-01-01"


def test_duplicate_sources_are_collapsed(monkeypatch):
    stub(monkeypatch, Answer(text="x", sources=[source(), source()], refused=False))

    body = client.post("/ask", json={"question": "deposit question"}).json()

    assert len(body["sources"]) == 1


def test_fallback_flag_is_passed_through(monkeypatch):
    stub(monkeypatch, Answer(text="quoted text", sources=[source()], refused=False, fallback=True))

    body = client.post("/ask", json={"question": "deposit question"}).json()

    assert body["fallback"] is True


def test_question_is_stripped_before_the_pipeline(monkeypatch):
    seen = stub(monkeypatch, Answer(text="x", sources=[], refused=True))

    client.post("/ask", json={"question": "   my deposit?   "})

    assert seen == ["my deposit?"]


def test_short_question_is_rejected(monkeypatch):
    stub(monkeypatch, Answer(text="x", sources=[], refused=True))

    response = client.post("/ask", json={"question": "hi"})

    assert response.status_code == 422


def test_overlong_question_is_rejected(monkeypatch):
    stub(monkeypatch, Answer(text="x", sources=[], refused=True))

    response = client.post("/ask", json={"question": "a" * (api.MAX_QUESTION_CHARS + 1)})

    assert response.status_code == 422


def test_database_outage_returns_503(monkeypatch):
    def down(question: str) -> Answer:
        raise psycopg.OperationalError("connection refused")

    monkeypatch.setattr(api, "ask_service", down)

    response = client.post("/ask", json={"question": "deposit question"})

    assert response.status_code == 503
