from datetime import date

import psycopg
import pytest
from fastapi.testclient import TestClient

from leaselens import api
from leaselens.config import Settings
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


@pytest.fixture(autouse=True)
def fresh_limiter(monkeypatch):
    # Each test starts with empty counters, so one test's requests don't affect another.
    monkeypatch.setattr(api, "limiter", api.QuestionLimiter())


def test_per_ip_limit_returns_429_after_the_window_quota(monkeypatch):
    stub(monkeypatch, Answer(text="x", sources=[], refused=True))
    monkeypatch.setattr(api, "get_settings", lambda: Settings(ask_limit_per_ip=2))

    codes = [client.post("/ask", json={"question": "deposit question"}).status_code for _ in range(3)]

    assert codes == [200, 200, 429]


def test_daily_cap_blocks_further_questions():
    limiter = api.QuestionLimiter()
    now = 1_800_000_000.0  # a fixed moment in UTC

    assert limiter.check("a", now, per_ip=10, window_seconds=600, daily_cap=2) is None
    assert limiter.check("b", now, per_ip=10, window_seconds=600, daily_cap=2) is None
    assert "daily" in limiter.check("c", now, per_ip=10, window_seconds=600, daily_cap=2)


def test_ip_limit_recovers_after_the_window():
    limiter = api.QuestionLimiter()
    now = 1_800_000_000.0

    assert limiter.check("a", now, per_ip=1, window_seconds=600, daily_cap=100) is None
    assert limiter.check("a", now + 1, per_ip=1, window_seconds=600, daily_cap=100) is not None
    assert limiter.check("a", now + 601, per_ip=1, window_seconds=600, daily_cap=100) is None


def test_daily_counter_resets_on_a_new_day():
    limiter = api.QuestionLimiter()
    now = 1_800_000_000.0

    assert limiter.check("a", now, per_ip=10, window_seconds=600, daily_cap=1) is None
    assert limiter.check("b", now, per_ip=10, window_seconds=600, daily_cap=1) is not None
    assert limiter.check("b", now + 86_400, per_ip=10, window_seconds=600, daily_cap=1) is None


class _NullConnection:
    def __enter__(self):
        return object()

    def __exit__(self, *exc):
        return False


def _wire_pipeline(monkeypatch, answers: list[Answer]) -> list[str]:
    """Replace the real pipeline with a counter so caching can be tested without a database or LLM."""
    calls: list[str] = []

    def fake_answer(question, conn, embed, generate, settings):
        calls.append(question)
        return answers[len(calls) - 1]

    monkeypatch.setattr(api, "answer_question", fake_answer)
    monkeypatch.setattr(api, "connect", lambda url: _NullConnection())
    monkeypatch.setattr(api, "make_embedder", lambda model: None)
    monkeypatch.setattr(api, "make_generator", lambda client, model: None)
    api._answer_cache.clear()
    return calls


def test_repeated_question_is_served_from_cache(monkeypatch):
    calls = _wire_pipeline(monkeypatch, [Answer(text="A", sources=[], refused=True)])

    first = api.ask_service("My landlord kept my deposit?")
    second = api.ask_service("  my landlord KEPT my deposit?  ")

    assert calls == ["My landlord kept my deposit?"]
    assert first is second


def test_fallback_answers_are_not_cached(monkeypatch):
    fallback = Answer(text="quoted", sources=[], refused=False, fallback=True)
    real = Answer(text="summary", sources=[], refused=False)
    calls = _wire_pipeline(monkeypatch, [fallback, real])

    api.ask_service("deposit question")
    later = api.ask_service("deposit question")

    assert len(calls) == 2
    assert later.text == "summary"


def test_excerpt_mode_does_not_need_a_groq_key(monkeypatch):
    # Regression: the API used to build the Groq client on every request, which crashed
    # when no key was set. Excerpt-only mode must work with no key at all.
    monkeypatch.setattr(api, "get_settings", lambda: Settings(answer_with_llm=False, groq_api_key=""))
    monkeypatch.setattr(api, "_groq_client", lambda: (_ for _ in ()).throw(AssertionError("built")))
    monkeypatch.setattr(api, "connect", lambda url: _NullConnection())
    monkeypatch.setattr(api, "make_embedder", lambda model: None)
    monkeypatch.setattr(api, "answer_question", lambda *a, **k: Answer(text="ok", sources=[], refused=True))
    api._answer_cache.clear()

    assert api.ask_service("deposit question").text == "ok"
