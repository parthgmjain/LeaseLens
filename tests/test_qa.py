from datetime import date

from leaselens import qa
from leaselens.config import Settings
from leaselens.store import StoredChunk

SETTINGS = Settings(top_k=5, max_distance=0.5)


def hit(section: str, distance: float) -> StoredChunk:
    return StoredChunk(
        section=section,
        heading="Security deposits",
        text="A landlord may keep only part of a deposit for specified reasons.",
        source_url="https://example.test",
        as_of=date(2025, 1, 1),
        distance=distance,
    )


def run(hits, generate_reply):
    calls = []

    def fake_search(conn, vector, k):
        return hits

    def fake_generate(system, user):
        calls.append((system, user))
        return generate_reply

    original = qa.search
    qa.search = fake_search
    try:
        answer = qa.answer_question(
            "Can my landlord keep my whole deposit?",
            conn=None,
            embed=lambda texts: [[0.0] * 3 for _ in texts],
            generate=fake_generate,
            settings=SETTINGS,
        )
    finally:
        qa.search = original
    return answer, calls


def test_refuses_without_calling_llm_when_nothing_is_close():
    answer, calls = run([hit("1950.5", 0.9)], "should not be used")

    assert answer.refused is True
    assert answer.sources == []
    assert calls == []


def test_refuses_when_there_are_no_hits():
    answer, calls = run([], "should not be used")

    assert answer.refused is True
    assert calls == []


def test_answers_from_relevant_sources_with_citations():
    answer, calls = run(
        [hit("1950.5", 0.2)], "Your landlord can keep part of it [Civ. Code §1950.5]."
    )

    assert answer.refused is False
    assert answer.text.endswith("[Civ. Code §1950.5].")
    assert [s.section for s in answer.sources] == ["1950.5"]
    system, user = calls[0]
    assert "Answer ONLY from the numbered sources" in system
    assert "Civ. Code §1950.5" in user


def test_model_declining_returns_refusal():
    answer, _ = run([hit("1950.5", 0.2)], qa.INSUFFICIENT)

    assert answer.refused is True
    assert answer.text == qa.REFUSAL


def test_irrelevant_hits_are_excluded_from_sources():
    answer, _ = run(
        [hit("1950.5", 0.2), hit("1941", 0.8)], "Answer [Civ. Code §1950.5]."
    )

    assert [s.section for s in answer.sources] == ["1950.5"]


def test_every_answer_carries_the_disclaimer():
    answer, _ = run([hit("1950.5", 0.2)], "Answer [Civ. Code §1950.5].")

    assert "legal information, not legal advice" in answer.disclaimer
