from dataclasses import dataclass
from typing import Protocol

import psycopg
from openai import APIConnectionError, InternalServerError, OpenAI, RateLimitError

from leaselens.config import Settings
from leaselens.embed import QUERY_PREFIX, embed_texts
from leaselens.store import StoredChunk, search

DISCLAIMER = (
    "This is legal information, not legal advice. Your city may add more tenant "
    "protections than state law. For help with your situation, contact a legal aid "
    "office or your campus student legal services."
)

REFUSAL = (
    "I don't have a source that answers this, so I won't guess. "
    "Please contact a local tenant rights organization, a legal aid office, or your "
    "campus student legal services."
)

# The model returns this exact token when the sources do not answer the question.
INSUFFICIENT = "INSUFFICIENT_SOURCES"

SYSTEM_PROMPT = f"""You answer California tenant-rights questions for college students.

Rules:
- Answer ONLY from the numbered sources provided. Never use outside knowledge.
- Every factual claim must cite its source inline, like [Civ. Code §1950.5].
- Use plain English a first-year student can follow.
- If the sources do not answer the question, reply with exactly: {INSUFFICIENT}
- Residential rentals (a home the tenant lives in) are governed by Civil Code §1950.5. Section 1950.7 covers non-residential deposits; do not apply its deadlines or rules to a residential deposit question.
- State a deadline, dollar amount, or penalty only if it appears in the sources, and quote its section. If the sources say a penalty exists but not its amount, say that instead of guessing.
- Do not call a landlord's conduct illegal or presumed illegal. Say what the section requires, and what remedies the section gives a tenant.
- If a relevant section gives the tenant a remedy (such as damages for bad faith), include it, with its amount when the sources state one.
- Cite only as [Civ. Code §X]. Do not use source markers or special tokens.
- Do not give personal legal advice. Describe what the law says."""


class Embedder(Protocol):
    def __call__(self, texts: list[str]) -> list[list[float]]: ...


class Generator(Protocol):
    def __call__(self, system: str, user: str) -> str: ...


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[StoredChunk]
    refused: bool
    # True when the LLM was unavailable (e.g. free-tier rate limit) and the text is
    # the matched statute excerpts instead of a generated summary.
    fallback: bool = False
    disclaimer: str = DISCLAIMER


FALLBACK_SOURCES = 2  # quote at most this many sections when the LLM is unavailable
EXCERPT_CHARS = 700  # keep each quoted excerpt short; the full text is one click away


def _excerpt(text: str, limit: int = EXCERPT_CHARS) -> str:
    """Cut at a sentence end before the limit, or at a word boundary with an ellipsis."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    sentence_end = max(cut.rfind(". "), cut.rfind("; "))
    if sentence_end > limit // 2:
        return cut[: sentence_end + 1]
    return cut.rsplit(" ", 1)[0] + " …"


def source_excerpts(sources: list[StoredChunk]) -> str:
    """No-LLM answer: quote the top matched sections, trimmed, with citations."""
    # One excerpt per section: two chunks of the same section would repeat the same citation.
    seen: set[str] = set()
    unique: list[StoredChunk] = []
    for s in sources:
        if s.section not in seen:
            seen.add(s.section)
            unique.append(s)
    lines = [
        f"Civ. Code §{s.section} ({s.heading}), as of {s.as_of.isoformat()}:\n"
        f"{_excerpt(s.text)} [Civ. Code §{s.section}]"
        for s in unique[:FALLBACK_SOURCES]
    ]
    return (
        "The AI summary is unavailable right now, so here is the matching statute text:\n\n"
        + "\n\n".join(lines)
    )


def build_user_prompt(question: str, sources: list[StoredChunk]) -> str:
    blocks = [
        f"[Source {i}] Civ. Code §{s.section} ({s.heading}), as of {s.as_of.isoformat()}\n{s.text}"
        for i, s in enumerate(sources, start=1)
    ]
    return "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"


def answer_question(
    question: str,
    conn: psycopg.Connection,
    embed: Embedder,
    generate: Generator,
    settings: Settings,
) -> Answer:
    query_vector = embed([question])[0]
    hits = search(
        conn,
        query_vector,
        settings.top_k,
        query_text=question if settings.hybrid_search else None,
    )
    relevant = [h for h in hits if h.distance <= settings.max_distance]

    # Refuse without calling the LLM when nothing retrieved is close enough.
    if not relevant:
        return Answer(text=REFUSAL, sources=[], refused=True)

    try:
        text = generate(SYSTEM_PROMPT, build_user_prompt(question, relevant)).strip()
    except (RateLimitError, APIConnectionError, InternalServerError):
        # Free-tier quota exhausted or provider down: still answer, from the sources only.
        # Auth and bad-request errors are not caught, so a misconfigured key stays visible.
        return Answer(
            text=source_excerpts(relevant), sources=relevant, refused=False, fallback=True
        )

    if text == INSUFFICIENT or not text:
        return Answer(text=REFUSAL, sources=[], refused=True)

    return Answer(text=text, sources=relevant, refused=False)


def make_embedder(model_name: str) -> Embedder:
    # Questions get the query prefix; the stored statute chunks were embedded without it.
    def embed(texts: list[str]) -> list[list[float]]:
        return embed_texts(model_name, texts, prefix=QUERY_PREFIX)

    return embed


def make_generator(client: OpenAI, model: str, max_tokens: int = 1024) -> Generator:
    def generate(system: str, user: str) -> str:
        response = client.chat.completions.create(
            model=model,
            max_completion_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""

    return generate
