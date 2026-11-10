from dataclasses import dataclass
from typing import Protocol

import psycopg
from anthropic import Anthropic
from openai import OpenAI

from leaselens.config import Settings
from leaselens.embed import embed_texts
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
    disclaimer: str = DISCLAIMER


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
    hits = search(conn, query_vector, settings.top_k)
    relevant = [h for h in hits if h.distance <= settings.max_distance]

    # Refuse without calling the LLM when nothing retrieved is close enough.
    if not relevant:
        return Answer(text=REFUSAL, sources=[], refused=True)

    text = generate(SYSTEM_PROMPT, build_user_prompt(question, relevant)).strip()
    if text == INSUFFICIENT or not text:
        return Answer(text=REFUSAL, sources=[], refused=True)

    return Answer(text=text, sources=relevant, refused=False)


def make_embedder(openai_client: OpenAI, model: str) -> Embedder:
    def embed(texts: list[str]) -> list[list[float]]:
        return embed_texts(openai_client, model, texts)

    return embed


def make_generator(
    anthropic_client: Anthropic, model: str, max_tokens: int = 1024
) -> Generator:
    def generate(system: str, user: str) -> str:
        response = anthropic_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in response.content if block.type == "text")

    return generate
