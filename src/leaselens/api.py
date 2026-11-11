"""HTTP API for the web app. One endpoint: POST /ask.

The answer logic lives in leaselens.qa. This module only validates input, runs the
pipeline, and shapes the JSON response.
"""

from datetime import date
from functools import lru_cache

import psycopg
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field

from leaselens.config import get_settings
from leaselens.qa import Answer, answer_question, make_embedder, make_generator
from leaselens.store import connect

MAX_QUESTION_CHARS = 500

app = FastAPI(title="LeaseLens", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=MAX_QUESTION_CHARS)


class SourceOut(BaseModel):
    section: str
    heading: str
    source_url: str
    as_of: date
    text: str


class AskResponse(BaseModel):
    answer: str
    refused: bool
    # True when the LLM was unavailable and `answer` is quoted statute text, not a summary.
    fallback: bool
    sources: list[SourceOut]
    disclaimer: str


@lru_cache
def _groq_client() -> OpenAI:
    settings = get_settings()
    # No SDK retries and a short timeout: on a free-tier rate limit the user should get the
    # quoted-statute fallback right away, not a long spinner while the SDK backs off.
    return OpenAI(
        api_key=settings.groq_api_key,
        base_url=settings.groq_base_url,
        max_retries=0,
        timeout=30.0,
    )


def ask_service(question: str) -> Answer:
    """Run the full pipeline for one question. Tests replace this function."""
    settings = get_settings()
    embed = make_embedder(settings.embedding_model)
    generate = make_generator(_groq_client(), settings.llm_model)
    with connect(settings.database_url) as conn:
        return answer_question(
            question,
            conn=conn,
            embed=embed,
            generate=generate,
            settings=settings,
        )


def _unique_sources(answer: Answer) -> list[SourceOut]:
    seen: set[tuple[str, str, str, date]] = set()
    out: list[SourceOut] = []
    for s in answer.sources:
        key = (s.section, s.heading, s.source_url, s.as_of)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            SourceOut(
                section=s.section,
                heading=s.heading,
                source_url=s.source_url,
                as_of=s.as_of,
                text=s.text,
            )
        )
    return out


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        answer = ask_service(request.question.strip())
    except psycopg.OperationalError as exc:
        raise HTTPException(status_code=503, detail="The statute database is unavailable.") from exc
    return AskResponse(
        answer=answer.text,
        refused=answer.refused,
        fallback=answer.fallback,
        sources=_unique_sources(answer),
        disclaimer=answer.disclaimer,
    )
