"""HTTP API for the web app. One endpoint: POST /ask.

The answer logic lives in leaselens.qa. This module only validates input, runs the
pipeline, and shapes the JSON response.
"""

import threading
import time
from collections import OrderedDict, defaultdict, deque
from datetime import UTC, date, datetime
from functools import lru_cache

import psycopg
from fastapi import FastAPI, HTTPException, Request
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


CACHE_MAX_ENTRIES = 500
_answer_cache: OrderedDict[str, Answer] = OrderedDict()
_cache_lock = threading.Lock()


def _cache_key(question: str) -> str:
    return " ".join(question.lower().split())


def ask_service(question: str) -> Answer:
    """Run the full pipeline for one question, reusing a cached answer when there is one.

    Only real model answers and refusals are cached. Fallback answers (the LLM was
    unavailable) are not, so a later request can still get a summary.
    """
    key = _cache_key(question)
    with _cache_lock:
        if key in _answer_cache:
            _answer_cache.move_to_end(key)
            return _answer_cache[key]

    settings = get_settings()
    embed = make_embedder(settings.embedding_model)
    generate = make_generator(_groq_client(), settings.llm_model)
    with connect(settings.database_url) as conn:
        answer = answer_question(
            question,
            conn=conn,
            embed=embed,
            generate=generate,
            settings=settings,
        )

    if not answer.fallback:
        with _cache_lock:
            _answer_cache[key] = answer
            if len(_answer_cache) > CACHE_MAX_ENTRIES:
                _answer_cache.popitem(last=False)
    return answer


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


class QuestionLimiter:
    """In-memory limits: a sliding window per client IP, and a cap per UTC day.

    In-memory means the counts reset when the server restarts and do not share across
    several server processes. That is fine for one small instance; move to a shared store
    before scaling out.
    """

    def __init__(self) -> None:
        self._recent: dict[str, deque[float]] = defaultdict(deque)
        self._day: date | None = None
        self._day_count = 0
        self._lock = threading.Lock()

    def check(
        self,
        client: str,
        now: float,
        per_ip: int,
        window_seconds: int,
        daily_cap: int,
    ) -> str | None:
        """Return a reason string if the question is refused, otherwise record it and return None."""
        with self._lock:
            today = datetime.fromtimestamp(now, tz=UTC).date()
            if today != self._day:
                self._day, self._day_count = today, 0
            if self._day_count >= daily_cap:
                return "The daily question limit has been reached. Please try again tomorrow."
            hits = self._recent[client]
            while hits and now - hits[0] > window_seconds:
                hits.popleft()
            if len(hits) >= per_ip:
                return "You have asked many questions recently. Please wait a few minutes and try again."
            hits.append(now)
            self._day_count += 1
            return None


limiter = QuestionLimiter()


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, http_request: Request) -> AskResponse:
    settings = get_settings()
    client_ip = http_request.client.host if http_request.client else "unknown"
    reason = limiter.check(
        client_ip,
        time.time(),
        per_ip=settings.ask_limit_per_ip,
        window_seconds=settings.ask_limit_window_seconds,
        daily_cap=settings.daily_question_cap,
    )
    if reason:
        raise HTTPException(status_code=429, detail=reason)
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
