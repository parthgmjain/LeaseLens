"""Run the eval set against the live corpus and print a report.

Usage: uv run python evals/run_evals.py
Needs the ingested corpus and GROQ_API_KEY in .env.
"""

import json
import time
from pathlib import Path

from openai import OpenAI

from leaselens.config import get_settings
from leaselens.embed import QUERY_PREFIX, embed_texts
from leaselens.evaluation import CaseResult, load_cases, score
from leaselens.qa import answer_question, make_embedder, make_generator
from leaselens.store import connect, search

ROOT = Path(__file__).resolve().parent
QUESTIONS = ROOT / "questions.jsonl"
RESULTS = ROOT / "results.json"
PAUSE_SECONDS = 65  # Groq free tier allows roughly one large request per minute (tokens per minute)
RETRY_WAIT_SECONDS = 60  # one retry after a rate-limit fallback, so scores reflect the model


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def main() -> None:
    settings = get_settings()
    cases = load_cases(QUESTIONS)
    groq = OpenAI(api_key=settings.groq_api_key, base_url=settings.groq_base_url)
    embed = make_embedder(settings.embedding_model)
    generate = make_generator(groq, settings.llm_model)

    results: list[CaseResult] = []
    skipped: list[str] = []  # rate-limited even after a retry: not scored, so they can't skew metrics
    with connect(settings.database_url) as conn:
        for case in cases:
            vector = embed_texts(settings.embedding_model, [case.question], prefix=QUERY_PREFIX)[0]
            hits = search(
                conn,
                vector,
                settings.top_k,
                query_text=case.question if settings.hybrid_search else None,
            )
            answer = answer_question(
                case.question,
                conn=conn,
                embed=embed,
                generate=generate,
                settings=settings,
            )
            if answer.fallback:
                time.sleep(RETRY_WAIT_SECONDS)
                answer = answer_question(
                    case.question,
                    conn=conn,
                    embed=embed,
                    generate=generate,
                    settings=settings,
                )
            if answer.fallback:
                skipped.append(case.id)
                print(f"SKIPPED {case.id}: rate limited", flush=True)
            else:
                results.append(
                    CaseResult(
                        case=case,
                        retrieved_sections=[h.section for h in hits],
                        answer_text=answer.text,
                        refused=answer.refused,
                        fallback=False,
                    )
                )
                print(f"done {case.id}", flush=True)
            if settings.answer_with_llm:  # only model calls need pacing for the free-tier quota
                time.sleep(PAUSE_SECONDS)

    report = score(results)
    print()
    print(f"cases scored:       {report.total} of {len(cases)}")
    if skipped:
        print(f"skipped (rate limited, not scored): {', '.join(skipped)}")
        print("WARNING: metrics below cover only the scored cases.")
    print(f"retrieval recall@{settings.top_k}: {pct(report.retrieval_recall)}")
    print(f"refusal accuracy:   {pct(report.refusal_accuracy)}")
    print(f"fact coverage:      {pct(report.fact_coverage)}")
    print(f"forbidden phrases:  {report.forbidden_violations}")
    print()
    for r in results:
        status = "ok" if (r.retrieval_hit is not False) and r.refusal_correct and not r.facts_missing else "FAIL"
        print(
            f"[{status}] {r.case.id}: retrieved {r.retrieved_sections[:5]}, "
            f"refused={r.refused}, missing facts={r.facts_missing or '-'}"
        )

    RESULTS.write_text(
        json.dumps(
            {
                "skipped": skipped,
                "cases": [
                    {
                        "id": r.case.id,
                        "retrieved_sections": r.retrieved_sections,
                        "refused": r.refused,
                        "retrieval_hit": r.retrieval_hit,
                        "refusal_correct": r.refusal_correct,
                        "facts_missing": r.facts_missing,
                        "forbidden_found": r.forbidden_found,
                        "answer": r.answer_text,
                    }
                    for r in results
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {RESULTS}")


if __name__ == "__main__":
    main()
