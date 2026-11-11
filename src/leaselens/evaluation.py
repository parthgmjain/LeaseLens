"""Scoring for the eval set in evals/questions.jsonl.

Three metrics, matching CLAUDE.md:
- retrieval recall@k: did an expected section come back in the sources?
- refusal accuracy: do out-of-scope questions get refused, and in-scope ones answered?
- fact coverage: does the answer contain the facts the statute requires?
"""

import json
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    expect_sections: list[str]
    must_include: list[str]
    must_not_include: list[str]
    expect_refusal: bool
    verified: bool
    notes: str = ""


@dataclass(frozen=True)
class CaseResult:
    case: EvalCase
    retrieved_sections: list[str]
    answer_text: str
    refused: bool
    fallback: bool

    @property
    def retrieval_hit(self) -> bool | None:
        if not self.case.expect_sections:
            return None
        return any(
            _section_matches(expected, got)
            for expected in self.case.expect_sections
            for got in self.retrieved_sections
        )

    @property
    def refusal_correct(self) -> bool:
        return self.refused == self.case.expect_refusal

    @property
    def facts_found(self) -> list[str]:
        text = _normalize(self.answer_text)
        return [f for f in self.case.must_include if _fact_found(f, text)]

    @property
    def facts_missing(self) -> list[str]:
        found = set(self.facts_found)
        return [f for f in self.case.must_include if f not in found]

    @property
    def forbidden_found(self) -> list[str]:
        text = _normalize(self.answer_text)
        return [f for f in self.case.must_not_include if _normalize(f) in text]


def _normalize(text: str) -> str:
    """Lowercase and fold typographic spaces and dashes, so '21\\u00a0calendar' still matches."""
    folded = unicodedata.normalize("NFKC", text)
    folded = folded.replace("‐", "-").replace("‑", "-").replace("–", "-")
    folded = " ".join(folded.lower().split())
    # "10 %" and "10%" are the same fact.
    return folded.replace(" %", "%")


def _fact_found(fact: str, text: str) -> bool:
    """A fact is found if any of its '|'-separated alternatives appears in the text."""
    return any(_normalize(option) in text for option in fact.split("|"))


def _section_matches(expected: str, got: str) -> bool:
    """'1950.5' matches only itself. A bare '1941' matches the whole 1941 family (1941, 1941.1, ...)."""
    if expected == got:
        return True
    return "." not in expected and got.split(".")[0] == expected


def load_cases(path: Path) -> list[EvalCase]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(EvalCase(**json.loads(line)))
    return cases


def _rate(hits: Iterable[bool]) -> float | None:
    values = list(hits)
    if not values:
        return None
    return sum(values) / len(values)


@dataclass(frozen=True)
class Report:
    retrieval_recall: float | None
    refusal_accuracy: float | None
    fact_coverage: float | None
    forbidden_violations: int
    fallbacks: int
    total: int
    results: list[CaseResult] = field(default_factory=list)


def score(results: list[CaseResult]) -> Report:
    retrieval = [r.retrieval_hit for r in results if r.retrieval_hit is not None]
    fact_totals = [len(r.case.must_include) for r in results]
    found_total = sum(len(r.facts_found) for r in results)
    total_facts = sum(fact_totals)
    return Report(
        retrieval_recall=_rate(retrieval),
        refusal_accuracy=_rate(r.refusal_correct for r in results),
        fact_coverage=(found_total / total_facts) if total_facts else None,
        forbidden_violations=sum(len(r.forbidden_found) for r in results),
        fallbacks=sum(1 for r in results if r.fallback),
        total=len(results),
        results=results,
    )
