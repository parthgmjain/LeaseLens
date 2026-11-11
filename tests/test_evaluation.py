from leaselens.evaluation import CaseResult, EvalCase, score


def case(**overrides) -> EvalCase:
    base = {
        "id": "c",
        "question": "q",
        "expect_sections": ["1950.5"],
        "must_include": ["21 calendar days"],
        "must_not_include": ["$200"],
        "expect_refusal": False,
        "verified": True,
    }
    base.update(overrides)
    return EvalCase(**base)


def result(c: EvalCase, sections, answer, refused=False, fallback=False) -> CaseResult:
    return CaseResult(
        case=c,
        retrieved_sections=sections,
        answer_text=answer,
        refused=refused,
        fallback=fallback,
    )


def test_retrieval_hit_when_expected_section_is_retrieved():
    r = result(case(), ["1950.7", "1950.5"], "x")
    assert r.retrieval_hit is True


def test_retrieval_miss_when_expected_section_is_absent():
    r = result(case(), ["1950.7"], "x")
    assert r.retrieval_hit is False


def test_refusal_cases_have_no_retrieval_expectation():
    c = case(expect_sections=[], expect_refusal=True, must_include=[])
    r = result(c, ["1941"], "", refused=True)
    assert r.retrieval_hit is None
    assert r.refusal_correct is True


def test_fact_coverage_is_case_insensitive():
    r = result(case(), ["1950.5"], "Return within 21 Calendar Days.")
    assert r.facts_found == ["21 calendar days"]
    assert r.facts_missing == []


def test_forbidden_phrase_is_flagged():
    r = result(case(), ["1950.5"], "There is a $200 penalty.")
    assert r.forbidden_found == ["$200"]


def test_score_aggregates_across_cases():
    good = result(case(), ["1950.5"], "Within 21 calendar days.")
    bad = result(case(), ["1941"], "No idea.")
    report = score([good, bad])

    assert report.total == 2
    assert report.retrieval_recall == 0.5
    assert report.fact_coverage == 0.5
    assert report.forbidden_violations == 0
    assert report.fallbacks == 0


def test_bare_title_number_matches_its_subsections():
    r = result(case(expect_sections=["1941"]), ["1941.1"], "x")
    assert r.retrieval_hit is True


def test_dotted_section_does_not_match_a_different_section():
    r = result(case(expect_sections=["1950.5"]), ["1950.50"], "x")
    assert r.retrieval_hit is False


def test_fact_matching_ignores_typographic_spaces_and_dashes():
    # The model writes narrow no-break spaces and non-breaking hyphens.
    r = result(case(must_include=["21 calendar days", "12-month"]), ["1950.5"], "within 21 calendar days in a 12‑month span")
    assert r.facts_missing == []
