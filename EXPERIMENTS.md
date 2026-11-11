# Experiments

Log of every eval run and the decision it led to, per `CLAUDE.md`.

## Baseline 1: eval harness v0 (2026-11-10)

**Setup:** BAAI/bge-small-en-v1.5 embeddings (384-dim), section-aware chunks (193 chunks from 141 sections, §1940-1954), top-k 5, refusal threshold `max_distance` 0.55, generation with `openai/gpt-oss-120b` on Groq's free tier. Eval set: `evals/questions.jsonl` (8 questions). Run with `uv run python evals/run_evals.py`.

**Results:**

| Metric | Value |
|---|---|
| Retrieval recall@5 | 100% (5 of 5 cases with an expected section) |
| Refusal accuracy | 88% (7 of 8) |
| Fact coverage | 100% (2 of 2 required facts found) |
| Forbidden phrases | 0 |
| LLM fallbacks | 0 |

**Failures:**
- `repairs-heat-broken`: the app refused an in-scope question about a broken heater. Retrieval found §1941.1, so the failure is in answering (the model declined), not finding the section.

**Caveats (read before trusting these numbers):**
- **Small set.** Eight questions can't show much. Only two checks test facts, and both are deposit questions.
- **Weak checks.** The eviction, rent-cap, and repairs questions only check that the expected section is retrieved, not that the answer is right.
- **Unverified expectations.** Only some questions are marked `verified: true` in the set. The others need checking against the official statute text before their results count.
- **Noisy.** The model is run once per question, and its output changes between runs. The bad-faith question passed on this run, but an earlier single answer omitted the damages, so one run is not a reliable measure.
- **Rate-limit fallbacks.** A run can hit Groq's free-tier limit and fall back to quoted text. That case gets scored as a fallback, so check the fallback count before comparing runs.

**Known issues the set does not cover yet:**
- Every citation labels the section "Hiring of Real Property", not the section's own heading.
- The bad-faith damages sentence spans a chunk boundary in §1950.5. Retrieval still finds the chunk, but the answer can miss the sentence.
- Out-of-scope refusal depends on the distance threshold, which was tuned for a different embedding model. Refusals for these questions currently look right, but there are too few to rely on.

**Decision:** Keep this as the baseline. Next, verify the expected facts for the unverified questions against the official text, add questions that test the known issues, and rerun. Then decide on chunk overlap, which is an experiment in its own right.

## Baseline 1, revised: verified expectations (2026-11-10)

Checked every expected fact against the stored official statute text (the leginfo bulk data loaded at ingest):

- `deposit-over-limit`: §1950.5(c)(1) caps security at one month's rent. Verified.
- `eviction-no-reason`: §1946.2(a) requires just cause after 12 months of lawful occupancy. Verified.
- `rent-increase-cap`: §1947.12(a)(1) caps increases at 5% plus the cost-of-living change, or 10%, whichever is lower. Verified.
- `repairs-heat-broken`: §1941.1(a)(4) requires heating facilities in good working order. Verified.

All eight questions are now marked verified.

**Scoring fixes made while verifying** (these change what counts as a pass):
- Matching ignored typographic spaces and dashes, which the model uses (for example, narrow no-break spaces). Without this fix, a correct "21 calendar days" failed the check.
- "10 %" and "10%" count as the same fact. Facts can list alternatives separated by `|`.

**Results after verification:**

| Metric | Value |
|---|---|
| Retrieval recall@5 | 100% |
| Refusal accuracy | 88% (7 of 8) |
| Fact coverage | 83% (5 of 6 required facts) |
| Forbidden phrases | 0 |
| LLM fallbacks | 0 |

**Failures:**
- `repairs-heat-broken`: refused again, even though §1941.1 is retrieved. This is a real failure in answering, and the refusal logic needs a look.
- `out-of-scope-local` passes, but the earlier run that hit a rate limit did not, so keep checking fallbacks.

**Caveats:**
- Still eight questions, and only one of them checks a fact against a sentence that spans a chunk boundary.
- Single runs of the model vary. Run it several times before drawing conclusions from a small change.

**Root cause of the heater failure:** Retrieval, not the model. The chunk holding §1941.1(a)(4), "heating facilities", ranks 11th for the question. The top five contain no heating rule, so the model correctly answered `INSUFFICIENT_SOURCES` for the sources it received.

## Experiment 2: split long statute sections at numbered subdivisions (negative result)

**Hypothesis:** Splitting oversized paragraphs at "(1)", "(2)", ... markers and capping chunks at 1,200 characters would let a single subdivision such as §1941.1(a)(4) rank in the top five.

**Change:** `MAX_CHUNK_CHARS` 4,000 → 1,200, plus splitting oversized paragraphs at subdivision markers. Re-ingested to 443 chunks.

**Result:**

| Metric | Baseline | Experiment 2 |
|---|---|---|
| Retrieval recall@5 | 100% | 100% |
| Refusal accuracy | 88% | 75% |
| Fact coverage | 83% | 33% |

- The heater question now retrieves §1941.1 at rank 1, but it is still refused.
- Other answers regressed. Smaller chunks lost the context that answers need, and the top five filled up with chunks from one section (e.g., every hit was §1947.12 for the rent question).

**Decision:** Reverted to the committed 4,000-character chunking and re-ingested (193 chunks). Baseline numbers are restored. The idea is not dead, but splitting alone is too blunt. A better version would keep each subdivision's section heading and surrounding context in its chunk, or cap results per section so one section can't fill the top five. Both are separate experiments.

**Open:** Hybrid keyword and vector search is still untested, and it's the other experiment `CLAUDE.md` lists. It would address the heater case directly because "heating" would match by keyword.

## Experiment 3: hybrid search (vector + Postgres full-text, reciprocal rank fusion)

**Change:** `search()` takes the question text and fuses the vector ranking with a Postgres full-text ranking (`websearch_to_tsquery`, English stemming, GIN index on a generated `fts` column). Distance for the refusal threshold is still the vector distance. Toggle with `HYBRID_SEARCH` (default true).

**Status: not yet measured.** The one run that completed under the harness was rate-limited and is invalid (see below).

**Partial observations from an earlier run (before pacing and skip logic, so treat with care):**
- The heater question was answered rather than refused, citing §1941 and §1941.2, but did not state the heating rule.
- Full-text matching on "heater" does not match "heating": Porter stemming maps "heating" to "heat" but "heater" stays "heater". The heater win came from other words in the question, so keyword matching alone won't fix that case.
- An out-of-scope question (Berkeley rent control) was answered from §1947.x rather than refused, when the run was not rate-limited. The distance threshold admitted those hits.

**Invalid run, 2026-11-10:** Seven of eight questions hit Groq's free-tier rate limit and fell back to raw statute text, even with a 65-second pause between questions. The most likely cause is Groq's daily token cap, which was exhausted by earlier runs today (not confirmed in the Groq console). The refusal figure (62%) from that run reflects fallback text, not model behavior, so it is not a result.

**Harness fix (same day):** `evals/run_evals.py` now retries a rate-limited question once after a wait. If it is still rate limited, the question is listed as skipped and excluded from every metric, and the report warns that metrics cover only the scored cases. `results.json` records the skipped list.

**Next:** Run the eight questions when the quota resets, compare hybrid against vector-only on the same run, and keep hybrid only if it helps the heater case without hurting the others.

## Note: citation headings are not a parsing bug

Earlier notes flagged every citation as labelled "Hiring of Real Property". That is the official article title for the section's position in the code (TOC path `8.4.78.3`). The bulk download has no section-level titles such as "Security deposits", so the label is accurate to the source but not descriptive. Any friendlier label would be a hand-curated list, not something parsed from the data.
