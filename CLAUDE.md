# CLAUDE.md: California Tenant Rights Assistant

Context for Claude Code (and any AI assistant) working on this repo. Read this before writing code.

## What this project is

A RAG web app for college students in California. A student asks a plain-English question about renting ("My landlord kept my whole deposit, is that legal?"), and the assistant answers in plain English, quotes and links the exact Civil Code section or official guide it relied on, and says "I don't know, here's who to ask" when its sources don't cover the question. It's backed by an evaluation harness that measures retrieval and citation accuracy on a 100-question benchmark and runs in CI.

## Why it exists

1. **Real problem:** students sign leases without knowing their rights on deposits, repairs, illegal fees, and eviction. The law is public but buried in statute text.
2. **Resume gaps it closes:** RAG, embeddings, vector databases, evaluation harnesses and experiments, OpenAI API, Next.js + TypeScript + Tailwind, AWS deployment, CI/CD, monitoring.

**Definition of done:** deployed with a public link, clean README, evals passing in CI, `EXPERIMENTS.md` documenting results, and 5-10+ real students using it with one tracked usage number (weekly questions answered). That number goes on the resume.

## Ground rules

- **No work or lab systems.** Never connect this to UC Merced's Pinnacles HPC cluster, lab data, or any LLT Lab resource. Public data only.
- **AI output is a draft.** Read every diff before accepting it. Don't accept whole files blind. Run it. Test it.
- **Check external APIs against real docs.** Don't trust generated calls to Groq, pgvector, AWS, or other APIs without checking current documentation.
- **Tests are not optional.** Write tests alongside features. If a test or eval fails, fix the code, not the test (unless the test is actually wrong, and say why).
- **Never commit secrets.** Keys go in `.env` (gitignored) locally and in GitHub Actions secrets / AWS secrets in deployment.
- **Small commits, clear messages.** One logical change per commit.
- **Log what matters, not personal data.** Questions are logged anonymously.

## Legal guardrails (required, not optional)

- **Every claim needs a citation.** If nothing retrieved supports an answer, the app does not answer. It says so and points to resources.
- **Answer only from retrieved sources.** Never from the model's training data. The law changes, and a confident outdated answer is the worst failure mode.
- **Visible "legal information, not legal advice" notice**, plus links to legal aid and campus student legal services where available.
- **Show each source's "as of" date.**
- **State law only in v1.** Cities like Berkeley, LA, and SF have extra local rules. The app must say "your city may add more protections."
- **No personal data stored.** No names, addresses, or lease details saved.
- When building the test set, **verify current law against the official sources**, especially anything recently changed (e.g., security deposit limits).

## Data sources (all public)

- California Civil Code landlord-tenant sections, starting with §1940-1954, including:
  - §1950.5 security deposits
  - §1941 repairs / habitability
  - §1946.2 just-cause eviction (Tenant Protection Act)
  - §1947.12 rent cap (Tenant Protection Act)
- The state's official California Tenants guide
- California Courts Self-Help Center pages on renting and eviction
- **Later phase:** local rent control ordinances

## Stack

- **Backend:** Python + FastAPI
- **Retrieval:** PostgreSQL + pgvector; local sentence-transformers `BAAI/bge-small-en-v1.5` (384-dim, free)
- **LLM:** Groq free tier, `openai/gpt-oss-120b` via Groq's OpenAI-compatible API; falls back to quoting matched statute text when rate limited
- **Frontend:** Next.js + TypeScript + Tailwind, chat UI with clickable citations
- **Deploy:** AWS App Runner for the FastAPI Docker container; Amplify or Vercel for the Next.js web app
- **CI:** GitHub Actions runs tests and evals on every push
- **Monitoring:** log every query, retrieved chunks, latency, and whether the answer was a refusal

## Decisions

- **Backend:** FastAPI (Python 3.12). Not Next.js + tRPC + Drizzle + SST.
- **Python tooling:** uv (lockfile), pytest, ruff.
- **Database:** Docker Postgres with pgvector for local dev; Amazon RDS with pgvector in production.
- **Embeddings:** local sentence-transformers `BAAI/bge-small-en-v1.5` by default, so the project costs nothing to run. Changed from OpenAI `text-embedding-3-small` to keep the free goal.
- **LLM:** Groq free tier (`openai/gpt-oss-120b`), chosen over Gemini's free tier because Google's unpaid tier may use prompts to improve its products. Changed from Claude Sonnet 5.5 to avoid paid API usage. Groq's data policy is only partly verified (see Groq Data Processing Addendum).
- **Chunking (default):** section-aware, one chunk per statute section. Fixed token windows are the experiment.
- **Retrieval (default):** pure vector search. Hybrid keyword + vector is the experiment.

## Evaluation harness

- **Test set:** 80-100 realistic tenant questions, each tagged with the correct statute section(s). Mix of self-written and real questions collected from students.
- **Out-of-scope questions** (local ordinances, non-tenant law, nonsense) where the correct behavior is refusing.
- **"Stale law" questions** where an outdated answer would be wrong.
- **Metrics:**
  - Retrieval recall@5: did the correct section come back?
  - Citation accuracy: does the quoted source actually support the answer?
  - Answer correctness: LLM judge + manual spot-check of a sample
  - Refusal correctness: out-of-scope questions get "I don't know"
- **Experiments:**
  - Chunking by statute section vs. fixed token windows
  - Two embedding models
  - Pure vector search vs. hybrid keyword + vector
- **CI gate:** the build fails if any metric drops below the current baseline.
- **`EXPERIMENTS.md`:** log every experiment: what changed, metrics before/after, decision made.

## Build phases

1. **Ingest + search (~3 days):** scrape and chunk the code sections, embed, store in pgvector, basic Q&A with citations from the terminal.
2. **Eval harness (~3 days):** build the test set and metrics script, run the baseline, run chunking/retrieval experiments.
3. **Web app (~4 days):** Next.js chat with clickable citations and the disclaimer.
4. **Deploy + monitor (~2 days):** AWS, CI running evals, logging.
5. **Real users (ongoing):** share in UC Merced housing groups, r/ucmerced, Theta Tau; track weekly questions answered.

## Resume bullet target

> Built and deployed a RAG assistant answering California tenant-rights questions with statute-level citations; evaluated on a 100-question benchmark, improving citation accuracy from X% to Y% via section-aware chunking and hybrid retrieval; used by N students.