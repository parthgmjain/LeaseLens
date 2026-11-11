"use client";

import { Fragment, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Source = {
  section: string;
  heading: string;
  source_url: string;
  as_of: string;
  text: string;
};

type AskResponse = {
  answer: string;
  refused: boolean;
  fallback: boolean;
  excerpt_only: boolean;
  sources: Source[];
  excerpts: Excerpt[];
  disclaimer: string;
};

type Excerpt = {
  section: string;
  heading: string;
  source_url: string;
  as_of: string;
  text: string;
};

// Matches citations the model writes like [Civ. Code §1950.5] or [Civ. Code §1950.5 (h)(1)].
const CITATION = /\[Civ\. Code §([0-9.]+)[^\]]*\]/g;

/** Turn model text into paragraphs, with citations as links to the matching source. */
function renderAnswer(text: string, sources: Source[]) {
  const linkFor = new Map(sources.map((s) => [s.section, s.source_url]));
  const cleaned = text.replace(/\*\*/g, "");
  return cleaned.split(/\n\s*\n/).map((paragraph, p) => (
    <p key={p} className="mb-3 leading-7 last:mb-0">
      {paragraph.split(CITATION).map((part, i) => {
        // split() with a capture group alternates text and section numbers.
        if (i % 2 === 1) {
          const url = linkFor.get(part);
          return url ? (
            <a
              key={i}
              href={url}
              target="_blank"
              rel="noopener noreferrer"
              className="font-medium text-sky-800 underline underline-offset-2"
            >
              Civ. Code §{part}
            </a>
          ) : (
            <Fragment key={i}>Civ. Code §{part}</Fragment>
          );
        }
        return <Fragment key={i}>{part}</Fragment>;
      })}
    </p>
  ));
}

function ExcerptCard({ excerpt, open }: { excerpt: Excerpt; open: boolean }) {
  return (
    <details open={open} className="rounded-lg bg-white p-5 shadow-sm">
      <summary className="cursor-pointer">
        <a
          href={excerpt.source_url}
          target="_blank"
          rel="noopener noreferrer"
          className="font-medium text-sky-800 underline underline-offset-2"
        >
          Civil Code §{excerpt.section}
        </a>
        <span className="text-zinc-500"> · {excerpt.heading} · as of {excerpt.as_of}</span>
      </summary>
      <p className="mt-3 whitespace-pre-line leading-7 text-zinc-800">{excerpt.text}</p>
    </details>
  );
}

export default function Home() {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AskResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function ask(e: React.FormEvent) {
    e.preventDefault();
    const trimmed = question.trim();
    if (trimmed.length < 3) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmed }),
      });
      if (!res.ok) {
        // 429 carries a readable reason from the server (per-visitor or daily limit).
        const body = await res.json().catch(() => null);
        if (res.status === 429 && body?.detail) {
          throw new Error(body.detail);
        }
        throw new Error(
          res.status === 503
            ? "The statute database is unavailable right now. Try again later."
            : "Something went wrong. Please try again."
        );
      }
      setResult((await res.json()) as AskResponse);
    } catch (err) {
      setError(
        err instanceof TypeError
          ? "Could not reach the server. Check that the API is running."
          : err instanceof Error
            ? err.message
            : "Something went wrong."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-zinc-50 text-zinc-900">
      <main className="mx-auto flex w-full max-w-2xl flex-col gap-6 px-4 py-10">
        <header>
          <h1 className="text-3xl font-semibold tracking-tight">LeaseLens</h1>
          <p className="mt-2 text-zinc-600">
            Plain-English answers to California tenant-rights questions, with each answer tied to
            the Civil Code section it came from.
          </p>
        </header>

        <div
          role="note"
          className="rounded-lg border border-amber-300 bg-amber-50 p-4 text-sm text-amber-900"
        >
          <strong>This is not legal advice.</strong> The text shown is from California Civil Code, taken from the official leginfo.legislature.ca.gov download on November 10, 2026. Make any legal decision with a lawyer, not with this tool. Your city may add more tenant protections than state law. For help, contact a legal aid office or your campus student legal services. Do not include your name, address, or lease details.
        </div>

        <form onSubmit={ask} className="flex flex-col gap-3">
          <label htmlFor="question" className="font-medium">
            Ask a question about renting in California
          </label>
          <textarea
            id="question"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            maxLength={500}
            rows={3}
            placeholder="My landlord kept my whole deposit. Is that legal?"
            className="rounded-lg border border-zinc-300 bg-white p-3 focus:border-sky-600 focus:outline-none"
          />
          <div className="flex items-center justify-between">
            <span className="text-xs text-zinc-500">{question.length}/500</span>
            <button
              type="submit"
              disabled={loading || question.trim().length < 3}
              className="rounded-lg bg-sky-800 px-4 py-2 font-medium text-white disabled:opacity-50"
            >
              {loading ? "Checking the statutes…" : "Ask"}
            </button>
          </div>
        </form>

        {error && (
          <p role="alert" className="rounded-lg bg-red-50 p-4 text-red-800">
            {error}
          </p>
        )}

        {result && (
          <section aria-live="polite" className="flex flex-col gap-4">
            {result.excerpt_only && (
              <p className="rounded-lg bg-sky-50 p-3 text-sm text-sky-900">
                This tool shows the matching statute text without an AI summary. Read the sections
                below for the exact rules.
              </p>
            )}
            {result.fallback && (
              <p className="rounded-lg bg-zinc-100 p-3 text-sm text-zinc-700">
                The AI summary is unavailable right now, so the matching statute text is shown
                instead.
              </p>
            )}
            {result.refused && (
              <p className="rounded-lg bg-zinc-100 p-3 text-sm text-zinc-700">
                No source here answers this question, so the app will not guess.
              </p>
            )}
            {result.excerpt_only ? (
              <div className="flex flex-col gap-3">
                {result.excerpts.slice(0, 2).map((e) => (
                  <ExcerptCard key={e.section} excerpt={e} open />
                ))}
                {result.excerpts.length > 2 && (
                  <details className="rounded-lg bg-white p-4 shadow-sm">
                    <summary className="cursor-pointer font-medium text-zinc-700">
                      Show {result.excerpts.length - 2} more section
                      {result.excerpts.length - 2 === 1 ? "" : "s"}
                    </summary>
                    <div className="mt-3 flex flex-col gap-3">
                      {result.excerpts.slice(2).map((e) => (
                        <ExcerptCard key={e.section} excerpt={e} open={false} />
                      ))}
                    </div>
                  </details>
                )}
              </div>
            ) : (
              <div className="rounded-lg bg-white p-5 shadow-sm">
                {renderAnswer(result.answer, result.sources)}
              </div>
            )}

            {!result.excerpt_only && result.sources.length > 0 && (
              <div>
                <h2 className="mb-2 font-semibold">Sources</h2>
                <ul className="flex flex-col gap-3">
                  {result.sources.map((s) => (
                    <li key={`${s.section}-${s.source_url}`} className="rounded-lg border border-zinc-200 bg-white p-4 text-sm">
                      <a
                        href={s.source_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="font-medium text-sky-800 underline underline-offset-2"
                      >
                        Civil Code §{s.section}
                      </a>
                      <span className="text-zinc-500"> · as of {s.as_of}</span>
                      <details className="mt-2 text-zinc-700">
                        <summary className="cursor-pointer text-zinc-600">Show statute text</summary>
                        <p className="mt-2 whitespace-pre-line">{s.text}</p>
                      </details>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            <p className="text-xs text-zinc-500">{result.disclaimer}</p>
          </section>
        )}
      </main>
    </div>
  );
}
