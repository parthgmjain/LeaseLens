"use client";

import Link from "next/link";
import { Fragment, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Source = {
  section: string;
  heading: string;
  source_url: string;
  as_of: string;
  text: string;
};

type Excerpt = {
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

// Matches citations the model writes like [Civ. Code §1950.5] or [Civ. Code §1950.5 (h)(1)].
const CITATION = /\[Civ\. Code §([0-9.]+)[^\]]*\]/g;

/** A California condor in flight: the one motif the whole page is built around. */
function Condor({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 400 200"
      aria-hidden="true"
      focusable="false"
      className={className}
    >
      <path
        fill="currentColor"
        d="M200 70 C214 70 226 76 230 80 C270 66 310 58 340 56 L386 52 L364 62 L378 66 L352 74 L362 80 L338 86 C300 96 262 110 224 118 L200 124 L176 118 C138 110 100 96 62 86 L38 80 L48 74 L22 66 L36 62 L14 52 L60 56 C90 58 130 66 170 80 C178 76 186 70 200 70 Z"
      />
      <circle cx="204" cy="62" r="9" fill="var(--condor-head)" />
      <path fill="var(--condor-head)" d="M212 58 L226 63 L212 67 Z" />
    </svg>
  );
}

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
              className="font-semibold text-oxblood underline underline-offset-2"
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
    <details open={open} className="bg-surface p-5">
      <summary className="cursor-pointer">
        <a
          href={excerpt.source_url}
          target="_blank"
          rel="noopener noreferrer"
          className="font-serif text-xl font-semibold text-oxblood underline underline-offset-4"
        >
          Civil Code §{excerpt.section}
        </a>
        <span className="block pt-1 text-sm text-muted">
          {excerpt.heading} · as of {excerpt.as_of}
        </span>
      </summary>
      <p className="mt-4 whitespace-pre-line leading-7 text-ink">{excerpt.text}</p>
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
    <div className="min-h-screen bg-ground text-ink">
      <header className="mx-auto flex w-full max-w-5xl items-center justify-between px-5 pt-6 sm:px-8">
        <Link href="/" className="flex items-center gap-2 font-serif text-2xl font-bold text-ink">
          <Condor className="h-6 w-12 text-condor" />
          CondorCode
        </Link>
      </header>

      <section className="relative isolate mx-auto w-full max-w-5xl overflow-hidden px-5 pb-40 pt-12 sm:px-8 sm:pb-56 sm:pt-20">
        <Condor className="pointer-events-none absolute -bottom-6 left-1/2 -z-10 w-[120vw] max-w-[760px] -translate-x-1/2 text-condor sm:bottom-auto sm:left-auto sm:right-[-8vw] sm:top-10 sm:w-[64vw] sm:translate-x-0" />
        <h1 className="max-w-[14ch] font-serif text-5xl font-bold leading-[1.02] tracking-tight text-ink sm:text-7xl">
          Know your rights as a renter.
        </h1>
        <p className="mt-6 max-w-[38ch] text-lg leading-8 text-ink sm:text-xl">
          Plain-English answers to California tenant-rights questions, quoting the Civil Code
          section each one comes from.
        </p>
      </section>

      <main className="mx-auto flex w-full max-w-3xl flex-col gap-8 px-5 pb-16 sm:px-8">
        <div role="note" className="bg-surface p-5 text-sm leading-6 text-ink">
          <strong className="font-semibold text-oxblood">This is not legal advice.</strong> The
          text shown is from California Civil Code, taken from the official leginfo.legislature.ca.gov
          download on November 10, 2026. Make any legal decision with a lawyer, not with this tool.
          Your city may add more tenant protections than state law. For help, contact a legal aid
          office or your campus student legal services. Do not include your name, address, or lease
          details.
        </div>

        <form onSubmit={ask} className="flex flex-col gap-3">
          <label htmlFor="question" className="font-serif text-2xl font-semibold text-ink">
            Ask a question about renting in California
          </label>
          <textarea
            id="question"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            maxLength={500}
            rows={3}
            placeholder="My landlord kept my whole deposit. Is that legal?"
            className="border-2 border-ink bg-[#fbf6ea] p-4 text-lg text-ink placeholder:text-muted focus:border-oxblood focus:outline-none"
          />
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted">{question.length}/500</span>
            <button
              type="submit"
              disabled={loading || question.trim().length < 3}
              className="bg-oxblood px-6 py-3 font-semibold text-oxblood-ink hover:bg-ink focus-visible:outline-brass disabled:opacity-50"
            >
              {loading ? "Checking the statutes…" : "Ask CondorCode"}
            </button>
          </div>
        </form>

        {error && (
          <p role="alert" className="bg-[#f3d9d4] p-4 text-[#5c1a14]">
            {error}
          </p>
        )}

        {result && (
          <section aria-live="polite" className="flex flex-col gap-5">
            {result.excerpt_only && (
              <p className="bg-surface p-4 text-sm text-ink">
                This tool shows the matching statute text without an AI summary. Read the sections
                below for the exact rules.
              </p>
            )}
            {result.fallback && (
              <p className="bg-surface p-4 text-sm text-ink">
                The AI summary is unavailable right now, so the matching statute text is shown
                instead.
              </p>
            )}
            {result.refused && (
              <p className="bg-surface p-4 text-sm text-ink">
                No source here answers this question, so the app will not guess.
              </p>
            )}
            {result.excerpt_only ? (
              <div className="flex flex-col gap-4">
                {result.excerpts.slice(0, 2).map((e) => (
                  <ExcerptCard key={e.section} excerpt={e} open />
                ))}
                {result.excerpts.length > 2 && (
                  <details className="bg-surface p-5">
                    <summary className="cursor-pointer font-semibold text-ink">
                      Show {result.excerpts.length - 2} more section
                      {result.excerpts.length - 2 === 1 ? "" : "s"}
                    </summary>
                    <div className="mt-4 flex flex-col gap-4">
                      {result.excerpts.slice(2).map((e) => (
                        <ExcerptCard key={e.section} excerpt={e} open={false} />
                      ))}
                    </div>
                  </details>
                )}
              </div>
            ) : (
              <div className="bg-surface p-6 text-lg">{renderAnswer(result.answer, result.sources)}</div>
            )}

            {!result.excerpt_only && result.sources.length > 0 && (
              <div>
                <h2 className="mb-3 font-serif text-2xl font-semibold">Sources</h2>
                <ul className="flex flex-col gap-3">
                  {result.sources.map((s) => (
                    <li key={`${s.section}-${s.source_url}`} className="bg-surface p-4 text-sm">
                      <a
                        href={s.source_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="font-semibold text-oxblood underline underline-offset-2"
                      >
                        Civil Code §{s.section}
                      </a>
                      <span className="text-muted"> · as of {s.as_of}</span>
                      <details className="mt-2 text-ink">
                        <summary className="cursor-pointer text-muted">Show statute text</summary>
                        <p className="mt-2 whitespace-pre-line">{s.text}</p>
                      </details>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            <p className="text-sm text-muted">{result.disclaimer}</p>
          </section>
        )}
      </main>

      <footer className="relative mt-auto overflow-hidden bg-walnut px-5 py-14 text-ground sm:px-8">
        <p className="mx-auto max-w-3xl font-serif text-5xl font-bold tracking-tight sm:text-7xl">
          CondorCode
        </p>
        <p className="mx-auto mt-4 max-w-3xl text-sm text-ground/80">
          California tenant-rights information, with every answer tied to its statute. Not legal
          advice.
        </p>
      </footer>
    </div>
  );
}
