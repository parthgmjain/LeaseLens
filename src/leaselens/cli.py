import argparse
from datetime import date
from pathlib import Path

from anthropic import Anthropic
from openai import OpenAI

from leaselens.chunking import chunk_section
from leaselens.config import get_settings
from leaselens.embed import embed_texts
from leaselens.qa import answer_question, make_embedder, make_generator
from leaselens.sources import load_civ_sections
from leaselens.store import ChunkRecord, connect, init_schema, replace_chunks

# Phase 1 scope: Civil Code landlord-tenant sections 1940-1954.
FIRST_SECTION = 1940
LAST_SECTION = 1954


def ingest(zip_path: Path, as_of: date) -> int:
    settings = get_settings()
    sections = load_civ_sections(zip_path, FIRST_SECTION, LAST_SECTION, as_of)
    chunks = [chunk for section in sections for chunk in chunk_section(section)]

    openai_client = OpenAI(api_key=settings.openai_api_key)
    vectors = embed_texts(
        openai_client, settings.embedding_model, [c.text for c in chunks]
    )

    records = [
        ChunkRecord(
            section=c.section,
            heading=c.heading,
            chunk_index=c.chunk_index,
            text=c.text,
            source_url=c.source_url,
            as_of=c.as_of,
            embedding=vector,
        )
        for c, vector in zip(chunks, vectors, strict=True)
    ]

    with connect(settings.database_url) as conn:
        init_schema(conn, settings.embedding_dim)
        replace_chunks(conn, records)

    print(f"Ingested {len(sections)} sections as {len(records)} chunks.")
    return len(records)


def ask(question: str) -> None:
    settings = get_settings()
    openai_client = OpenAI(api_key=settings.openai_api_key)
    anthropic_client = Anthropic(api_key=settings.anthropic_api_key)

    with connect(settings.database_url) as conn:
        answer = answer_question(
            question,
            conn=conn,
            embed=make_embedder(openai_client, settings.embedding_model),
            generate=make_generator(anthropic_client, settings.llm_model),
            settings=settings,
        )

    print(answer.text)
    if answer.sources:
        print("\nSources:")
        for source in {
            (s.section, s.heading, s.source_url, s.as_of): s for s in answer.sources
        }.values():
            print(
                f"- Civ. Code §{source.section} ({source.heading}), as of {source.as_of.isoformat()}: {source.source_url}"
            )
    print(f"\n{answer.disclaimer}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="leaselens")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_cmd = sub.add_parser(
        "ingest", help="Load Civil Code sections into the vector store"
    )
    ingest_cmd.add_argument(
        "--zip", type=Path, default=Path("data/raw/pubinfo_2025.zip")
    )
    ingest_cmd.add_argument("--as-of", type=date.fromisoformat, default=date.today())

    ask_cmd = sub.add_parser("ask", help="Ask a tenant-rights question")
    ask_cmd.add_argument("question")

    args = parser.parse_args()
    if args.command == "ingest":
        ingest(args.zip, args.as_of)
    elif args.command == "ask":
        ask(args.question)


if __name__ == "__main__":
    main()
