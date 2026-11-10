from dataclasses import dataclass
from datetime import date

# Long sections are split at paragraph breaks so no single chunk gets too large to embed well.
MAX_CHUNK_CHARS = 4000


@dataclass(frozen=True)
class Section:
    number: str  # e.g. "1950.5"
    heading: str  # e.g. "Security deposits"
    body: str  # plain text, paragraphs separated by newlines
    source_url: str
    as_of: date


@dataclass(frozen=True)
class Chunk:
    section: str
    heading: str
    chunk_index: int
    text: str
    source_url: str
    as_of: date


def chunk_section(section: Section, max_chars: int = MAX_CHUNK_CHARS) -> list[Chunk]:
    """Section-aware chunking: one chunk per statute section, split only when a section is too long."""
    paragraphs = [p.strip() for p in section.body.split("\n") if p.strip()]
    pieces: list[str] = []
    current: list[str] = []
    current_len = 0
    for paragraph in paragraphs:
        if current and current_len + len(paragraph) + 1 > max_chars:
            pieces.append("\n".join(current))
            current, current_len = [], 0
        current.append(paragraph)
        current_len += len(paragraph) + 1
    if current:
        pieces.append("\n".join(current))

    return [
        Chunk(
            section=section.number,
            heading=section.heading,
            chunk_index=index,
            text=piece,
            source_url=section.source_url,
            as_of=section.as_of,
        )
        for index, piece in enumerate(pieces)
    ]
