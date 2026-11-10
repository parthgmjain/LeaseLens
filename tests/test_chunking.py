from datetime import date

from leaselens.chunking import Section, chunk_section

AS_OF = date(2025, 1, 1)
URL = "https://example.test/1950.5"


def make_section(body: str) -> Section:
    return Section(
        number="1950.5",
        heading="Security deposits",
        body=body,
        source_url=URL,
        as_of=AS_OF,
    )


def test_short_section_is_one_chunk():
    chunks = chunk_section(make_section("(a) First paragraph.\n(b) Second paragraph."))

    assert len(chunks) == 1
    assert chunks[0].section == "1950.5"
    assert chunks[0].chunk_index == 0
    assert chunks[0].text == "(a) First paragraph.\n(b) Second paragraph."
    assert chunks[0].source_url == URL
    assert chunks[0].as_of == AS_OF


def test_blank_lines_are_dropped():
    chunks = chunk_section(make_section("(a) One.\n\n\n(b) Two.\n"))

    assert chunks[0].text == "(a) One.\n(b) Two."


def test_long_section_splits_at_paragraph_breaks_and_keeps_every_paragraph():
    paragraphs = [f"({chr(97 + i)}) " + "x" * 300 for i in range(10)]
    chunks = chunk_section(make_section("\n".join(paragraphs)), max_chars=1000)

    assert len(chunks) > 1
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(len(c.text) <= 1000 for c in chunks)
    rejoined = "\n".join(c.text for c in chunks)
    assert rejoined.split("\n") == paragraphs


def test_every_chunk_keeps_the_section_identity():
    paragraphs = ["y" * 500 for _ in range(5)]
    chunks = chunk_section(make_section("\n".join(paragraphs)), max_chars=600)

    assert {c.section for c in chunks} == {"1950.5"}
    assert {c.heading for c in chunks} == {"Security deposits"}
