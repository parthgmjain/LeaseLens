"""Read California Civil Code sections from the official leginfo bulk download (pubinfo_YYYY.zip).

The zip holds pipe-free, backtick-quoted tables. NULL marks an empty field. Section text
lives in separate .lob files, each a CAML XML document with one <p> per paragraph.
"""

import io
import re
import zipfile
from collections.abc import Iterator
from datetime import date
from pathlib import Path

from lxml import etree

from leaselens.chunking import Section

CODE = "CIV"
SECTION_TABLE = "LAW_SECTION_TBL.dat"
TOC_TABLE = "LAW_TOC_TBL.dat"
TOC_SECTIONS_TABLE = "LAW_TOC_SECTIONS_TBL.dat"

# Columns in LAW_SECTION_TBL.dat (0-based).
COL_LAW_CODE = 1
COL_SECTION_NUM = 2
COL_HISTORY = 13
COL_LOB_FILE = 14

# Columns in LAW_TOC_TBL.dat (0-based).
TOC_COL_LAW_CODE = 0
TOC_COL_TITLE = 6
TOC_COL_PATH = 13

# Columns in LAW_TOC_SECTIONS_TBL.dat (0-based).
TOC_SECTIONS_COL_LAW_CODE = 1
TOC_SECTIONS_COL_PATH = 2
TOC_SECTIONS_COL_NUMBER = 3

_HEADING_PREFIX = re.compile(
    r"^(?:CHAPTER|ARTICLE|PART|TITLE)\s+[\dIVXLC.]+\.?\s*", re.IGNORECASE
)


def section_key(number: str) -> tuple[int, ...]:
    """'1950.5' -> (1950, 5). Numeric tuples order sections correctly, unlike strings or floats.

    Letter suffixes are ignored, so '3306a' -> (3306,).
    """
    keys = []
    for part in number.strip().rstrip(".").split("."):
        digits = re.match(r"\d+", part)
        if digits:
            keys.append(int(digits.group()))
    return tuple(keys)


def in_range(number: str, first: int, last: int) -> bool:
    return (first,) <= section_key(number) < (last + 1,)


def _rows(zf: zipfile.ZipFile, member: str) -> Iterator[list[str | None]]:
    with zf.open(member) as raw:
        for line in io.TextIOWrapper(raw, encoding="utf-8", errors="replace"):
            fields = line.rstrip("\n").split("\t")
            yield [None if f == "NULL" else f.strip("`") for f in fields]


def _clean_number(raw: str) -> str:
    return raw.strip().rstrip(".")


def load_toc_titles(zf: zipfile.ZipFile) -> dict[str, str]:
    """Map each Civil Code TOC path (e.g. '8.4.78.3') to its official chapter or article title."""
    titles = {}
    for row in _rows(zf, TOC_TABLE):
        if (
            len(row) <= TOC_COL_PATH
            or row[TOC_COL_LAW_CODE] != CODE
            or not row[TOC_COL_TITLE]
        ):
            continue
        title = row[TOC_COL_TITLE]
        bracket = title.find("[")
        if bracket != -1:
            title = title[:bracket]
        title = _HEADING_PREFIX.sub("", title).strip().rstrip(".")
        titles[row[TOC_COL_PATH]] = title
    return titles


def load_section_paths(zf: zipfile.ZipFile) -> dict[str, str]:
    """Map each Civil Code section number to its TOC path."""
    paths = {}
    for row in _rows(zf, TOC_SECTIONS_TABLE):
        if len(row) <= TOC_SECTIONS_COL_PATH or row[TOC_SECTIONS_COL_LAW_CODE] != CODE:
            continue
        paths.setdefault(
            _clean_number(row[TOC_SECTIONS_COL_NUMBER] or ""),
            row[TOC_SECTIONS_COL_PATH],
        )
    return paths


def parse_caml_body(xml_bytes: bytes) -> str:
    root = etree.fromstring(xml_bytes)
    paragraphs = []
    for p in root.iter("{*}p"):
        text = " ".join("".join(p.itertext()).split())
        # The source puts no space after a subdivision marker, e.g. "(a)Except as...".
        text = re.sub(r"^(\([0-9a-zA-Z]+\))(?=\S)", r"\1 ", text)
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs)


def load_civ_sections(
    zip_path: Path, first: int, last: int, as_of: date
) -> list[Section]:
    """Civil Code sections whose integer part falls between first and last, inclusive."""
    sections: list[Section] = []
    with zipfile.ZipFile(zip_path) as zf:
        titles = load_toc_titles(zf)
        paths = load_section_paths(zf)
        for row in _rows(zf, SECTION_TABLE):
            if (
                len(row) <= COL_LOB_FILE
                or row[COL_LAW_CODE] != CODE
                or not row[COL_SECTION_NUM]
            ):
                continue
            number = _clean_number(row[COL_SECTION_NUM])
            if not in_range(number, first, last):
                continue
            lob = row[COL_LOB_FILE]
            if not lob:
                continue
            body = parse_caml_body(zf.read(lob))
            if not body:
                continue
            sections.append(
                Section(
                    number=number,
                    heading=titles.get(paths.get(number, ""), f"Civil Code §{number}"),
                    body=body,
                    source_url=(
                        "https://leginfo.legislature.ca.gov/faces/codes_displaySection.xhtml"
                        f"?lawCode={CODE}&sectionNum={number}"
                    ),
                    as_of=as_of,
                )
            )
    sections.sort(key=lambda s: section_key(s.number))
    return sections
