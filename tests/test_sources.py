from datetime import date
from pathlib import Path

import pytest

from leaselens.sources import load_civ_sections, parse_caml_body, section_key

ZIP_PATH = Path("data/raw/pubinfo_2025.zip")
AS_OF = date(2026, 10, 27)


def test_section_key_orders_numerically():
    assert section_key("1950.5") > section_key("1950.2")
    assert section_key("1954.071") < section_key("1955")
    assert section_key("3306a") == (3306,)


def test_parse_caml_body_keeps_one_paragraph_per_p():
    xml = (
        b'<caml:Content xmlns:caml="http://lc.ca.gov/legalservices/schemas/caml.1#">'
        b'<p>(a)<span class="EnSpace"/>First   paragraph.</p>'
        b"<p>(b)\n  Second paragraph.</p></caml:Content>"
    )

    assert parse_caml_body(xml) == "(a) First paragraph.\n(b) Second paragraph."


@pytest.mark.skipif(not ZIP_PATH.exists(), reason="leginfo bulk file not downloaded")
def test_real_civil_code_1940_1954_contains_the_core_tenant_sections():
    sections = load_civ_sections(ZIP_PATH, 1940, 1954, AS_OF)
    by_number = {s.number: s for s in sections}

    assert len(sections) > 100
    assert all(section_key(s.number)[0] in range(1940, 1955) for s in sections)
    # Bare §1941 has no row in the 2025 bulk file. It is flagged for manual verification.
    for number in ("1941.1", "1946.2", "1947.12", "1950.5"):
        assert number in by_number, number
        assert by_number[number].body
    assert "security" in by_number["1950.5"].body.lower()
    assert by_number["1950.5"].heading
    assert by_number["1950.5"].source_url.endswith("sectionNum=1950.5")
