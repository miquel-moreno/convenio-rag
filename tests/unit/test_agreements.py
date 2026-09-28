from collections import Counter
from datetime import date
from pathlib import Path

import pytest

from convenio_rag.services.agreements import (
    MAX_CHUNK_CHARS,
    ParsedAgreement,
    clean,
    parse_boe_xml,
    parse_heading,
    split_text,
)

DATA = Path(__file__).resolve().parents[2] / "data" / "boe"


@pytest.fixture(scope="module")
def consultancy() -> ParsedAgreement:
    return parse_boe_xml((DATA / "BOE-A-2025-7766.xml").read_bytes())


@pytest.fixture(scope="module")
def metal() -> ParsedAgreement:
    return parse_boe_xml((DATA / "BOE-A-2022-479.xml").read_bytes())


# --- small pieces --------------------------------------------------------------


def test_clean_collapses_non_breaking_spaces() -> None:
    assert clean("Artículo\xa021.\n  Vacaciones ") == "Artículo 21. Vacaciones"


@pytest.mark.parametrize(
    ("heading", "expected"),
    [
        ("Artículo\xa021. Vacaciones.", ("article", "Artículo 21", "Vacaciones")),
        ("Artículo 40B. Teletrabajo.", ("article", "Artículo 40B", "Teletrabajo")),
        (
            "Artículo 23 bis. Desconexión digital.",
            ("article", "Artículo 23 bis", "Desconexión digital"),
        ),
        (
            "Disposición adicional primera. Igualdad.",
            ("provision", "Disposición adicional primera", "Igualdad"),
        ),
        ("Disposición preliminar segunda.", ("provision", "Disposición preliminar segunda", "")),
        (
            "Cláusula primera. Competencias y funciones.",
            ("other", "Cláusula primera", "Competencias y funciones"),
        ),
    ],
)
def test_parse_heading(heading: str, expected: tuple[str, str, str]) -> None:
    assert parse_heading(heading) == expected


def test_split_text_keeps_paragraphs_together_until_the_limit() -> None:
    pieces = split_text(["a" * 40, "b" * 40, "c" * 40], max_chars=90)

    assert pieces == ["a" * 40 + "\n" + "b" * 40, "c" * 40]


def test_split_text_cuts_a_paragraph_longer_than_the_limit_at_a_space() -> None:
    long = " ".join(["palabra"] * 30)  # 239 characters

    pieces = split_text([long], max_chars=100)

    assert all(len(p) <= 100 for p in pieces)
    assert " ".join(pieces).split() == long.split()


def test_minimal_xml_with_chapters_annex_articles_and_quotes() -> None:
    xml = """<?xml version="1.0" encoding="utf-8"?>
<documento><metadatos>
  <identificador>BOE-A-2099-1</identificador><titulo>Convenio de prueba</titulo>
  <fecha_publicacion>20990101</fecha_publicacion><url_pdf>/boe/dias/x.pdf</url_pdf>
</metadatos><texto>
  <p class="articulo">Primero.</p><p class="parrafo">Ordenar la inscripción.</p>
  <p class="capitulo_num">CAPÍTULO I</p><p class="capitulo_tit">Jornada</p>
  <p class="articulo">Artículo 1. Jornada anual.</p><p class="parrafo">1.760 horas.</p>
  <p class="articulo">«Artículo 25. Texto de una ley citada.</p>
  <p class="parrafo">Sigue la cita.</p>
  <p class="articulo">Artículo 2. Vacaciones.</p>
  <table><tr><th>Grupo</th><th>Días</th></tr><tr><td>A</td><td>23</td></tr></table>
  <p class="anexo_num">ANEXO II</p><p class="anexo_tit">Reglamento de la comisión</p>
  <p class="parrafo">Introducción del reglamento.</p>
  <p class="articulo">Artículo 1. Composición.</p><p class="parrafo">Cuatro miembros.</p>
</texto></documento>""".encode()

    parsed = parse_boe_xml(xml)

    assert parsed.pdf_url == "https://www.boe.es/boe/dias/x.pdf"
    assert parsed.publication_date == date(2099, 1, 1)
    assert [(c.kind, c.ref, c.title, c.chapter) for c in parsed.chunks] == [
        ("article", "Artículo 1", "Jornada anual", "CAPÍTULO I. Jornada"),
        ("article", "Artículo 2", "Vacaciones", "CAPÍTULO I. Jornada"),
        ("annex", "Anexo II", "Reglamento de la comisión", None),
        ("annex", "Anexo II · Artículo 1", "Composición", "Reglamento de la comisión"),
    ]
    assert "«Artículo 25" in parsed.chunks[0].text  # the quote stays in its article
    assert parsed.chunks[1].text == "Grupo | Días\nA | 23"  # tables become text
    assert all("Ordenar la inscripción" not in c.text for c in parsed.chunks)


def test_rejects_xml_that_is_not_a_boe_disposition() -> None:
    with pytest.raises(ValueError, match="metadatos"):
        parse_boe_xml(b"<html><body>error</body></html>")


# --- the real agreements (facts checked by hand on boe.es) ---------------------


@pytest.mark.parametrize("name", ["consultancy", "metal"])
def test_real_agreements_have_clean_unique_chunks(
    name: str, request: pytest.FixtureRequest
) -> None:
    parsed: ParsedAgreement = request.getfixturevalue(name)

    keys = Counter((c.ref, c.part) for c in parsed.chunks)
    assert all(n == 1 for n in keys.values()), "a citation reference appears twice"
    assert all(c.text for c in parsed.chunks)
    assert all(len(c.text) <= MAX_CHUNK_CHARS for c in parsed.chunks)
    assert [c.position for c in parsed.chunks] == list(range(len(parsed.chunks)))
    assert all("Ordenar la inscripción" not in c.text for c in parsed.chunks)
    assert parsed.pdf_url.startswith("https://www.boe.es/boe/dias/")


def test_consultancy_agreement_facts(consultancy: ParsedAgreement) -> None:
    assert consultancy.boe_id == "BOE-A-2025-7766"
    assert consultancy.publication_date == date(2025, 4, 16)
    assert "XIX Convenio colectivo estatal de empresas de consultoría" in consultancy.title
    vacations = next(c for c in consultancy.chunks if c.ref == "Artículo 21")
    assert vacations.title == "Vacaciones"
    assert "veintitrés (23) días laborables de vacaciones" in vacations.text
    articles = {c.ref for c in consultancy.chunks if c.kind == "article"}
    assert {"Artículo 1", "Artículo 42"} <= articles
    # Laws quoted inside an article are not separate articles.
    assert not any(c.ref.startswith("«") for c in consultancy.chunks)
    assert any("«Artículo 25" in c.text for c in consultancy.chunks)
    assert any(c.ref == "Anexo I" and c.title == "Tablas salariales" for c in consultancy.chunks)


def test_metal_agreement_facts(metal: ParsedAgreement) -> None:
    assert metal.boe_id == "BOE-A-2022-479"
    assert metal.publication_date == date(2022, 1, 12)
    assert "IV Convenio colectivo estatal de la industria" in metal.title
    articles = {c.ref for c in metal.chunks if c.kind == "article"}
    assert {"Artículo 1", "Artículo 123"} <= articles
    assert not any(c.ref == "Artículo 124" for c in metal.chunks)
    # Articles of regulations inside annexes are cited through their annex.
    assert any(c.ref.startswith("Anexo ") and " · Artículo " in c.ref for c in metal.chunks)
    first = next(c for c in metal.chunks if c.ref == "Artículo 1")
    assert first.chapter is not None and first.chapter.startswith("CAPÍTULO I")
