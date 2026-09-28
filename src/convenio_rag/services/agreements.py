"""Split a collective agreement published in the BOE into citable chunks.

Input is the official XML (https://www.boe.es/diario_boe/xml.php?id=...), where
every heading comes as <p class="articulo">, chapters as <p class="capitulo...">
and annexes as <p class="anexo_num"> / <p class="anexo_tit">. One chunk per
article, provision or annex; long ones are split on paragraph boundaries so each
chunk fits the embedding model. Pure functions: no network, no database.
"""

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from xml.etree.ElementTree import Element

from defusedxml.ElementTree import fromstring

BOE_BASE = "https://www.boe.es"
# The embedding model reads ~512 tokens; ~1500 characters of Spanish stay below it.
MAX_CHUNK_CHARS = 1500

# The resolution that publishes the agreement ("Primero. Ordenar la inscripción...")
# is not part of the agreement itself.
RESOLUTION_HEADINGS = {"Primero", "Segundo", "Tercero", "Cuarto"}
TEXT_CLASSES = {
    "parrafo",
    "parrafo_2",
    "centro_redonda",
    "centro_cursiva",
    "cita_con_pleca",
    "cuerpo_tabla_izq",
    "cuerpo_tabla_centro",
}

_ARTICLE_RE = re.compile(r"^(Artículo\s+\d+(?:\s*(?:bis|ter|[A-Z]))?)\.\s*(.*)$")
_PROVISION_RE = re.compile(
    r"^(Disposici[oó]n\s+(?:preliminar|adicional|transitoria|final|derogatoria)"
    r"(?:\s+[a-zá-úñ]+)?)\.?\s*(.*)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Chunk:
    position: int
    kind: str  # article | provision | annex | other
    ref: str  # "Artículo 23", "Disposición adicional primera", "Anexo I"
    title: str
    chapter: str | None
    part: int  # 1, 2... when a long section is split
    text: str


@dataclass(frozen=True)
class ParsedAgreement:
    boe_id: str
    title: str
    publication_date: date
    source_url: str
    pdf_url: str
    chunks: list[Chunk] = field(default_factory=list)


def clean(text: str) -> str:
    """Collapse whitespace, including the non-breaking spaces the BOE uses."""
    return " ".join(text.replace("\xa0", " ").split())


def parse_heading(text: str) -> tuple[str, str, str]:
    """'Artículo 23. Vacaciones.' -> ('article', 'Artículo 23', 'Vacaciones')."""
    text = clean(text)
    if match := _ARTICLE_RE.match(text):
        ref, title = match.groups()
        return "article", clean(ref), title.rstrip(".").strip()
    if match := _PROVISION_RE.match(text):
        ref, title = match.groups()
        return "provision", clean(ref), title.rstrip(".").strip()
    # Any other heading ("Cláusula primera. Competencias"): number before the first dot.
    ref, _, title = text.partition(". ")
    return "other", ref.rstrip("."), title.rstrip(".").strip()


def _element_text(element: Element) -> str:
    return clean("".join(element.itertext()))


def _table_text(table: Element) -> str:
    rows = []
    for row in table.iter("tr"):
        cells = [_element_text(cell) for cell in row if cell.tag in ("td", "th")]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def split_text(paragraphs: list[str], max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Group paragraphs into pieces of at most max_chars (a longer paragraph is cut)."""
    pieces: list[str] = []
    current = ""
    for paragraph in paragraphs:
        while len(paragraph) > max_chars:
            cut = paragraph.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            if current:
                pieces.append(current)
                current = ""
            pieces.append(paragraph[:cut])
            paragraph = paragraph[cut:].strip()
        if current and len(current) + 1 + len(paragraph) > max_chars:
            pieces.append(current)
            current = paragraph
        else:
            current = f"{current}\n{paragraph}" if current else paragraph
    if current:
        pieces.append(current)
    return pieces


@dataclass
class _Section:
    kind: str
    ref: str
    title: str
    chapter: str | None
    paragraphs: list[str] = field(default_factory=list)


def _annex_ref(text: str) -> str:
    """'ANEXO II' -> 'Anexo II' (keeps Roman numerals in capitals)."""
    words = clean(text).split(" ", 1)
    return words[0].capitalize() + (f" {words[1].upper()}" if len(words) > 1 else "")


def _sections(texto: Element) -> list[_Section]:
    sections: list[_Section] = []
    current: _Section | None = None
    chapter: str | None = None
    chapter_num = ""
    annex: _Section | None = None  # set once the annexes start
    for element in texto:
        css = element.get("class", "")
        if element.tag == "table":
            if current is not None and (table := _table_text(element)):
                current.paragraphs.append(table)
            continue
        text = _element_text(element)
        if not text:
            continue
        if css in ("capitulo", "titulo"):
            chapter = text
        elif css in ("capitulo_num", "titulo_num"):
            chapter_num = text
        elif css in ("capitulo_tit", "titulo_tit"):
            chapter = f"{chapter_num}. {text}" if chapter_num else text
        elif css == "articulo" and text.startswith(("«", '"', "“")):
            # A law quoted inside an article: part of that article, not a new one.
            if current is not None:
                current.paragraphs.append(text)
        elif css == "articulo":
            kind, ref, title = parse_heading(text)
            if annex is not None:
                # Articles of a regulation inside an annex: cite them through the annex.
                current = _Section("annex", f"{annex.ref} · {ref}", title, annex.title or None)
            else:
                current = _Section(kind, ref, title, chapter)
            sections.append(current)
        elif css == "anexo_num":
            annex = _Section("annex", _annex_ref(text), "", None)
            current = annex
            sections.append(current)
        elif css == "anexo_tit" and annex is not None and current is annex:
            annex.title = text
        elif current is not None and (css in TEXT_CLASSES or element.tag == "p"):
            current.paragraphs.append(text)
    return sections


def _absolute(url: str) -> str:
    return url if url.startswith("http") else BOE_BASE + url


def parse_boe_xml(xml: bytes) -> ParsedAgreement:
    root = fromstring(xml)
    meta = root.find("metadatos")
    texto = root.find("texto")
    if meta is None or texto is None:
        raise ValueError("not a BOE disposition XML (missing <metadatos> or <texto>)")

    boe_id = meta.findtext("identificador", "").strip()
    chunks: list[Chunk] = []
    for section in _sections(texto):
        if section.kind == "other" and section.ref in RESOLUTION_HEADINGS:
            continue
        pieces = split_text(section.paragraphs)  # a heading with no text gives no chunk
        for part, piece in enumerate(pieces, start=1):
            chunks.append(
                Chunk(
                    position=len(chunks),
                    kind=section.kind,
                    ref=section.ref,
                    title=section.title,
                    chapter=section.chapter,
                    part=part,
                    text=piece,
                )
            )
    return ParsedAgreement(
        boe_id=boe_id,
        title=clean(meta.findtext("titulo", "")),
        publication_date=datetime.strptime(meta.findtext("fecha_publicacion", ""), "%Y%m%d").date(),
        source_url=f"{BOE_BASE}/diario_boe/txt.php?id={boe_id}",
        pdf_url=_absolute(meta.findtext("url_pdf", "").strip()),
        chunks=chunks,
    )
