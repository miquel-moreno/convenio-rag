from convenio_rag.services.answer import (
    ANSWER_SCHEMA,
    NOT_FOUND,
    LLMAnswer,
    Source,
    build_messages,
    verify,
)
from convenio_rag.services.search import SearchResult


def result(ref: str, title: str = "Vacaciones") -> SearchResult:
    return SearchResult(
        chunk_id=1,
        agreement_id="BOE-A-2025-7766",
        agreement="Consultoría y TI (XIX)",
        ref=ref,
        title=title,
        part=1,
        text="…",
        source_url="https://www.boe.es/diario_boe/txt.php?id=BOE-A-2025-7766",
        score=0.1,
    )


def test_prompt_numbers_the_sources_and_includes_the_question() -> None:
    sources = [
        Source(1, result("Artículo 21"), "23 días laborables de vacaciones."),
        Source(2, result("Artículo 22", "Permisos retribuidos"), "15 días por matrimonio."),
    ]

    system, user = build_messages("  ¿Cuántos días de vacaciones tengo? ", sources)

    assert system.role == "system" and "ONLY the numbered sources" in system.content
    assert (
        "[1] Consultoría y TI (XIX) · Artículo 21. Vacaciones\n23 días laborables" in user.content
    )
    assert "[2] Consultoría y TI (XIX) · Artículo 22. Permisos retribuidos" in user.content
    assert user.content.endswith("Question: ¿Cuántos días de vacaciones tengo?")


def test_prompt_without_sources_says_so() -> None:
    _, user = build_messages("¿Algo?", [])

    assert "(no sources)" in user.content


def test_valid_answer_keeps_its_citations_once() -> None:
    verified = verify(LLMAnswer(found=True, answer=" 23 días. ", citations=[1, 1, 2]), 5)

    assert verified.found and verified.answer == "23 días."
    assert verified.cited == [1, 2]
    assert verified.dropped == []


def test_invented_citations_are_dropped() -> None:
    verified = verify(LLMAnswer(found=True, answer="23 días.", citations=[2, 9, 0]), 5)

    assert verified.cited == [2]
    assert verified.dropped == [9, 0]


def test_answer_without_any_valid_citation_is_not_trusted() -> None:
    verified = verify(LLMAnswer(found=True, answer="30 días.", citations=[7]), 5)

    assert not verified.found
    assert verified.answer == NOT_FOUND
    assert verified.dropped == [7]


def test_not_found_answer_is_respected() -> None:
    verified = verify(LLMAnswer(found=False, answer="", citations=[]), 5)

    assert not verified.found and verified.answer == NOT_FOUND


def test_found_with_an_empty_answer_is_not_trusted() -> None:
    assert not verify(LLMAnswer(found=True, answer="  ", citations=[1]), 5).found


def test_answer_schema_is_strict() -> None:
    assert ANSWER_SCHEMA["additionalProperties"] is False
    assert set(ANSWER_SCHEMA["required"]) == {"found", "answer", "citations"}
