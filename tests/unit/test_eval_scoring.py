import json
from pathlib import Path

import pytest
from evals.run import estimate_cost, load_questions
from evals.scoring import (
    Question,
    facts_present,
    reciprocal_rank,
    score_answer,
    summarize_answers,
    summarize_retrieval,
)

ROOT = Path(__file__).resolve().parents[2]
Q = Question("c01", "¿Vacaciones?", "BOE-A-2025-7766", ["Artículo 21"], [["23", "veintitrés"]])
TRICK = Question("t01", "¿Coche de empresa?", "BOE-A-2025-7766", trick=True)


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(["Artículo 21", "Artículo 22"], ["Artículo 21"]) == 1.0
    assert reciprocal_rank(["Artículo 20", "Artículo 21"], ["Artículo 21"]) == 0.5
    assert reciprocal_rank(["Artículo 20"], ["Artículo 21"]) == 0.0


def test_facts_need_every_group_and_any_option() -> None:
    assert facts_present("Tienes veintitrés días", [["23", "veintitrés"]])
    assert facts_present("En JULIO y en diciembre", [["julio"], ["diciembre"]])
    assert not facts_present("Solo en julio", [["julio"], ["diciembre"]])


def test_answer_is_correct_only_with_citation_and_facts() -> None:
    assert score_answer(Q, True, "23 días", ["Artículo 21"]).correct
    assert not score_answer(Q, True, "23 días", ["Artículo 22"]).correct  # wrong article
    assert not score_answer(Q, True, "30 días", ["Artículo 21"]).correct  # wrong fact
    assert not score_answer(Q, False, "No lo he encontrado", []).correct


def test_trick_question_is_correct_only_when_refused() -> None:
    assert score_answer(TRICK, False, "No lo he encontrado", []).correct
    assert not score_answer(TRICK, True, "Sí, un coche", ["Artículo 30"]).correct


def test_summaries() -> None:
    retrieval = summarize_retrieval({"hybrid": [1.0, 0.5, 0.0, 1.0]}, k=5)
    answers = summarize_answers(
        [
            score_answer(Q, True, "23 días", ["Artículo 21"]),
            score_answer(Q, False, "", []),
            score_answer(Q, True, "30 días", ["Artículo 21"]),
            score_answer(TRICK, False, "", []),
        ]
    )

    assert retrieval == {"hybrid": {"hit@5": 75.0, "mrr": 0.625}}
    assert answers["answerable"] == 3 and answers["correct"] == 1
    assert answers["correct_pct"] == 33.3
    assert answers["not_found_on_answerable"] == 1
    assert answers["wrong_answers"] == 1
    assert (answers["tricks"], answers["tricks_refused"]) == (1, 1)


def test_the_question_set_is_complete_and_consistent() -> None:
    questions = load_questions()
    raw = json.loads((ROOT / "evals" / "questions.json").read_text(encoding="utf-8"))

    assert len(questions) == len(raw["questions"]) == 35
    assert sum(q.trick for q in questions) == 5
    assert len({q.id for q in questions}) == 35
    for q in questions:
        assert q.agreement in {"BOE-A-2025-7766", "BOE-A-2022-479"}
        if not q.trick:
            assert q.expected_refs and q.facts, q.id


def test_expected_articles_exist_in_the_agreements() -> None:
    from convenio_rag.services.agreements import parse_boe_xml

    refs = {
        boe_id: {
            c.ref
            for c in parse_boe_xml((ROOT / "data" / "boe" / f"{boe_id}.xml").read_bytes()).chunks
        }
        for boe_id in ("BOE-A-2025-7766", "BOE-A-2022-479")
    }
    for q in load_questions():
        for ref in q.expected_refs:
            assert q.agreement is not None and ref in refs[q.agreement], (q.id, ref)


@pytest.mark.parametrize(
    ("model", "provider", "expected"),
    [
        ("gpt-4.1-mini-2025-04-14", "openai", 2.0),
        ("qwen2.5:3b", "ollama", 0.0),
        ("x", "openai", None),
    ],
)
def test_cost_estimate(model: str, provider: str, expected: float | None) -> None:
    assert estimate_cost(model, 1_000_000, 1_000_000, provider) == expected
