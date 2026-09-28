"""Scoring for the RAG evaluation. Pure functions, unit tested.

Retrieval: hit@k (an expected article among the top k) and MRR (1 / rank of the
first expected article, 0 if absent), without any LLM.
Answer: the answer cites an expected article, contains the key facts, and trick
questions (not in the agreement) are answered with "not found".
"""

import statistics
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    agreement: str | None
    expected_refs: list[str] = field(default_factory=list)
    facts: list[list[str]] = field(default_factory=list)
    trick: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Question":
        return cls(
            id=data["id"],
            question=data["question"],
            agreement=data.get("agreement"),
            expected_refs=list(data.get("expected_refs", [])),
            facts=[list(group) for group in data.get("facts", [])],
            trick=bool(data.get("trick", False)),
        )


def reciprocal_rank(retrieved: list[str], expected: list[str]) -> float:
    for position, ref in enumerate(retrieved, start=1):
        if ref in expected:
            return 1.0 / position
    return 0.0


def facts_present(answer: str, facts: list[list[str]]) -> bool:
    text = answer.lower()
    return all(any(option.lower() in text for option in group) for group in facts)


@dataclass(frozen=True)
class AnswerScore:
    found: bool
    cites_expected: bool
    facts_ok: bool
    trick: bool

    @property
    def correct(self) -> bool:
        if self.trick:
            return not self.found
        return self.found and self.cites_expected and self.facts_ok


def score_answer(question: Question, found: bool, answer: str, cited: list[str]) -> AnswerScore:
    return AnswerScore(
        found=found,
        cites_expected=any(ref in question.expected_refs for ref in cited),
        facts_ok=facts_present(answer, question.facts) if found else False,
        trick=question.trick,
    )


def _pct(part: float, total: int) -> float:
    return round(100 * part / total, 1) if total else 0.0


def summarize_retrieval(ranks: dict[str, list[float]], *, k: int) -> dict[str, dict[str, float]]:
    """ranks: mode -> reciprocal rank per answerable question."""
    return {
        mode: {
            f"hit@{k}": _pct(sum(1 for r in values if r > 0), len(values)),
            "mrr": round(statistics.mean(values), 3) if values else 0.0,
        }
        for mode, values in ranks.items()
    }


def summarize_answers(scores: list[AnswerScore]) -> dict[str, Any]:
    answerable = [s for s in scores if not s.trick]
    tricks = [s for s in scores if s.trick]
    return {
        "answerable": len(answerable),
        "correct": sum(s.correct for s in answerable),
        "correct_pct": _pct(sum(s.correct for s in answerable), len(answerable)),
        "cites_expected_pct": _pct(
            sum(s.found and s.cites_expected for s in answerable), len(answerable)
        ),
        "facts_ok_pct": _pct(sum(s.facts_ok for s in answerable), len(answerable)),
        "not_found_on_answerable": sum(not s.found for s in answerable),
        # The one that matters most: a wrong answer presented as an answer.
        "wrong_answers": sum(s.found and not s.correct for s in answerable),
        "tricks": len(tricks),
        "tricks_refused": sum(s.correct for s in tricks),
    }
