"""Evaluation of retrieval and answers (`make eval`).

Needs the agreements loaded with embeddings (`python -m scripts.ingest`). The
retrieval part is free; the answer part calls a REAL LLM (local only, never CI).

    uv run python -m evals.run                                   # .env settings
    uv run python -m evals.run --provider openai --model gpt-4.1-mini
    uv run python -m evals.run --retrieval-only                  # no LLM, no cost

Results go to evals/results/<UTC date and time>_<model>.json.
"""

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from convenio_rag.adapters.db import make_engine, make_session_factory
from convenio_rag.adapters.embeddings import Embedder, FastEmbedEmbedder
from convenio_rag.adapters.llm import LLMClient, LLMError, build_llm_client
from convenio_rag.core.config import Settings
from convenio_rag.services.answer import ask
from convenio_rag.services.search import SearchMode, search
from evals.scoring import (
    Question,
    reciprocal_rank,
    score_answer,
    summarize_answers,
    summarize_retrieval,
)

HERE = Path(__file__).parent
RESULTS = HERE / "results"
TOP_K = 5
ANSWER_MODES = (SearchMode.HYBRID, SearchMode.VECTOR)
# USD per 1M tokens (input, output), checked on developers.openai.com on 2026-09-28.
PRICES_PER_MILLION = {"gpt-4.1-mini": (0.40, 1.60)}


def load_questions(path: Path = HERE / "questions.json") -> list[Question]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Question.from_dict(q) for q in data["questions"]]


async def evaluate_retrieval(
    session: AsyncSession, embedder: Embedder, questions: list[Question]
) -> tuple[dict[str, list[float]], list[dict[str, Any]]]:
    ranks: dict[str, list[float]] = {mode.value: [] for mode in SearchMode}
    details = []
    for q in (q for q in questions if not q.trick):
        row: dict[str, Any] = {"id": q.id, "expected": q.expected_refs}
        for mode in SearchMode:
            results = await search(
                session, embedder, q.question, limit=TOP_K, mode=mode, agreement_id=q.agreement
            )
            refs = [r.ref for r in results]
            ranks[mode.value].append(reciprocal_rank(refs, q.expected_refs))
            row[mode.value] = refs
        details.append(row)
    return ranks, details


async def evaluate_answers(
    session: AsyncSession,
    embedder: Embedder,
    llm: LLMClient,
    questions: list[Question],
    mode: SearchMode,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scores, details, latencies = [], [], []
    input_tokens = output_tokens = 0
    model = ""
    for q in questions:
        result = await ask(
            q.question,
            session=session,
            embedder=embedder,
            llm=llm,
            agreement_id=q.agreement,
            mode=mode,
        )
        model = result.model or model
        input_tokens += result.input_tokens
        output_tokens += result.output_tokens
        latencies.append(result.latency_ms / 1000)
        cited = [c.ref for c in result.citations]
        score = score_answer(q, result.found, result.answer, cited)
        scores.append(score)
        details.append(
            {
                "id": q.id,
                "correct": score.correct,
                "found": result.found,
                "cited": cited,
                "expected": q.expected_refs,
                "facts_ok": score.facts_ok,
                "answer": result.answer,
            }
        )
        print(f"  [{mode.value}] {'OK ' if score.correct else 'ERR'} {q.id} {cited}", flush=True)
    summary = summarize_answers(scores)
    summary.update(
        model=model,
        seconds_per_question_mean=round(statistics.mean(latencies), 2) if latencies else 0.0,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    return summary, details


def estimate_cost(model: str, input_tokens: int, output_tokens: int, provider: str) -> float | None:
    if provider == "ollama":
        return 0.0
    for name, (price_in, price_out) in PRICES_PER_MILLION.items():
        if model == name or model.startswith(f"{name}-"):
            return round((input_tokens * price_in + output_tokens * price_out) / 1_000_000, 4)
    return None


async def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and answers.")
    parser.add_argument("--provider", choices=["openai", "ollama"])
    parser.add_argument("--model")
    parser.add_argument("--retrieval-only", action="store_true")
    args = parser.parse_args()
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
    if args.model:
        os.environ["LLM_MODEL"] = args.model

    settings = Settings()
    questions = load_questions()
    embedder = FastEmbedEmbedder(settings.embedding_model, cache_dir=settings.embedding_cache_dir)
    engine = make_engine(settings.database_url)
    payload: dict[str, Any] = {
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "questions": len(questions),
        "embedding_model": settings.embedding_model,
    }
    try:
        async with make_session_factory(engine)() as session:
            ranks, retrieval_details = await evaluate_retrieval(session, embedder, questions)
            payload["retrieval"] = summarize_retrieval(ranks, k=TOP_K)
            payload["retrieval_details"] = retrieval_details
            print("Recuperación:", json.dumps(payload["retrieval"], ensure_ascii=False))
            if not args.retrieval_only:
                llm = build_llm_client(settings)
                payload["provider"] = settings.llm_provider
                payload["answers"] = {}
                for mode in ANSWER_MODES:
                    summary, details = await evaluate_answers(
                        session, embedder, llm, questions, mode
                    )
                    summary["cost_usd"] = estimate_cost(
                        summary["model"],
                        summary["input_tokens"],
                        summary["output_tokens"],
                        settings.llm_provider,
                    )
                    payload["answers"][mode.value] = {"summary": summary, "details": details}
                    print(f"Respuestas ({mode.value}):", json.dumps(summary, ensure_ascii=False))
    except LLMError as exc:
        print(f"Error del proveedor de IA: {exc}", file=sys.stderr)
        return 1
    finally:
        await engine.dispose()

    RESULTS.mkdir(exist_ok=True)
    model = "retrieval-only" if args.retrieval_only else settings.llm_model
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H%MZ")
    out = RESULTS / f"{stamp}_{re.sub(r'[^A-Za-z0-9._-]', '-', model)}.json"
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"Guardado en {out}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
