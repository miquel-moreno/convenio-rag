"""Answer a question from the agreements, citing the articles it comes from (RAG).

1. Retrieve the most relevant articles (hybrid search).
2. Give them to the LLM as numbered sources, with strict rules: use only these
   sources, cite them by number, and say so when the answer is not there.
3. Verify the answer: every citation must be one of the sources we gave. Invented
   citations are dropped; an answer with no valid citation is not trusted.
The LLM proposes, the code verifies (same idea as the business rules of P1).
"""

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from convenio_rag.adapters.db import get_article_text
from convenio_rag.adapters.embeddings import Embedder
from convenio_rag.adapters.llm import ChatMessage, LLMClient, strict_json_schema
from convenio_rag.services.search import SearchResult, search

TOP_K = 5
MAX_SOURCE_CHARS = 4000  # an article longer than this is cut (salary tables, annexes)
MAX_ATTEMPTS = 2

NOT_FOUND = (
    "No he encontrado la respuesta en el convenio. Puede que no lo regule o que remita a "
    "otro convenio o a la ley; consulta el texto completo en el BOE."
)

SYSTEM_PROMPT = """\
You answer questions from workers and small companies about Spanish collective \
labour agreements (convenios colectivos). Answer in Spanish.

Rules:
- Use ONLY the numbered sources below. Do not use your own knowledge of labour law \
or of other agreements, and never invent numbers, days or amounts.
- If the sources contain the answer, set found=true, answer in 1 to 3 short, plain \
sentences, and list in "citations" the numbers of the sources you used.
- If the agreement explicitly leaves the matter to other agreements or to the law, \
that is an answer: say so and cite that source.
- If the sources do not contain the answer, set found=false, citations=[] and \
answer with an empty string.
"""

RETRY_PROMPT = """\
Your previous answer was not valid:
{errors}
Return the corrected JSON."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LLMAnswer(_Strict):
    found: bool
    answer: str
    citations: list[int]


ANSWER_SCHEMA = strict_json_schema(LLMAnswer)


@dataclass(frozen=True)
class Source:
    number: int
    result: SearchResult
    text: str  # the full article (all parts), possibly cut


@dataclass(frozen=True)
class Citation:
    agreement_id: str
    agreement: str
    ref: str
    title: str
    source_url: str


@dataclass(frozen=True)
class AskResult:
    question: str
    found: bool
    answer: str
    citations: list[Citation]
    sources: list[Citation]  # what was given to the LLM, in order
    dropped_citations: list[int]  # numbers the LLM cited that were not sources
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


def _cite(result: SearchResult) -> Citation:
    return Citation(
        agreement_id=result.agreement_id,
        agreement=result.agreement,
        ref=result.ref,
        title=result.title,
        source_url=result.source_url,
    )


def build_messages(question: str, sources: list[Source]) -> list[ChatMessage]:
    blocks = []
    for source in sources:
        r = source.result
        heading = f"[{source.number}] {r.agreement} · {r.ref}" + (f". {r.title}" if r.title else "")
        blocks.append(f"{heading}\n{source.text}")
    context = "\n\n".join(blocks) if blocks else "(no sources)"
    return [
        ChatMessage("system", SYSTEM_PROMPT),
        ChatMessage("user", f"Sources:\n\n{context}\n\nQuestion: {question.strip()}"),
    ]


@dataclass(frozen=True)
class Verified:
    found: bool
    answer: str
    cited: list[int]
    dropped: list[int]


def verify(answer: LLMAnswer, source_count: int) -> Verified:
    """Keep only citations that point to real sources; no valid citation, no answer."""
    valid = [n for n in dict.fromkeys(answer.citations) if 1 <= n <= source_count]
    dropped = [n for n in answer.citations if not 1 <= n <= source_count]
    if not answer.found or not answer.answer.strip() or not valid:
        return Verified(False, NOT_FOUND, [], dropped)
    return Verified(True, answer.answer.strip(), valid, dropped)


async def gather_sources(session: AsyncSession, results: list[SearchResult]) -> list[Source]:
    sources = []
    for number, result in enumerate(results, start=1):
        text = await get_article_text(session, result.agreement_id, result.ref)
        sources.append(Source(number, result, text[:MAX_SOURCE_CHARS]))
    return sources


async def ask(
    question: str,
    *,
    session: AsyncSession,
    embedder: Embedder,
    llm: LLMClient,
    agreement_id: str | None = None,
    top_k: int = TOP_K,
) -> AskResult:
    results = await search(session, embedder, question, limit=top_k, agreement_id=agreement_id)
    sources = await gather_sources(session, results)
    messages = build_messages(question, sources)

    model, input_tokens, output_tokens, latency_ms = "", 0, 0, 0.0
    parsed: LLMAnswer | None = None
    for _ in range(MAX_ATTEMPTS):
        response = await llm.chat(messages, json_schema=ANSWER_SCHEMA, schema_name="answer")
        model = response.model
        input_tokens += response.input_tokens
        output_tokens += response.output_tokens
        latency_ms += response.latency_ms
        try:
            parsed = LLMAnswer.model_validate_json(response.text)
            break
        except ValidationError as exc:
            errors = "\n".join(f"- {e['loc']}: {e['msg']}" for e in exc.errors()[:10])
            messages = [
                *messages,
                ChatMessage("assistant", response.text),
                ChatMessage("user", RETRY_PROMPT.format(errors=errors)),
            ]

    verified = (
        verify(parsed, len(sources)) if parsed is not None else Verified(False, NOT_FOUND, [], [])
    )
    return AskResult(
        question=question,
        found=verified.found,
        answer=verified.answer,
        citations=[_cite(sources[n - 1].result) for n in verified.cited],
        sources=[_cite(s.result) for s in sources],
        dropped_citations=verified.dropped,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
    )
