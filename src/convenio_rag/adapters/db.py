"""Database access: models, repository functions and the two searches.

PostgreSQL + pgvector in production. Unit tests use SQLite, which ignores the
PostgreSQL-only indexes; the searches themselves are tested against a real
PostgreSQL (tests marked `pg`). Every model change needs an Alembic migration.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ColumnElement,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    delete,
    func,
    literal_column,
    select,
    update,
)
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from convenio_rag.adapters.embeddings import EMBEDDING_DIM
from convenio_rag.services.agreements import ParsedAgreement


class Base(DeclarativeBase):
    pass


def _now() -> datetime:
    return datetime.now(UTC)


class AgreementRecord(Base):
    """A collective agreement as published in the BOE."""

    __tablename__ = "agreements"

    boe_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    short_name: Mapped[str] = mapped_column(String(100))
    title: Mapped[str] = mapped_column(Text)
    sector: Mapped[str] = mapped_column(String(200))
    note: Mapped[str | None] = mapped_column(Text)
    publication_date: Mapped[date] = mapped_column(Date)
    source_url: Mapped[str] = mapped_column(String(300))
    pdf_url: Mapped[str] = mapped_column(String(300))
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class ChunkRecord(Base):
    """One citable piece of an agreement: an article, a provision or an annex (or part)."""

    __tablename__ = "chunks"
    __table_args__ = (UniqueConstraint("agreement_id", "position"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    agreement_id: Mapped[str] = mapped_column(
        ForeignKey("agreements.boe_id", ondelete="CASCADE"), index=True
    )
    position: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(20))
    ref: Mapped[str] = mapped_column(String(300))
    title: Mapped[str] = mapped_column(Text)
    chapter: Mapped[str | None] = mapped_column(Text)
    part: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    # Meaning of ref + title + text as numbers (see adapters/embeddings.py).
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)


def _fts_document() -> ColumnElement[str]:
    """Spanish full-text document. The index and the query must use this same expression."""
    spanish: ColumnElement[str] = literal_column("'spanish'::regconfig")
    # "||", not concat_ws(): an index expression only accepts IMMUTABLE functions.
    text = ChunkRecord.ref + " " + ChunkRecord.title + " " + ChunkRecord.text
    return func.to_tsvector(spanish, text)


# PostgreSQL-only indexes (SQLite in unit tests skips them). Declared at module level so
# they attach to the table; the full-text one is an expression index.
Index("ix_chunks_fts", _fts_document(), postgresql_using="gin").ddl_if(dialect="postgresql")
Index(
    "ix_chunks_embedding",
    ChunkRecord.embedding,
    postgresql_using="hnsw",
    postgresql_ops={"embedding": "vector_cosine_ops"},
).ddl_if(dialect="postgresql")


# Alembic cannot compare expression indexes: they are written by hand in the migrations
# and ignored when comparing the models with the database.
EXPRESSION_INDEXES = {"ix_chunks_fts"}


def include_object(
    obj: object, name: str | None, type_: str, reflected: bool, compare_to: object
) -> bool:
    return not (type_ == "index" and name in EXPRESSION_INDEXES)


def make_engine(database_url: str) -> AsyncEngine:
    return create_async_engine(database_url, pool_pre_ping=True)


def make_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def replace_agreement(
    session: AsyncSession,
    parsed: ParsedAgreement,
    *,
    short_name: str,
    sector: str,
    note: str | None,
) -> int:
    """Store an agreement and its chunks, replacing any previous version. Returns chunks."""
    await session.execute(delete(ChunkRecord).where(ChunkRecord.agreement_id == parsed.boe_id))
    await session.merge(
        AgreementRecord(
            boe_id=parsed.boe_id,
            short_name=short_name,
            title=parsed.title,
            sector=sector,
            note=note,
            publication_date=parsed.publication_date,
            source_url=parsed.source_url,
            pdf_url=parsed.pdf_url,
            ingested_at=_now(),
        )
    )
    session.add_all(
        ChunkRecord(
            agreement_id=parsed.boe_id,
            position=c.position,
            kind=c.kind,
            ref=c.ref,
            title=c.title,
            chapter=c.chapter,
            part=c.part,
            text=c.text,
        )
        for c in parsed.chunks
    )
    await session.commit()
    return len(parsed.chunks)


async def list_chunks(session: AsyncSession, agreement_id: str) -> Sequence[ChunkRecord]:
    result = await session.execute(
        select(ChunkRecord)
        .where(ChunkRecord.agreement_id == agreement_id)
        .order_by(ChunkRecord.position)
    )
    return result.scalars().all()


def embedding_text(chunk: ChunkRecord) -> str:
    return f"{chunk.ref}. {chunk.title}\n{chunk.text}"


async def set_embeddings(
    session: AsyncSession, chunk_ids: Sequence[int], vectors: Sequence[list[float]]
) -> None:
    for chunk_id, vector in zip(chunk_ids, vectors, strict=True):
        await session.execute(
            update(ChunkRecord).where(ChunkRecord.id == chunk_id).values(embedding=vector)
        )
    await session.commit()


@dataclass(frozen=True)
class Hit:
    chunk_id: int
    score: float


def _or_query(question: str) -> str:
    """'¿Cuántos días de vacaciones?' -> 'cuántos | días | vacaciones'.

    OR instead of AND: a question has words the article does not ("tengo", "cuántos");
    ranking puts the chunks with more (and rarer) matching words first. Only word
    characters reach to_tsquery, so the query cannot be malformed.
    """
    return " | ".join(re.findall(r"\w+", question.lower()))


async def fulltext_search(
    session: AsyncSession, question: str, *, limit: int, agreement_id: str | None = None
) -> list[Hit]:
    """PostgreSQL only: Spanish stemming and stop words, ranked by ts_rank_cd."""
    terms = _or_query(question)
    if not terms:
        return []
    query = func.to_tsquery(literal_column("'spanish'::regconfig"), terms)
    rank = func.ts_rank_cd(_fts_document(), query)
    stmt = (
        select(ChunkRecord.id, rank.label("rank"))
        .where(_fts_document().op("@@")(query))
        .order_by(rank.desc(), ChunkRecord.id)
        .limit(limit)
    )
    if agreement_id:
        stmt = stmt.where(ChunkRecord.agreement_id == agreement_id)
    return [Hit(row.id, float(row.rank)) for row in await session.execute(stmt)]


async def vector_search(
    session: AsyncSession, vector: list[float], *, limit: int, agreement_id: str | None = None
) -> list[Hit]:
    """PostgreSQL + pgvector only: nearest chunks by cosine distance (HNSW index)."""
    distance = ChunkRecord.embedding.cosine_distance(vector)
    stmt = (
        select(ChunkRecord.id, distance.label("distance"))
        .where(ChunkRecord.embedding.is_not(None))
        .order_by(distance, ChunkRecord.id)
        .limit(limit)
    )
    if agreement_id:
        stmt = stmt.where(ChunkRecord.agreement_id == agreement_id)
    return [Hit(row.id, 1.0 - float(row.distance)) for row in await session.execute(stmt)]


async def get_chunks_with_agreements(
    session: AsyncSession, chunk_ids: Sequence[int]
) -> dict[int, tuple[ChunkRecord, AgreementRecord]]:
    result = await session.execute(
        select(ChunkRecord, AgreementRecord)
        .join(AgreementRecord, AgreementRecord.boe_id == ChunkRecord.agreement_id)
        .where(ChunkRecord.id.in_(list(chunk_ids)))
    )
    return {chunk.id: (chunk, agreement) for chunk, agreement in result}
