"""Database access: SQLAlchemy 2 async engine and the declarative base.

PostgreSQL in production (asyncpg), SQLite (aiosqlite) in tests: models should use
portable column types so both work. Every model change needs an Alembic migration
(tests/integration/test_migrations.py checks it).
"""

from collections.abc import Sequence
from datetime import UTC, date, datetime

from sqlalchemy import (
    Date,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    delete,
    select,
)
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

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
