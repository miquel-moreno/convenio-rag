"""Database access: SQLAlchemy 2 async engine and the declarative base.

PostgreSQL in production (asyncpg), SQLite (aiosqlite) in tests: models should use
portable column types so both work. Every model change needs an Alembic migration
(tests/integration/test_migrations.py checks it).
"""

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


def make_engine(database_url: str) -> AsyncEngine:
    return create_async_engine(database_url, pool_pre_ping=True)


def make_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)
