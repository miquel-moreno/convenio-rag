"""Fixtures for tests that need a real PostgreSQL with pgvector.

Set TEST_DATABASE_URL (CI does, with a service container). Locally:
    docker run -d -e POSTGRES_USER=app -e POSTGRES_PASSWORD=app -e POSTGRES_DB=app_test \\
        -p 55434:5432 pgvector/pgvector:pg16
    TEST_DATABASE_URL=postgresql+asyncpg://app:app@localhost:55434/app_test make test
The database is migrated to head once and emptied before each test.
"""

import os
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from convenio_rag.adapters.db import make_engine, make_session_factory

ROOT = Path(__file__).resolve().parents[2]
TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if "tests/pg/" in item.nodeid.replace("\\", "/"):
            item.add_marker(pytest.mark.pg)
            if not TEST_DATABASE_URL:
                item.add_marker(pytest.mark.skip(reason="TEST_DATABASE_URL not set"))


def alembic_config() -> Config:
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", TEST_DATABASE_URL)
    config.attributes["configure_logger"] = False
    return config


@pytest.fixture(scope="session")
def migrated() -> Iterator[None]:
    command.upgrade(alembic_config(), "head")
    yield


@pytest.fixture
async def pg_session(migrated: None) -> AsyncIterator[AsyncSession]:
    engine = make_engine(TEST_DATABASE_URL)
    async with engine.begin() as conn:
        await conn.execute(text("TRUNCATE agreements, chunks RESTART IDENTITY CASCADE"))
    async with make_session_factory(engine)() as session:
        yield session
    await engine.dispose()
