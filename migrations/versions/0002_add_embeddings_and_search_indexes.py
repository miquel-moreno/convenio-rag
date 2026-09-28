"""add embeddings and search indexes

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-28 20:19:39.290146

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: str | Sequence[str] | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Semantic search (pgvector + HNSW) and Spanish full-text search (GIN)."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.add_column("chunks", sa.Column("embedding", Vector(dim=384), nullable=True))
    op.create_index(
        "ix_chunks_embedding",
        "chunks",
        ["embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    # Must match adapters.db._fts_document() exactly, or the index is not used.
    op.execute(
        "CREATE INDEX ix_chunks_fts ON chunks USING gin "
        "(to_tsvector('spanish'::regconfig, ref || ' ' || title || ' ' || text))"
    )


def downgrade() -> None:
    op.drop_index("ix_chunks_fts", table_name="chunks")
    op.drop_index("ix_chunks_embedding", table_name="chunks")
    op.drop_column("chunks", "embedding")
