"""Text embeddings: a list of numbers per text; similar meaning -> similar numbers.

Search depends on the Embedder Protocol. FastEmbedEmbedder runs a multilingual
model locally with ONNX (no API cost, no GPU); tests use FakeEmbedder.
"""

import asyncio
import hashlib
import math
import re
from collections.abc import Sequence
from typing import Any, Protocol

EMBEDDING_DIM = 384  # dimension of the default model; changing it needs a migration


class Embedder(Protocol):
    dim: int

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class FastEmbedEmbedder:
    """Local ONNX model, loaded on first use (downloaded once into cache_dir)."""

    def __init__(self, model_name: str, *, cache_dir: str | None = None) -> None:
        self.model_name = model_name
        self.dim = EMBEDDING_DIM
        self._cache_dir = cache_dir
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from fastembed import TextEmbedding  # heavy import, only when needed

            self._model = TextEmbedding(self.model_name, cache_dir=self._cache_dir)
        return self._model

    def _embed_sync(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(x) for x in vector] for vector in self._load().embed(list(texts))]

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        # CPU-bound: run in a thread so the event loop keeps serving requests.
        return await asyncio.to_thread(self._embed_sync, texts)


class FakeEmbedder:
    """Deterministic bag-of-words vectors: texts sharing words are similar. Tests only."""

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim
        self.calls = 0

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for word in re.findall(r"\w+", text.lower()):
            slot = int(hashlib.sha256(word.encode()).hexdigest(), 16) % self.dim
            vector[slot] += 1.0
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        return [self._vector(t) for t in texts]
