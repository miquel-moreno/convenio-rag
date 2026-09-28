import sys
import types
from collections.abc import Iterator

import pytest

from convenio_rag.adapters.db import _or_query
from convenio_rag.adapters.embeddings import FakeEmbedder, FastEmbedEmbedder
from convenio_rag.services.search import rrf


def test_rrf_rewards_items_high_in_both_rankings() -> None:
    fused = rrf({"text": [1, 2, 3], "vector": [2, 1, 4]}, k=60)

    ids = [item for item, _ in fused]
    assert ids[:2] == [1, 2]  # both appear in both lists, near the top
    assert set(ids) == {1, 2, 3, 4}
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)


def test_rrf_with_a_single_ranking_keeps_its_order() -> None:
    assert [i for i, _ in rrf({"text": [7, 3, 9]})] == [7, 3, 9]


def test_rrf_breaks_ties_by_id_and_handles_empty_rankings() -> None:
    assert [i for i, _ in rrf({"text": [5], "vector": [4]})] == [4, 5]
    assert rrf({"text": [], "vector": []}) == []


def test_or_query_keeps_only_words() -> None:
    assert _or_query("¿Cuántos días de vacaciones?") == "cuántos | días | de | vacaciones"
    assert _or_query("a & b | !c'") == "a | b | c"
    assert _or_query("¿?!") == ""


async def test_fake_embedder_makes_texts_with_shared_words_closer() -> None:
    embedder = FakeEmbedder(dim=64)
    [question, close, far] = await embedder.embed(
        ["vacaciones anuales", "días de vacaciones anuales", "horas extraordinarias"]
    )

    def cosine(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b, strict=True))

    assert cosine(question, close) > cosine(question, far)
    assert len(question) == 64


async def test_fastembed_embedder_loads_the_model_once(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded: list[tuple[str, str | None]] = []

    class StubModel:
        def __init__(self, name: str, cache_dir: str | None = None) -> None:
            loaded.append((name, cache_dir))

        def embed(self, texts: list[str]) -> Iterator[list[float]]:
            return iter([[float(len(t)), 0.5] for t in texts])

    monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=StubModel))
    embedder = FastEmbedEmbedder("some/model", cache_dir="cache")

    first = await embedder.embed(["ab", "abcd"])
    second = await embedder.embed(["x"])

    assert first == [[2.0, 0.5], [4.0, 0.5]] and second == [[1.0, 0.5]]
    assert loaded == [("some/model", "cache")]  # loaded once, reused
