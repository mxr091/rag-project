"""Configurable lexical, vector, and hybrid retrieval."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from document import Chunk


class SearchResultLike(Protocol):
    chunk: Chunk
    score: float


class RetrieverLike(Protocol):
    def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        filters: Mapping[str, str] | None = None,
    ) -> Sequence[SearchResultLike]: ...


@dataclass(frozen=True)
class RankedSearchResult:
    chunk: Chunk
    score: float


class HybridRetriever:
    """Combine BM25 and vector rankings with reciprocal-rank fusion (RRF)."""

    def __init__(self, lexical: RetrieverLike, semantic: RetrieverLike, *, rrf_k: int = 60) -> None:
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        self._lexical = lexical
        self._semantic = semantic
        self._rrf_k = rrf_k

    def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        filters: Mapping[str, str] | None = None,
    ) -> list[RankedSearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        candidate_count = max(top_k * 3, top_k)
        ranked_lists = (
            self._lexical.search(query, top_k=candidate_count, filters=filters),
            self._semantic.search(query, top_k=candidate_count, filters=filters),
        )
        fused_scores: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}
        for ranked in ranked_lists:
            for rank, result in enumerate(ranked, start=1):
                chunk_id = result.chunk.chunk_id
                chunks[chunk_id] = result.chunk
                fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (self._rrf_k + rank)
        results = [RankedSearchResult(chunks[key], score) for key, score in fused_scores.items()]
        results.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
        return results[:top_k]


class RetrievalRouter:
    """Whitelist retrieval strategies so callers cannot select arbitrary code."""

    def __init__(self, retrievers: Mapping[str, RetrieverLike], *, strategy: str = "hybrid") -> None:
        if strategy not in retrievers:
            raise ValueError(f"unknown retrieval strategy: {strategy}")
        self._retrievers = dict(retrievers)
        self.strategy = strategy

    @property
    def allowed_strategies(self) -> tuple[str, ...]:
        return tuple(sorted(self._retrievers))

    def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        filters: Mapping[str, str] | None = None,
    ) -> Sequence[SearchResultLike]:
        return self._retrievers[self.strategy].search(query, top_k=top_k, filters=filters)
