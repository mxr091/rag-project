"""Optional second-stage reranking for vector and hybrid retrieval candidates."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

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


class RerankerLike(Protocol):
    def score(self, query: str, documents: Sequence[str]) -> Sequence[float]: ...


@dataclass(frozen=True)
class RerankedSearchResult:
    chunk: Chunk
    score: float
    initial_rank: int
    initial_score: float
    rerank_score: float
    rerank_latency_ms: float


class CrossEncoderReranker:
    """Lazy Sentence Transformers CrossEncoder adapter.

    Construction does not download or load a model. The first real ``score``
    call loads it, which keeps lexical/vector-only paths and offline tests cheap.
    """

    def __init__(self, model_name: str, *, model: Any | None = None) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("reranker model_name must be a non-empty string")
        self.model_name = model_name.strip()
        self._model = model

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("reranker query must be a non-empty string")
        materialized = list(documents)
        if not materialized:
            return []
        if any(not isinstance(document, str) or not document.strip() for document in materialized):
            raise ValueError("reranker documents must contain non-empty strings")

        pairs = [(query, document) for document in materialized]
        raw_scores = self._get_model().predict(pairs, show_progress_bar=False)
        scores = [float(value) for value in raw_scores]
        _validate_scores(scores, expected_count=len(materialized))
        return scores

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as error:
                raise RuntimeError(
                    "Install requirements-embedding.txt before using CrossEncoder reranking."
                ) from error
            self._model = CrossEncoder(self.model_name)
        return self._model


class RerankingRetriever:
    """Retrieve a wider candidate pool, then rerank it to the requested top_k."""

    def __init__(
        self,
        base_retriever: RetrieverLike,
        reranker: RerankerLike,
        *,
        candidate_k: int = 20,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")
        self._base_retriever = base_retriever
        self._reranker = reranker
        self.candidate_k = candidate_k

    def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        filters: Mapping[str, str] | None = None,
    ) -> list[RerankedSearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        candidate_count = max(self.candidate_k, top_k)
        candidates = tuple(
            self._base_retriever.search(
                query,
                top_k=candidate_count,
                filters=filters,
            )
        )
        if not candidates:
            return []

        started = perf_counter()
        scores = [float(value) for value in self._reranker.score(
            query,
            [result.chunk.text for result in candidates],
        )]
        latency_ms = (perf_counter() - started) * 1000
        _validate_scores(scores, expected_count=len(candidates))

        reranked = [
            RerankedSearchResult(
                chunk=result.chunk,
                score=rerank_score,
                initial_rank=initial_rank,
                initial_score=float(result.score),
                rerank_score=rerank_score,
                rerank_latency_ms=latency_ms,
            )
            for initial_rank, (result, rerank_score) in enumerate(
                zip(candidates, scores, strict=True),
                start=1,
            )
        ]
        reranked.sort(
            key=lambda item: (
                -item.rerank_score,
                item.initial_rank,
                item.chunk.chunk_id,
            )
        )
        return reranked[:top_k]


def _validate_scores(scores: Sequence[float], *, expected_count: int) -> None:
    if len(scores) != expected_count:
        raise RuntimeError(
            f"reranker returned {len(scores)} scores for {expected_count} candidates"
        )
    if not all(math.isfinite(float(score)) for score in scores):
        raise RuntimeError("reranker returned a non-finite score")
