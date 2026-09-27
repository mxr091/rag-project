"""Build the runnable job-demand RAG service from local project artifacts."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from grounded_service import (
    BoundVectorRetriever,
    ExtractiveGroundedGenerator,
    GroundedRAGService,
    ModelGroundedGenerator,
)
from market_analysis import MarketAnalyzer, MarketAwareJobService
from reranker import CrossEncoderReranker, RerankingRetriever
from retrieval_pipeline import HybridRetriever, RetrievalRouter
from retriever import InMemoryRetriever
from vector_retriever import JsonVectorStore, SentenceTransformerEmbedder, load_chunks_jsonl

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"
DEFAULT_RERANK_CANDIDATE_K = 20
DEFAULT_HYBRID_RERANK_CANDIDATE_K = 30
DEFAULT_CHUNKS = ROOT / "data" / "processed" / "jobs_enriched_chunks.jsonl"
DEFAULT_INDEX = ROOT / "data" / "index" / "jobs_enriched_vectors.json"


def create_job_service(
    *,
    strategy: str = "vector",
    generator_mode: str = "extractive",
    model_name: str = DEFAULT_MODEL,
    chunks_path: str | Path = DEFAULT_CHUNKS,
    index_path: str | Path = DEFAULT_INDEX,
    embedder: Any | None = None,
    reranker_model_name: str = DEFAULT_RERANKER_MODEL,
    reranker: Any | None = None,
    rerank_candidate_k: int | None = None,
    market_analysis_enabled: bool = True,
) -> GroundedRAGService | MarketAwareJobService:
    allowed_strategies = {"lexical", "vector", "hybrid", "vector_rerank", "hybrid_rerank"}
    if strategy not in allowed_strategies:
        raise ValueError("strategy must be lexical, vector, hybrid, vector_rerank or hybrid_rerank")
    if generator_mode not in {"extractive", "model"}:
        raise ValueError("generator_mode must be extractive or model")
    selected_candidate_k = (
        default_rerank_candidate_k(strategy)
        if rerank_candidate_k is None
        else rerank_candidate_k
    )
    if selected_candidate_k <= 0:
        raise ValueError("rerank_candidate_k must be positive")
    chunks = load_chunks_jsonl(chunks_path)

    def build_rag() -> GroundedRAGService:
        return _build_rag_service(
            chunks, strategy=strategy, generator_mode=generator_mode,
            model_name=model_name, index_path=index_path, embedder=embedder,
            reranker_model_name=reranker_model_name, reranker=reranker,
            rerank_candidate_k=selected_candidate_k,
        )

    if market_analysis_enabled:
        return MarketAwareJobService(build_rag, MarketAnalyzer(chunks))
    return build_rag()


def create_job_retriever(
    *,
    strategy: str = "vector",
    model_name: str = DEFAULT_MODEL,
    chunks_path: str | Path = DEFAULT_CHUNKS,
    index_path: str | Path = DEFAULT_INDEX,
    embedder: Any | None = None,
    reranker_model_name: str = DEFAULT_RERANKER_MODEL,
    reranker: Any | None = None,
    rerank_candidate_k: int | None = None,
) -> RetrievalRouter:
    """Build only the retrieval stage for ranking evaluation and diagnostics."""
    allowed_strategies = {"lexical", "vector", "hybrid", "vector_rerank", "hybrid_rerank"}
    if strategy not in allowed_strategies:
        raise ValueError("strategy must be lexical, vector, hybrid, vector_rerank or hybrid_rerank")
    selected_candidate_k = (
        default_rerank_candidate_k(strategy)
        if rerank_candidate_k is None
        else rerank_candidate_k
    )
    if selected_candidate_k <= 0:
        raise ValueError("rerank_candidate_k must be positive")
    chunks = load_chunks_jsonl(chunks_path)
    return _build_retriever(
        chunks,
        strategy=strategy,
        model_name=model_name,
        index_path=index_path,
        embedder=embedder,
        reranker_model_name=reranker_model_name,
        reranker=reranker,
        rerank_candidate_k=selected_candidate_k,
    )


def default_rerank_candidate_k(strategy: str) -> int:
    """Return the frozen dev-set choice without changing the default strategy."""
    if strategy == "hybrid_rerank":
        return DEFAULT_HYBRID_RERANK_CANDIDATE_K
    return DEFAULT_RERANK_CANDIDATE_K


def _build_rag_service(
    chunks: Any,
    *,
    strategy: str,
    generator_mode: str,
    model_name: str,
    index_path: str | Path,
    embedder: Any | None,
    reranker_model_name: str,
    reranker: Any | None,
    rerank_candidate_k: int,
) -> GroundedRAGService:
    router = _build_retriever(
        chunks,
        strategy=strategy,
        model_name=model_name,
        index_path=index_path,
        embedder=embedder,
        reranker_model_name=reranker_model_name,
        reranker=reranker,
        rerank_candidate_k=rerank_candidate_k,
    )
    generator = _create_generator(generator_mode)
    return GroundedRAGService(router, generator)


def _build_retriever(
    chunks: Any,
    *,
    strategy: str,
    model_name: str,
    index_path: str | Path,
    embedder: Any | None,
    reranker_model_name: str,
    reranker: Any | None,
    rerank_candidate_k: int,
) -> RetrievalRouter:
    lexical = InMemoryRetriever(chunks)
    retrievers: dict[str, Any] = {"lexical": lexical}

    if strategy in {"vector", "hybrid", "vector_rerank", "hybrid_rerank"}:
        vector_store = JsonVectorStore(index_path)
        if vector_store.count == 0:
            raise RuntimeError("vector index is empty; run job_vector_cli.py build first")
        bound_vector = BoundVectorRetriever(
            vector_store,
            embedder or SentenceTransformerEmbedder(model_name),
        )
        retrievers["vector"] = bound_vector
        hybrid = HybridRetriever(lexical, bound_vector)
        retrievers["hybrid"] = hybrid
        if strategy in {"vector_rerank", "hybrid_rerank"}:
            selected_reranker = reranker or CrossEncoderReranker(reranker_model_name)
            base_retriever = bound_vector if strategy == "vector_rerank" else hybrid
            retrievers[strategy] = RerankingRetriever(
                base_retriever,
                selected_reranker,
                candidate_k=rerank_candidate_k,
            )

    router = RetrievalRouter(retrievers, strategy=strategy)
    return router


def _create_generator(mode: str) -> Any:
    if mode == "extractive":
        return ExtractiveGroundedGenerator()
    if mode != "model":
        raise ValueError("generator_mode must be extractive or model")

    api_key = os.getenv("MODEL_API_KEY", "")
    endpoint = os.getenv("MODEL_ENDPOINT", "")
    model_name = os.getenv("MODEL_NAME", "")
    if not api_key or not endpoint or not model_name:
        raise RuntimeError("model mode requires MODEL_API_KEY, MODEL_ENDPOINT and MODEL_NAME")
    try:
        max_retries = int(os.getenv("MODEL_MAX_RETRIES", "0"))
        retry_backoff_seconds = float(os.getenv("MODEL_RETRY_BACKOFF_SECONDS", "0.5"))
        raw_max_tokens = os.getenv("MODEL_MAX_TOKENS", "").strip()
        max_tokens = int(raw_max_tokens) if raw_max_tokens else None
    except ValueError as error:
        raise RuntimeError(
            "MODEL_MAX_RETRIES, MODEL_RETRY_BACKOFF_SECONDS and MODEL_MAX_TOKENS "
            "must be numeric"
        ) from error
    if max_retries < 0 or retry_backoff_seconds < 0 or (
        max_tokens is not None and max_tokens <= 0
    ):
        raise RuntimeError(
            "MODEL_MAX_RETRIES and MODEL_RETRY_BACKOFF_SECONDS must not be negative; "
            "MODEL_MAX_TOKENS must be positive"
        )

    adapter_path = ROOT.parents[1] / "experiments" / "02-openai-compatible-adapter"
    sys.path.insert(0, str(adapter_path))
    from openai_compatible import OpenAICompatibleModel

    model = OpenAICompatibleModel(
        endpoint,
        api_key,
        model_name,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        max_tokens=max_tokens,
    )
    return ModelGroundedGenerator(model)

