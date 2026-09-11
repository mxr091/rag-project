"""Grounded RAG answer service with refusal and citation validation."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

from prompt import build_grounded_prompt
from security import is_safe_evidence

REFUSAL_TEXT = "资料中没有足够证据。"
_ALLOWED_FILTERS = frozenset({"city", "category", "job_title", "company"})
_CITATION_PATTERN = re.compile(r"\[(\d+)]")
_CITY_PATTERN = re.compile(r"(上海|深圳|广州)")


class _ChunkLike(Protocol):
    chunk_id: str
    source: str
    text: str
    metadata: Mapping[str, str]


class SearchResultLike(Protocol):
    chunk: _ChunkLike
    score: float


class Retriever(Protocol):
    def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        filters: Mapping[str, str] | None = None,
    ) -> Sequence[SearchResultLike]: ...


class AnswerGenerator(Protocol):
    def generate(self, prompt: str, results: Sequence[SearchResultLike]) -> str: ...


@dataclass(frozen=True)
class Citation:
    number: int
    chunk_id: str
    source: str
    source_url: str = ""


@dataclass(frozen=True)
class RetrievedEvidence:
    chunk_id: str
    source: str
    score: float
    city: str = ""
    initial_rank: int | None = None
    initial_score: float | None = None
    rerank_score: float | None = None
    rerank_latency_ms: float | None = None


@dataclass(frozen=True)
class GroundedAnswer:
    question: str
    resolved_query: str
    answer: str
    refused: bool
    citations: tuple[Citation, ...] = ()
    retrieved: tuple[RetrievedEvidence, ...] = ()
    failure_type: str | None = None
    latency_ms: float = 0.0
    answer_mode: str = "rag"
    market_analysis: Mapping[str, Any] | None = None
    model_usage: Mapping[str, int] | None = None


class ExtractiveGroundedGenerator:
    """Deterministic offline generator used for local smoke tests and fallback."""

    def generate(self, prompt: str, results: Sequence[SearchResultLike]) -> str:
        del prompt
        statements: list[str] = []
        for number, result in enumerate(results, start=1):
            text = " ".join(result.chunk.text.split())
            if text:
                statements.append(f"{text[:260]} [{number}]")
        return "\n".join(statements) if statements else REFUSAL_TEXT


class ModelGroundedGenerator:
    """Adapter for a model exposing respond(messages, tools)."""

    def __init__(self, model: Any) -> None:
        self._model = model
        self.last_usage: Mapping[str, int] | None = None

    def generate(self, prompt: str, results: Sequence[SearchResultLike]) -> str:
        del results
        response = self._model.respond([{"role": "user", "content": prompt}], [])
        self.last_usage = getattr(response, "usage", None)
        if getattr(response, "tool_call", None) is not None:
            raise RuntimeError("grounded answer model must return text, not a tool call")
        content = getattr(response, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("grounded answer model returned no text")
        return content.strip()


class BoundVectorRetriever:
    """Bind an embedder to JsonVectorStore so the service sees one search interface."""

    def __init__(self, vector_store: Any, embedder: Any) -> None:
        self._vector_store = vector_store
        self._embedder = embedder

    def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        filters: Mapping[str, str] | None = None,
    ) -> Sequence[SearchResultLike]:
        return self._vector_store.search(
            query,
            embedder=self._embedder,
            top_k=top_k,
            filters=filters,
        )


class GroundedRAGService:
    def __init__(self, retriever: Retriever, generator: AnswerGenerator | None = None,
                 *, prompt_builder=build_grounded_prompt) -> None:
        self._retriever = retriever
        self._generator = generator or ExtractiveGroundedGenerator()
        self._prompt_builder = prompt_builder

    def answer(
        self,
        question: str,
        *,
        top_k: int = 3,
        filters: Mapping[str, str] | None = None,
        previous_query: str | None = None,
    ) -> GroundedAnswer:
        started = perf_counter()
        cleaned_question = validate_question(question)
        if not 1 <= top_k <= 5:
            raise ValueError("top_k must be between 1 and 5")
        normalized_filters = validate_filters(filters)
        resolved_query = rewrite_follow_up(cleaned_question, previous_query)
        raw_results = tuple(
            self._retriever.search(
                resolved_query,
                top_k=top_k,
                filters=normalized_filters,
            )
        )
        results = tuple(result for result in raw_results if is_safe_evidence(result.chunk.text))
        retrieved = tuple(
            RetrievedEvidence(
                chunk_id=result.chunk.chunk_id,
                source=result.chunk.source,
                score=float(result.score),
                city=result.chunk.metadata.get("city", ""),
                initial_rank=getattr(result, "initial_rank", None),
                initial_score=getattr(result, "initial_score", None),
                rerank_score=getattr(result, "rerank_score", None),
                rerank_latency_ms=getattr(result, "rerank_latency_ms", None),
            )
            for result in raw_results
        )
        if raw_results and not results:
            return GroundedAnswer(
                question=cleaned_question,
                resolved_query=resolved_query,
                answer=REFUSAL_TEXT,
                refused=True,
                retrieved=retrieved,
                failure_type="prompt_injection",
                latency_ms=(perf_counter() - started) * 1000,
            )
        if not results:
            return GroundedAnswer(
                question=cleaned_question,
                resolved_query=resolved_query,
                answer=REFUSAL_TEXT,
                refused=True,
                retrieved=retrieved,
                failure_type="retrieval_empty",
                latency_ms=(perf_counter() - started) * 1000,
            )

        prompt = self._prompt_builder(resolved_query, results)
        generated = self._generator.generate(prompt, results).strip()
        model_usage = getattr(self._generator, "last_usage", None)
        if not generated or REFUSAL_TEXT.rstrip("。") in generated:
            return GroundedAnswer(
                question=cleaned_question,
                resolved_query=resolved_query,
                answer=REFUSAL_TEXT,
                refused=True,
                retrieved=retrieved,
                failure_type="evidence_insufficient",
                latency_ms=(perf_counter() - started) * 1000,
                model_usage=model_usage,
            )

        numbers = sorted({int(value) for value in _CITATION_PATTERN.findall(generated)})
        if not numbers or any(number < 1 or number > len(results) for number in numbers):
            return GroundedAnswer(
                question=cleaned_question,
                resolved_query=resolved_query,
                answer=REFUSAL_TEXT,
                refused=True,
                retrieved=retrieved,
                failure_type="citation_validation",
                latency_ms=(perf_counter() - started) * 1000,
                model_usage=model_usage,
            )

        citations = tuple(
            Citation(
                number=number,
                chunk_id=results[number - 1].chunk.chunk_id,
                source=results[number - 1].chunk.source,
                source_url=results[number - 1].chunk.metadata.get("source_url", ""),
            )
            for number in numbers
        )
        return GroundedAnswer(
            question=cleaned_question,
            resolved_query=resolved_query,
            answer=generated,
            refused=False,
            citations=citations,
            retrieved=retrieved,
            latency_ms=(perf_counter() - started) * 1000,
            model_usage=model_usage,
        )


def rewrite_follow_up(question: str, previous_query: str | None) -> str:
    """Resolve the small but common follow-up: '那广州呢？'."""
    if not previous_query:
        return question
    new_city = _CITY_PATTERN.search(question)
    old_city = _CITY_PATTERN.search(previous_query)
    compact = re.sub(r"[\s？?。！!]", "", question)
    looks_like_follow_up = compact.startswith("那") or compact.endswith("呢") or len(compact) <= 6
    if new_city and old_city and looks_like_follow_up:
        return _CITY_PATTERN.sub(new_city.group(1), previous_query, count=1)
    return question


def validate_question(question: str) -> str:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    cleaned = question.strip()
    if len(cleaned) > 500:
        raise ValueError("question must not exceed 500 characters")
    if any(ord(character) < 32 and character not in "\t\n\r" for character in cleaned):
        raise ValueError("question contains unsupported control characters")
    return cleaned


def validate_filters(filters: Mapping[str, str] | None) -> dict[str, str]:
    if filters is None:
        return {}
    normalized: dict[str, str] = {}
    for key, value in filters.items():
        if key not in _ALLOWED_FILTERS:
            raise ValueError(f"unsupported metadata filter: {key}")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"metadata filter {key} must be a non-empty string")
        normalized[key] = value.strip()
    return normalized

