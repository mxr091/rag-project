"""Read-only FastAPI entry point for the job-demand RAG assistant."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from rag_observability import append_rag_trace
from service_factory import (
    DEFAULT_RERANKER_MODEL,
    ROOT,
    create_job_service,
    default_rerank_candidate_k,
)

logger = logging.getLogger("job_rag_api")
PLAYGROUND_HTML = ROOT / "personal_live_web.html"


def dependency_failure_detail(error: RuntimeError) -> dict[str, Any]:
    failure_type = getattr(error, "failure_type", None)
    raw_usage = getattr(error, "usage", None)
    usage = (
        {
            key: value
            for key, value in raw_usage.items()
            if isinstance(key, str)
            and isinstance(value, int)
            and not isinstance(value, bool)
        }
        if isinstance(raw_usage, dict)
        else None
    )
    if failure_type == "output_truncated":
        return {
            "message": "模型输出达到本次 token 上限，已拒绝使用半截答案；请缩小问题或减少证据数量。",
            "failure_type": failure_type,
            "model_usage": usage,
        }
    return {
        "message": "RAG service is temporarily unavailable",
        "failure_type": failure_type or "dependency_failure",
        "model_usage": usage,
    }


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=3, ge=1, le=5)
    city: str | None = Field(default=None, max_length=100)
    category: str | None = Field(default=None, max_length=100)
    job_title: str | None = Field(default=None, max_length=100)
    company: str | None = Field(default=None, max_length=100)
    previous_query: str | None = Field(default=None, max_length=500)


class CitationResponse(BaseModel):
    number: int
    chunk_id: str
    source: str
    source_url: str


class RetrievedResponse(BaseModel):
    chunk_id: str
    source: str
    score: float
    city: str
    initial_rank: int | None = None
    initial_score: float | None = None
    rerank_score: float | None = None
    rerank_latency_ms: float | None = None


class AskResponse(BaseModel):
    question: str
    resolved_query: str
    answer: str
    refused: bool
    failure_type: str | None
    citations: list[CitationResponse]
    retrieved: list[RetrievedResponse]
    latency_ms: float
    answer_mode: str
    market_analysis: dict[str, Any] | None
    model_usage: dict[str, int] | None


def create_app(
    service: Any | None = None,
    *,
    strategy: str | None = None,
    generator_mode: str | None = None,
    timeout_seconds: float = 30.0,
    trace_path: str | Path | None = None,
    reranker_model_name: str | None = None,
    rerank_candidate_k: int | None = None,
) -> FastAPI:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    selected_strategy = strategy or os.getenv("RETRIEVAL_STRATEGY", "vector")
    selected_generator = generator_mode or os.getenv("ANSWER_GENERATOR", "extractive")
    selected_reranker_model = reranker_model_name or os.getenv(
        "RERANKER_MODEL", DEFAULT_RERANKER_MODEL
    )
    selected_candidate_k = rerank_candidate_k
    if selected_candidate_k is None:
        try:
            selected_candidate_k = int(
                os.getenv(
                    "RERANK_CANDIDATE_K",
                    str(default_rerank_candidate_k(selected_strategy)),
                )
            )
        except ValueError as error:
            raise ValueError("RERANK_CANDIDATE_K must be an integer") from error
    if selected_candidate_k <= 0:
        raise ValueError("rerank_candidate_k must be positive")
    selected_trace_path = Path(trace_path or ROOT / "logs" / "api_traces.jsonl")

    app = FastAPI(
        title="岗位需求 RAG 分析助手",
        version="1.0.0",
        description="只读接口：检索岗位证据，返回带来源的答案或安全拒答。",
    )
    app.state.service = service
    app.state.strategy = selected_strategy
    service_initialization_lock = Lock()

    def get_service() -> Any:
        if app.state.service is None:
            with service_initialization_lock:
                if app.state.service is None:
                    app.state.service = create_job_service(
                        strategy=selected_strategy,
                        generator_mode=selected_generator,
                        reranker_model_name=selected_reranker_model,
                        rerank_candidate_k=selected_candidate_k,
                    )
        return app.state.service

    @app.get("/", include_in_schema=False)
    def playground() -> FileResponse:
        return FileResponse(PLAYGROUND_HTML, media_type="text/html")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service_loaded": app.state.service is not None,
            "retrieval_strategy": selected_strategy,
        }

    @app.post("/ask", response_model=AskResponse)
    async def ask(payload: AskRequest) -> dict[str, Any]:
        filters = {
            key: value
            for key, value in {
                "city": payload.city,
                "category": payload.category,
                "job_title": payload.job_title,
                "company": payload.company,
            }.items()
            if value
        }

        def execute() -> Any:
            return get_service().answer(
                payload.question,
                top_k=payload.top_k,
                filters=filters,
                previous_query=payload.previous_query,
            )

        try:
            result = await asyncio.wait_for(asyncio.to_thread(execute), timeout=timeout_seconds)
        except TimeoutError as error:
            logger.warning("request timed out")
            raise HTTPException(status_code=504, detail="RAG request timed out") from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except RuntimeError as error:
            logger.exception("RAG dependency failed")
            raise HTTPException(
                status_code=503,
                detail=dependency_failure_detail(error),
            ) from error

        try:
            append_rag_trace(
                selected_trace_path,
                result,
                strategy=selected_strategy,
                top_k=payload.top_k,
                filters=filters,
            )
        except OSError:
            logger.exception("failed to persist request trace")

        return {
            "question": result.question,
            "resolved_query": result.resolved_query,
            "answer": result.answer,
            "refused": result.refused,
            "failure_type": result.failure_type,
            "citations": [
                {
                    "number": item.number,
                    "chunk_id": item.chunk_id,
                    "source": item.source,
                    "source_url": item.source_url,
                }
                for item in result.citations
            ],
            "retrieved": [
                {
                    "chunk_id": item.chunk_id,
                    "source": item.source,
                    "score": item.score,
                    "city": item.city,
                    "initial_rank": item.initial_rank,
                    "initial_score": item.initial_score,
                    "rerank_score": item.rerank_score,
                    "rerank_latency_ms": item.rerank_latency_ms,
                }
                for item in result.retrieved
            ],
            "latency_ms": result.latency_ms,
            "answer_mode": getattr(result, "answer_mode", "rag"),
            "market_analysis": getattr(result, "market_analysis", None),
            "model_usage": getattr(result, "model_usage", None),
        }

    return app


app = create_app()
