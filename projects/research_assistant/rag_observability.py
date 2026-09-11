"""JSONL observability for one grounded RAG run."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from grounded_service import GroundedAnswer


_TRACE_WRITE_LOCK = Lock()


def build_rag_trace(
    result: GroundedAnswer,
    *,
    strategy: str,
    top_k: int,
    filters: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": result.question,
        "resolved_query": result.resolved_query,
        "strategy": strategy,
        "answer_mode": result.answer_mode,
        "top_k": top_k,
        "filters": dict(filters or {}),
        "retrieved": [
            {
                "chunk_id": item.chunk_id,
                "source": item.source,
                "score": round(item.score, 6),
                "city": item.city,
                "initial_rank": item.initial_rank,
                "initial_score": (
                    round(item.initial_score, 6) if item.initial_score is not None else None
                ),
                "rerank_score": (
                    round(item.rerank_score, 6) if item.rerank_score is not None else None
                ),
                "rerank_latency_ms": (
                    round(item.rerank_latency_ms, 3)
                    if item.rerank_latency_ms is not None
                    else None
                ),
            }
            for item in result.retrieved
        ],
        "answer": result.answer,
        "citations": [
            {
                "number": item.number,
                "chunk_id": item.chunk_id,
                "source": item.source,
                "source_url": item.source_url,
            }
            for item in result.citations
        ],
        "refused": result.refused,
        "failure_type": result.failure_type,
        "market_analysis": result.market_analysis,
        "model_usage": dict(result.model_usage) if result.model_usage is not None else None,
        "latency_ms": round(result.latency_ms, 3),
    }


def append_rag_trace(
    path: str | Path,
    result: GroundedAnswer,
    *,
    strategy: str,
    top_k: int,
    filters: Mapping[str, str] | None = None,
) -> Path:
    target = Path(path)
    line = json.dumps(
        build_rag_trace(result, strategy=strategy, top_k=top_k, filters=filters),
        ensure_ascii=False,
    )
    with _TRACE_WRITE_LOCK:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as file:
            file.write(line + "\n")
    return target
