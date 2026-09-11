"""Unified CLI for the runnable job-demand RAG pipeline."""

from __future__ import annotations

import argparse
import json

from rag_observability import append_rag_trace
from service_factory import (
    DEFAULT_CHUNKS,
    DEFAULT_INDEX,
    DEFAULT_MODEL,
    DEFAULT_RERANKER_MODEL,
    ROOT,
    create_job_service,
    default_rerank_candidate_k,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="岗位需求 RAG 分析助手")
    parser.add_argument("question", help="例如：深圳 AI Agent 岗位需要哪些技能？")
    parser.add_argument(
        "--strategy",
        choices=("lexical", "vector", "hybrid", "vector_rerank", "hybrid_rerank"),
        default="vector",
    )
    parser.add_argument("--generator", choices=("extractive", "model"), default="extractive")
    parser.add_argument("--top-k", type=int, choices=range(1, 6), default=3)
    parser.add_argument("--city")
    parser.add_argument("--category")
    parser.add_argument("--job-title")
    parser.add_argument("--company")
    parser.add_argument("--previous-query")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=None,
        help="default: vector_rerank=20, hybrid_rerank=30 (selected on the dev set)",
    )
    parser.add_argument("--chunks-path", default=str(DEFAULT_CHUNKS))
    parser.add_argument("--index-path", default=str(DEFAULT_INDEX))
    parser.add_argument("--trace-path", default=str(ROOT / "logs" / "cli_traces.jsonl"))
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    filters = {
        key: value
        for key, value in {
            "city": args.city,
            "category": args.category,
            "job_title": args.job_title,
            "company": args.company,
        }.items()
        if value
    }
    service = create_job_service(
        strategy=args.strategy,
        generator_mode=args.generator,
        model_name=args.model,
        chunks_path=args.chunks_path,
        index_path=args.index_path,
        reranker_model_name=args.reranker_model,
        rerank_candidate_k=(
            args.candidate_k
            if args.candidate_k is not None
            else default_rerank_candidate_k(args.strategy)
        ),
    )
    result = service.answer(
        args.question,
        top_k=args.top_k,
        filters=filters,
        previous_query=args.previous_query,
    )
    append_rag_trace(
        args.trace_path,
        result,
        strategy=args.strategy,
        top_k=args.top_k,
        filters=filters,
    )
    payload = {
        "question": result.question,
        "resolved_query": result.resolved_query,
        "answer": result.answer,
        "refused": result.refused,
        "failure_type": result.failure_type,
        "citations": [item.__dict__ for item in result.citations],
        "retrieved": [item.__dict__ for item in result.retrieved],
        "latency_ms": result.latency_ms,
        "answer_mode": result.answer_mode,
        "market_analysis": result.market_analysis,
        "model_usage": result.model_usage,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print(result.answer)
    if result.citations:
        print("\n来源：")
        for citation in result.citations:
            suffix = f" {citation.source_url}" if citation.source_url else ""
            print(f"[{citation.number}] {citation.source}{suffix}")
    if result.refused:
        print(f"\n[安全拒答：{result.failure_type}]")
    print(
        f"\n[mode={result.answer_mode} strategy={args.strategy} "
        f"top_k={args.top_k} latency_ms={result.latency_ms:.1f}]"
    )


if __name__ == "__main__":
    main()
