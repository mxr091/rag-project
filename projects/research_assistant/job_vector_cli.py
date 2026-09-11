"""Build and query the U6 embedding index for job-demand chunks."""

from __future__ import annotations

import argparse

from vector_retriever import (
    JsonVectorStore,
    SentenceTransformerEmbedder,
    append_retrieval_record,
    load_chunks_jsonl,
)

DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Job embedding retrieval")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--index-path", default="data/index/jobs_enriched_vectors.json")
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="rebuild the index from U5 JSONL chunks")
    build.add_argument("--chunks-path", default="data/processed/jobs_enriched_chunks.jsonl")

    query = commands.add_parser("query", help="query the existing vector index")
    query.add_argument("question")
    query.add_argument("--top-k", type=int, default=3)
    query.add_argument("--city")
    query.add_argument("--category")
    query.add_argument("--job-title")
    query.add_argument("--observation", help="manual observation to append to a U6 record")
    query.add_argument("--record-path", default="logs/retrieval_records.jsonl")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    embedder = SentenceTransformerEmbedder(args.model)
    store = JsonVectorStore(args.index_path)

    if args.command == "build":
        chunks = load_chunks_jsonl(args.chunks_path)
        count = store.rebuild(chunks, embedder)
        print(f"已重建向量索引：{count} 个 Chunk -> {args.index_path}")
        return

    filters = {
        field: value
        for field, value in {
            "city": args.city,
            "category": args.category,
            "job_title": args.job_title,
        }.items()
        if value
    }
    results = store.search(args.question, embedder=embedder, top_k=args.top_k, filters=filters)
    print(f"命中 {len(results)} 条 | 过滤条件：{filters or '无'}")
    for number, result in enumerate(results, start=1):
        print(f"[{number}] score={result.score:.4f} source={result.chunk.source}")
        print(result.chunk.text)
        print()
    if args.observation:
        append_retrieval_record(
            args.record_path,
            question=args.question,
            filters=filters,
            top_k=args.top_k,
            results=results,
            observation=args.observation,
        )
        print(f"已追加检索记录：{args.record_path}")


if __name__ == "__main__":
    main()


