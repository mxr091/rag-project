"""Auditable ranking metrics for the manually annotated ReRank question set."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol


class RankedResultLike(Protocol):
    chunk: Any


class RetrieverLike(Protocol):
    def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        filters: Mapping[str, str] | None = None,
    ) -> Sequence[RankedResultLike]: ...


@dataclass(frozen=True)
class RankingCase:
    case_id: str
    split: str
    question: str
    rubric: str
    qrels: Mapping[str, int]


def load_ranking_cases(path: str | Path) -> list[RankingCase]:
    """Load and validate frozen manual relevance judgments from JSONL."""
    cases: list[RankingCase] = []
    seen_case_ids: set[str] = set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            case_id = str(item["id"]).strip()
            split = str(item["split"]).strip()
            question = str(item["question"]).strip()
            rubric = str(item["rubric"]).strip()
            judgments = item["qrels"]
            if not isinstance(judgments, list):
                raise TypeError("qrels must be a list")
            qrels: dict[str, int] = {}
            for judgment in judgments:
                chunk_id = str(judgment["chunk_id"]).strip()
                relevance = int(judgment["relevance"])
                if not chunk_id or chunk_id in qrels or relevance not in {1, 2}:
                    raise ValueError("invalid qrel")
                qrels[chunk_id] = relevance
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid ranking case at line {line_number}") from error
        if not case_id or case_id in seen_case_ids:
            raise ValueError(f"duplicate or empty case id at line {line_number}")
        if split not in {"dev", "holdout"} or not question or not rubric or not qrels:
            raise ValueError(f"incomplete ranking case at line {line_number}")
        seen_case_ids.add(case_id)
        cases.append(RankingCase(case_id, split, question, rubric, qrels))
    if not cases:
        raise ValueError("ranking evaluation set must not be empty")
    if {case.split for case in cases} != {"dev", "holdout"}:
        raise ValueError("ranking evaluation set must contain dev and holdout cases")
    return cases


def evaluate_rankings(
    retriever: RetrieverLike,
    cases: Sequence[RankingCase],
    *,
    top_k: int = 3,
) -> dict[str, Any]:
    """Evaluate one retriever with hit, precision, recall, MRR, and nDCG."""
    if not cases:
        raise ValueError("ranking cases must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    details: list[dict[str, Any]] = []
    totals = {"hit_rate": 0.0, "precision": 0.0, "recall": 0.0, "mrr": 0.0, "ndcg": 0.0}
    total_latency_ms = 0.0
    for case in cases:
        started = perf_counter()
        results = tuple(retriever.search(case.question, top_k=top_k))
        latency_ms = (perf_counter() - started) * 1000
        total_latency_ms += latency_ms
        ranked_ids = [str(result.chunk.chunk_id) for result in results]
        metrics = ranking_metrics(ranked_ids, case.qrels, top_k=top_k)
        for key in totals:
            totals[key] += metrics[key]
        details.append(
            {
                "id": case.case_id,
                "question": case.question,
                "ranked_chunk_ids": ranked_ids,
                "relevant_chunk_ids": sorted(case.qrels),
                "relevant_ranks": [
                    rank for rank, chunk_id in enumerate(ranked_ids, start=1) if chunk_id in case.qrels
                ],
                "latency_ms": round(latency_ms, 3),
                **{key: round(value, 6) for key, value in metrics.items()},
            }
        )

    count = len(cases)
    return {
        "questions": count,
        "top_k": top_k,
        **{f"macro_{key}_at_{top_k}": totals[key] / count for key in totals},
        "average_latency_ms": total_latency_ms / count,
        "details": details,
    }


def ranking_metrics(
    ranked_chunk_ids: Sequence[str],
    qrels: Mapping[str, int],
    *,
    top_k: int,
) -> dict[str, float]:
    """Calculate standard top-k ranking metrics for one query."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not qrels:
        raise ValueError("qrels must not be empty")
    top_ids = list(ranked_chunk_ids[:top_k])
    relevant_ranks = [rank for rank, chunk_id in enumerate(top_ids, start=1) if chunk_id in qrels]
    relevant_count = len(relevant_ranks)
    dcg = sum(
        (2 ** qrels.get(chunk_id, 0) - 1) / math.log2(rank + 1)
        for rank, chunk_id in enumerate(top_ids, start=1)
    )
    ideal_relevances = sorted(qrels.values(), reverse=True)[:top_k]
    idcg = sum(
        (2 ** relevance - 1) / math.log2(rank + 1)
        for rank, relevance in enumerate(ideal_relevances, start=1)
    )
    return {
        "hit_rate": float(bool(relevant_ranks)),
        "precision": relevant_count / top_k,
        "recall": relevant_count / len(qrels),
        "mrr": 0.0 if not relevant_ranks else 1.0 / relevant_ranks[0],
        "ndcg": 0.0 if idcg == 0 else dcg / idcg,
    }


def split_cases(cases: Sequence[RankingCase], split: str) -> list[RankingCase]:
    if split not in {"dev", "holdout"}:
        raise ValueError("split must be dev or holdout")
    selected = [case for case in cases if case.split == split]
    if not selected:
        raise ValueError(f"no ranking cases found for split: {split}")
    return selected


def select_candidate_k(
    reports: Mapping[int, Mapping[str, Any]],
    *,
    top_k: int = 3,
) -> int:
    """Select on dev metrics only; prefer the smaller pool on an exact tie."""
    if not reports:
        raise ValueError("candidate reports must not be empty")
    metric_names = (
        f"macro_ndcg_at_{top_k}",
        f"macro_recall_at_{top_k}",
        f"macro_mrr_at_{top_k}",
    )
    for candidate_k, report in reports.items():
        if candidate_k <= 0 or any(name not in report for name in metric_names):
            raise ValueError("candidate reports are incomplete")
    return max(
        reports,
        key=lambda candidate_k: (
            *(float(reports[candidate_k][name]) for name in metric_names),
            -candidate_k,
        ),
    )


def analyze_candidate_ceiling(
    candidate_report: Mapping[str, Any],
    rerank_report: Mapping[str, Any],
    *,
    top_k: int,
) -> dict[str, Any]:
    """Explain whether each rerank result is limited by recall or ordering.

    The candidate pool defines the best result a second-stage reranker could
    possibly achieve. This diagnostic separates missing candidates from a
    reranker that did not promote all reachable relevant documents.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    candidate_details = candidate_report.get("details")
    rerank_details = rerank_report.get("details")
    if not isinstance(candidate_details, Sequence) or not isinstance(rerank_details, Sequence):
        raise ValueError("candidate and rerank reports must contain details")
    candidate_by_id = {
        str(item.get("id")): item
        for item in candidate_details
        if isinstance(item, Mapping) and item.get("id") is not None
    }
    rerank_by_id = {
        str(item.get("id")): item
        for item in rerank_details
        if isinstance(item, Mapping) and item.get("id") is not None
    }
    if not candidate_by_id or candidate_by_id.keys() != rerank_by_id.keys():
        raise ValueError("candidate and rerank reports must contain the same case ids")

    classifications: Counter[str] = Counter()
    cases: list[dict[str, Any]] = []
    total_ceiling_recall = 0.0
    total_rerank_recall = 0.0
    for case_id, candidate in candidate_by_id.items():
        rerank = rerank_by_id[case_id]
        relevant_ids = set(map(str, candidate.get("relevant_chunk_ids", [])))
        rerank_relevant_ids = set(map(str, rerank.get("relevant_chunk_ids", [])))
        if not relevant_ids or relevant_ids != rerank_relevant_ids:
            raise ValueError(f"inconsistent or empty qrels for case: {case_id}")

        candidate_ids = list(map(str, candidate.get("ranked_chunk_ids", [])))
        final_ids = list(map(str, rerank.get("ranked_chunk_ids", [])))[:top_k]
        reachable_ids = relevant_ids.intersection(candidate_ids)
        unreachable_ids = relevant_ids.difference(candidate_ids)
        final_relevant_ids = relevant_ids.intersection(final_ids)
        target_relevant_count = min(len(relevant_ids), top_k)
        candidate_ceiling_count = min(len(reachable_ids), top_k)
        candidate_ceiling_recall = candidate_ceiling_count / len(relevant_ids)
        rerank_recall = len(final_relevant_ids) / len(relevant_ids)

        if not reachable_ids:
            classification = "candidate_miss"
        elif len(final_relevant_ids) < candidate_ceiling_count:
            classification = "rerank_gap"
        elif candidate_ceiling_count < target_relevant_count:
            classification = "candidate_limited"
        else:
            classification = "ceiling_reached"
        classifications[classification] += 1
        total_ceiling_recall += candidate_ceiling_recall
        total_rerank_recall += rerank_recall
        cases.append(
            {
                "id": case_id,
                "question": str(candidate.get("question", rerank.get("question", ""))),
                "classification": classification,
                "relevant_count": len(relevant_ids),
                "reachable_relevant_count": len(reachable_ids),
                "reachable_relevant_ids": sorted(reachable_ids),
                "unreachable_relevant_ids": sorted(unreachable_ids),
                "candidate_relevant_ranks": [
                    rank
                    for rank, chunk_id in enumerate(candidate_ids, start=1)
                    if chunk_id in relevant_ids
                ],
                "rerank_relevant_ranks": [
                    rank
                    for rank, chunk_id in enumerate(final_ids, start=1)
                    if chunk_id in relevant_ids
                ],
                f"candidate_ceiling_recall_at_{top_k}": candidate_ceiling_recall,
                f"rerank_recall_at_{top_k}": rerank_recall,
                f"gap_to_candidate_ceiling_at_{top_k}": (
                    candidate_ceiling_recall - rerank_recall
                ),
            }
        )

    count = len(cases)
    return {
        "questions": count,
        "top_k": top_k,
        "classification_counts": dict(sorted(classifications.items())),
        f"macro_candidate_ceiling_recall_at_{top_k}": total_ceiling_recall / count,
        f"macro_rerank_recall_at_{top_k}": total_rerank_recall / count,
        f"macro_gap_to_candidate_ceiling_at_{top_k}": (
            total_ceiling_recall - total_rerank_recall
        ) / count,
        "cases": cases,
    }


def save_ranking_report(report: Mapping[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return target
