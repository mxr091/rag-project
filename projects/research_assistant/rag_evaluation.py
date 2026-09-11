"""Reproducible evaluation for retrieval, citation, refusal, and latency."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from grounded_service import GroundedRAGService


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    question: str
    expected_sources: tuple[str, ...] = ()
    expect_refusal: bool = False
    filters: dict[str, str] | None = None


def load_evaluation_cases(path: str | Path) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            cases.append(
                EvaluationCase(
                    case_id=str(item["id"]),
                    question=str(item["question"]),
                    expected_sources=tuple(item.get("expected_sources", [])),
                    expect_refusal=bool(item.get("expect_refusal", False)),
                    filters=item.get("filters"),
                )
            )
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid evaluation case at line {line_number}") from error
    if not cases:
        raise ValueError("evaluation set must not be empty")
    return cases


def evaluate_service(
    service: GroundedRAGService,
    cases: Sequence[EvaluationCase],
    *,
    top_k: int = 3,
) -> dict[str, Any]:
    if not cases:
        raise ValueError("evaluation cases must not be empty")
    details: list[dict[str, Any]] = []
    source_hits = 0
    citation_correct = 0
    refusal_correct = 0
    failure_types: Counter[str] = Counter()
    total_latency_ms = 0.0

    for case in cases:
        answer = service.answer(case.question, top_k=top_k, filters=case.filters)
        retrieved_sources = {item.source for item in answer.retrieved}
        expected_sources = set(case.expected_sources)
        source_hit = not expected_sources or bool(retrieved_sources.intersection(expected_sources))
        citations_ok = answer.refused or (
            bool(answer.citations)
            and all(citation.source in retrieved_sources for citation in answer.citations)
        )
        refusal_ok = answer.refused == case.expect_refusal
        source_hits += int(source_hit)
        citation_correct += int(citations_ok)
        refusal_correct += int(refusal_ok)
        total_latency_ms += answer.latency_ms
        if answer.failure_type:
            failure_types[answer.failure_type] += 1
        details.append(
            {
                "id": case.case_id,
                "source_hit": source_hit,
                "citation_correct": citations_ok,
                "refusal_correct": refusal_ok,
                "refused": answer.refused,
                "failure_type": answer.failure_type,
                "retrieved_sources": sorted(retrieved_sources),
                "latency_ms": round(answer.latency_ms, 3),
            }
        )

    count = len(cases)
    return {
        "questions": count,
        "top_k": top_k,
        "source_hit_rate": source_hits / count,
        "citation_accuracy": citation_correct / count,
        "refusal_accuracy": refusal_correct / count,
        "average_latency_ms": total_latency_ms / count,
        "failure_types": dict(failure_types),
        "details": details,
    }


def save_evaluation_report(report: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return target
