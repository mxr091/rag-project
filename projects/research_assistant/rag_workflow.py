"""Framework-free workflow state, bounded retry, and JSON checkpoints."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from grounded_service import GroundedAnswer, GroundedRAGService, REFUSAL_TEXT

_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9-]{1,64}$")


@dataclass(frozen=True)
class WorkflowPlan:
    action: str
    question: str
    top_k: int
    filters: dict[str, str]
    requires_confirmation: bool = False


@dataclass
class WorkflowResult:
    run_id: str
    status: str
    plan: WorkflowPlan
    attempts: int = 0
    answer: GroundedAnswer | None = None
    events: list[dict[str, Any]] = field(default_factory=list)


class CheckpointStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, result: WorkflowResult) -> Path:
        target = self._path(result.run_id)
        payload = asdict(result)
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(target)
        return target

    def load(self, run_id: str) -> dict[str, Any]:
        target = self._path(run_id)
        if not target.exists():
            raise FileNotFoundError(f"checkpoint not found: {run_id}")
        data = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("invalid checkpoint")
        return data

    def _path(self, run_id: str) -> Path:
        if not _SAFE_RUN_ID.fullmatch(run_id):
            raise ValueError("invalid run_id")
        return self.root / f"{run_id}.json"


class RAGWorkflow:
    """Plan -> optional approval -> execute -> validate/fail, with finite retries."""

    def __init__(
        self,
        service: GroundedRAGService,
        checkpoint_store: CheckpointStore,
        *,
        max_retries: int = 1,
    ) -> None:
        if not 0 <= max_retries <= 3:
            raise ValueError("max_retries must be between 0 and 3")
        self._service = service
        self._checkpoints = checkpoint_store
        self._max_retries = max_retries

    def run(
        self,
        question: str,
        *,
        top_k: int = 3,
        filters: dict[str, str] | None = None,
        require_confirmation: bool = False,
        approved: bool = False,
    ) -> WorkflowResult:
        plan = WorkflowPlan(
            action="answer_job_demand_question",
            question=question,
            top_k=top_k,
            filters=dict(filters or {}),
            requires_confirmation=require_confirmation,
        )
        result = WorkflowResult(
            run_id=str(uuid.uuid4()),
            status="planned",
            plan=plan,
            events=[{"type": "plan", "action": plan.action}],
        )
        if require_confirmation and not approved:
            result.status = "awaiting_confirmation"
            result.events.append({"type": "checkpoint", "reason": "human_confirmation_required"})
            self._checkpoints.save(result)
            return result
        return self._execute(result)

    def resume(self, run_id: str, *, approved: bool) -> WorkflowResult:
        payload = self._checkpoints.load(run_id)
        if payload.get("status") != "awaiting_confirmation":
            raise ValueError("workflow is not awaiting confirmation")
        plan = WorkflowPlan(**payload["plan"])
        result = WorkflowResult(
            run_id=run_id,
            status="awaiting_confirmation",
            plan=plan,
            attempts=int(payload.get("attempts", 0)),
            events=list(payload.get("events", [])),
        )
        if not approved:
            result.status = "cancelled"
            result.events.append({"type": "cancelled", "reason": "human_rejected"})
            self._checkpoints.save(result)
            return result
        result.events.append({"type": "confirmation", "approved": True})
        return self._execute(result)

    def _execute(self, result: WorkflowResult) -> WorkflowResult:
        for attempt in range(1, self._max_retries + 2):
            result.attempts = attempt
            result.status = "running"
            result.events.append({"type": "attempt", "number": attempt})
            try:
                answer = self._service.answer(
                    result.plan.question,
                    top_k=result.plan.top_k,
                    filters=result.plan.filters,
                )
                result.answer = answer
                result.status = "completed" if not answer.refused else "failed_safely"
                result.events.append({"type": "answer_validation", "failure_type": answer.failure_type})
                self._checkpoints.save(result)
                return result
            except (TimeoutError, ConnectionError) as error:
                result.events.append({"type": "retryable_error", "error": type(error).__name__})
                if attempt <= self._max_retries:
                    result.status = "retrying"
                    continue
                result.answer = _safe_failure(result.plan.question, "tool_failure")
            except RuntimeError:
                result.answer = _safe_failure(result.plan.question, "model_failure")
            result.status = "failed_safely"
            result.events.append({"type": "safe_failure", "failure_type": result.answer.failure_type})
            self._checkpoints.save(result)
            return result
        raise AssertionError("bounded workflow loop exited unexpectedly")


def _safe_failure(question: str, failure_type: str) -> GroundedAnswer:
    return GroundedAnswer(
        question=question,
        resolved_query=question,
        answer=REFUSAL_TEXT,
        refused=True,
        failure_type=failure_type,
    )
