"""LangGraph implementation of the existing bounded RAG execution loop.

The public planning, confirmation, and JSON-checkpoint boundaries remain in
``RAGWorkflow``.  This class replaces only the execute/retry control flow with
a small StateGraph so it can be compared directly with the handwritten loop.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from rag_workflow import RAGWorkflow, WorkflowResult, _safe_failure


class ExecutionState(TypedDict):
    """The graph's shared running record for one execution phase."""

    result: WorkflowResult


class LangGraphRAGWorkflow(RAGWorkflow):
    """Keep the public workflow contract while using a graph for retries."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._execution_graph = self._build_execution_graph()

    def _execute(self, result: WorkflowResult) -> WorkflowResult:
        state = self._execution_graph.invoke({"result": result})
        completed = state["result"]
        self._checkpoints.save(completed)
        return completed

    def _build_execution_graph(self) -> Any:
        builder = StateGraph(ExecutionState)
        builder.add_node("execute_attempt", self._execute_attempt)
        builder.add_edge(START, "execute_attempt")
        builder.add_conditional_edges("execute_attempt", self._route_after_attempt)
        return builder.compile()

    def _execute_attempt(self, state: ExecutionState) -> dict[str, WorkflowResult]:
        previous = state["result"]
        attempt = previous.attempts + 1
        running = replace(
            previous,
            attempts=attempt,
            status="running",
            events=[*previous.events, {"type": "attempt", "number": attempt}],
        )
        try:
            answer = self._service.answer(
                running.plan.question,
                top_k=running.plan.top_k,
                filters=running.plan.filters,
            )
        except (TimeoutError, ConnectionError) as error:
            events = [*running.events, {"type": "retryable_error", "error": type(error).__name__}]
            if attempt <= self._max_retries:
                return {"result": replace(running, status="retrying", events=events)}
            answer = _safe_failure(running.plan.question, "tool_failure")
            return {
                "result": replace(
                    running,
                    answer=answer,
                    status="failed_safely",
                    events=[*events, {"type": "safe_failure", "failure_type": answer.failure_type}],
                )
            }
        except RuntimeError:
            answer = _safe_failure(running.plan.question, "model_failure")
            return {
                "result": replace(
                    running,
                    answer=answer,
                    status="failed_safely",
                    events=[*running.events, {"type": "safe_failure", "failure_type": answer.failure_type}],
                )
            }

        status = "failed_safely" if answer.refused else "completed"
        return {
            "result": replace(
                running,
                answer=answer,
                status=status,
                events=[*running.events, {"type": "answer_validation", "failure_type": answer.failure_type}],
            )
        }

    @staticmethod
    def _route_after_attempt(state: ExecutionState) -> str:
        if state["result"].status == "retrying":
            return "execute_attempt"
        return END
