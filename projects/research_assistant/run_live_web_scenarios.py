"""Run a bounded real-world scenario suite against the personal local web API.

The suite is deliberately opt-in.  It sends at most four evidence questions to
the already running real-model service, never retries a request, and keeps one
deterministic market-analysis case to verify that not every question consumes a
model call.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    ROOT.parents[1]
    / "output"
    / f"live-web-scenarios-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
)
PERSONAL_UI_MARKER = "personal-live-rag-ui"


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    question: str
    top_k: int
    city: str | None = None
    expected_mode: str = "rag"
    require_non_refused: bool = True
    require_model_usage: bool = True
    allowed_failure_types: tuple[str, ...] = ()

    def request_body(self) -> dict[str, Any]:
        body: dict[str, Any] = {"question": self.question, "top_k": self.top_k}
        if self.city:
            body["city"] = self.city
        return body


DEFAULT_SCENARIOS = (
    Scenario(
        scenario_id="shenzhen_langgraph_evidence",
        question="哪些深圳岗位明确提到 LangGraph？请列出岗位名称并逐条引用证据。",
        city="深圳",
        top_k=3,
    ),
    Scenario(
        scenario_id="multi_skill_comparison",
        question="找出同时要求 RAG、Python 和部署能力的岗位，比较它们的职责差异；没有证据的内容不要推断。",
        top_k=5,
        allowed_failure_types=("output_truncated",),
    ),
    Scenario(
        scenario_id="candidate_fit",
        question="候选人有 PyTorch、CUDA 和 Agent 项目经验。哪些岗位与这些经历更匹配？请区分直接证据和你的归纳。",
        top_k=5,
    ),
    Scenario(
        scenario_id="job_freshness_boundary",
        question="这些岗位现在是否仍在招聘？如果现有证据不能确认，请明确说明不能确认，不要猜测。",
        top_k=3,
        require_non_refused=False,
        require_model_usage=False,
    ),
    Scenario(
        scenario_id="deterministic_market_count",
        question="当前独立职位页目标样本中，Python、RAG 和 Agent 分别出现多少次？",
        top_k=3,
        expected_mode="market_analysis",
        require_model_usage=False,
    ),
)


def validate_scenario(scenario: Scenario, payload: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if payload.get("question") != scenario.question:
        failures.append("response question does not match request")
    if payload.get("answer_mode") != scenario.expected_mode:
        failures.append(
            f"expected answer_mode={scenario.expected_mode}, got {payload.get('answer_mode')!r}"
        )
    answer = payload.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        failures.append("answer is empty")
        answer = ""
    refused = payload.get("refused") is True
    if scenario.require_non_refused and refused:
        failures.append(f"unexpected refusal: {payload.get('failure_type')!r}")

    citations = payload.get("citations")
    if not isinstance(citations, list):
        failures.append("citations is not a list")
        citations = []
    if scenario.expected_mode == "rag" and not refused:
        if not citations:
            failures.append("grounded answer has no citations")
        for citation in citations:
            number = citation.get("number") if isinstance(citation, dict) else None
            if not isinstance(number, int) or f"[{number}]" not in answer:
                failures.append(f"answer is missing citation marker [{number}]")

    usage = payload.get("model_usage")
    if scenario.require_model_usage and not isinstance(usage, dict):
        failures.append("real-model answer has no provider usage")
    if scenario.expected_mode == "market_analysis" and usage is not None:
        failures.append("deterministic market analysis unexpectedly used the model")
    return failures


def validate_http_result(
    scenario: Scenario,
    status: int,
    payload: dict[str, Any],
) -> tuple[list[str], str]:
    if status == 200:
        return validate_scenario(scenario, payload), "answer"
    detail = payload.get("detail")
    failure_type = detail.get("failure_type") if isinstance(detail, dict) else None
    if failure_type in scenario.allowed_failure_types:
        return [], "safe_failure"
    return [f"HTTP {status}: {failure_type or 'unknown_failure'}"], "error"


def _read_json(url: str, *, timeout_seconds: float) -> dict[str, Any]:
    with urlopen(url, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def _read_text(url: str, *, timeout_seconds: float) -> str:
    with urlopen(url, timeout=timeout_seconds) as response:
        return response.read().decode("utf-8")


def _post_json(
    url: str,
    body: dict[str, Any],
    *,
    timeout_seconds: float,
) -> tuple[int, dict[str, Any]]:
    request = Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        raw = error.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"detail": raw}
        return error.code, payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run five bounded real-world cases against a running personal RAG web service"
    )
    parser.add_argument(
        "--execute-real-api",
        action="store_true",
        help="required acknowledgement: up to four cases may call the paid model API",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout-seconds", type=float, default=130.0)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--case",
        action="append",
        choices=[item.scenario_id for item in DEFAULT_SCENARIOS],
        help="run selected case(s); omit to run the complete five-case suite",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.execute_real_api:
        raise SystemExit(
            "refusing to run: pass --execute-real-api to acknowledge up to four paid calls"
        )
    if args.timeout_seconds <= 0:
        raise SystemExit("timeout-seconds must be positive")

    selected_ids = set(args.case or ())
    scenarios = [
        item for item in DEFAULT_SCENARIOS if not selected_ids or item.scenario_id in selected_ids
    ]
    base_url = args.base_url.rstrip("/")
    started_at = datetime.now(timezone.utc)
    report: dict[str, Any] = {
        "schema_version": 1,
        "started_at": started_at.isoformat(),
        "base_url": base_url,
        "request_retries": 0,
        "maximum_paid_calls": sum(
            1 for item in scenarios if item.expected_mode == "rag"
        ),
        "cases": [],
    }

    try:
        health = _read_json(base_url + "/health", timeout_seconds=5.0)
        page = _read_text(base_url + "/", timeout_seconds=5.0)
    except (OSError, URLError, UnicodeDecodeError, json.JSONDecodeError) as error:
        report["startup_error"] = f"{type(error).__name__}: {error}"
        _write_report(Path(args.output), report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2

    report["health"] = health
    report["personal_ui_detected"] = PERSONAL_UI_MARKER in page
    startup_failures: list[str] = []
    if health.get("status") != "ok":
        startup_failures.append("health status is not ok")
    if health.get("retrieval_strategy") != "hybrid_rerank":
        startup_failures.append("service is not using hybrid_rerank")
    if PERSONAL_UI_MARKER not in page:
        startup_failures.append("personal web UI marker is missing")
    report["startup_failures"] = startup_failures

    for scenario in scenarios:
        case_started = perf_counter()
        try:
            status, payload = _post_json(
                base_url + "/ask",
                scenario.request_body(),
                timeout_seconds=args.timeout_seconds,
            )
            failures, outcome = validate_http_result(scenario, status, payload)
            report["cases"].append(
                {
                    "scenario_id": scenario.scenario_id,
                    "request": scenario.request_body(),
                    "http_status": status,
                    "outcome": outcome,
                    "answer_produced": status == 200,
                    "automatic_contract_passed": not failures,
                    "automatic_failures": failures,
                    "wall_latency_ms": round((perf_counter() - case_started) * 1000, 3),
                    "response": payload,
                    "human_review": "pending",
                }
            )
        except (OSError, URLError, UnicodeDecodeError, json.JSONDecodeError) as error:
            report["cases"].append(
                {
                    "scenario_id": scenario.scenario_id,
                    "request": scenario.request_body(),
                    "automatic_contract_passed": False,
                    "automatic_failures": [f"{type(error).__name__}: {error}"],
                    "wall_latency_ms": round((perf_counter() - case_started) * 1000, 3),
                    "human_review": "not_run",
                }
            )

    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["automatic_suite_passed"] = not startup_failures and all(
        item["automatic_contract_passed"] for item in report["cases"]
    )
    output = _write_report(Path(args.output), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report: {output}")
    return 0 if report["automatic_suite_passed"] else 1


def _write_report(path: Path, report: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
