"""Build a local-only replay from selected public excerpts and actual project code."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from unittest.mock import patch
from urllib.parse import urlsplit
from uuid import uuid4

from document import Chunk
from grounded_service import ExtractiveGroundedGenerator, GroundedRAGService
from retriever import InMemoryRetriever

ROOT = Path(__file__).resolve().parent
DEMO = ROOT / "demo"


def check_public_text(text: str) -> None:
    """A small export check, not a comprehensive personal-data detector."""
    patterns = {
        "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        "mobile": r"(?<![A-Za-z0-9])1[3-9]\d{9}(?![A-Za-z0-9])",
        "credential": r"sk-[A-Za-z0-9_-]{20,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----",
        "local_path": r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]|/(?:Users|home)/",
    }
    for label, expression in patterns.items():
        if re.search(expression, text):
            raise ValueError(f"public export rejected: {label}")


def validate_url(url: str) -> str:
    parts = urlsplit(url)
    if (parts.scheme != "https" or parts.hostname not in {
            "talent.baidu.com", "documentation.espressif.com"}
            or parts.username or parts.password or parts.query or parts.fragment):
        raise ValueError("public export rejected: source URL")
    return url


def load_sources(path: Path = DEMO / "selected_sources.json") -> dict:
    raw = path.read_text(encoding="utf-8")
    check_public_text(raw)
    source = json.loads(raw)
    if set(source) != {"schema_version", "scope", "jobs", "pdf_recording"} or source["schema_version"] != 1:
        raise ValueError("unexpected source schema")
    if len(source["jobs"]) != 2:
        raise ValueError("this replay expects exactly two selected job excerpts")
    fields = {"id", "title", "company", "city", "text", "skills", "source_url", "collected_at", "source_kind"}
    for job in source["jobs"]:
        if set(job) != fields or not 1 <= len(job["text"]) <= 400:
            raise ValueError("unexpected job export fields or excerpt length")
        validate_url(job["source_url"])
    recording = source["pdf_recording"]
    fields = {"question", "answer", "recorded_at", "model", "model_usage", "latency_ms", "refused",
              "source_url", "source_title", "pdf_page", "printed_page", "quote", "origin_report_sha256",
              "origin_case_id", "origin_document_sha256"}
    if set(recording) != fields or len(recording["quote"]) > 180 or len(recording["answer"]) > 500:
        raise ValueError("unexpected PDF recording fields or excerpt length")
    validate_url(recording["source_url"])
    return source


@contextmanager
def without_network():
    def denied(*args, **kwargs):
        raise RuntimeError("replay generation does not make network connections")
    with patch("socket.socket.connect", denied), patch("socket.create_connection", denied), patch("socket.getaddrinfo", denied):
        yield


def source_card(job):
    return {"title": f'{job["company"]} · {job["title"]}', "text": job["text"],
            "url": job["source_url"], "label": f'{job["source_kind"]} · 采集于 {job["collected_at"]}'}


def base_case(case_id, title, category, subtitle, question, result, evidence, steps, facts, note):
    return {"id": case_id, "title": title, "category": category, "subtitle": subtitle,
            "question": question, "result": result, "evidence": evidence, "steps": steps,
            "facts": facts, "note": note}


def job_cases(source):
    chunks = [Chunk(j["id"], j["id"], j["text"], 0, len(j["text"]),
                    {"city": j["city"], "company": j["company"], "job_title": j["title"],
                     "source_url": j["source_url"]}) for j in source["jobs"]]
    service = GroundedRAGService(InMemoryRetriever(chunks))
    answer = service.answer("RAG Agent", top_k=2)
    by_id = {j["id"]: j for j in source["jobs"]}
    evidence = [{**source_card(by_id[c.chunk_id]), "number": c.number} for c in answer.citations]
    successful = base_case("job-evidence", "岗位要求，有据可查", "岗位检索", "从问题找到两条岗位证据",
        "这两条岗位摘要中，哪些内容涉及 RAG 和 Agent？", answer.answer, evidence,
        ["读取两条已选岗位摘要", "以 RAG Agent 做关键词检索", "返回匹配的摘要与引用编号"],
        [{"label": "检索范围", "value": "2 条历史摘要"}, {"label": "运行方式", "value": "关键词检索＋摘录"},
         {"label": "来源引用", "value": str(len(answer.citations)) + " 条"}],
        "这是项目对选定摘要的实际运行结果。摘录帮助回查要求，不代表模型生成，也不证明岗位目前仍在招聘。")
    successful["raw_result"] = asdict(answer)
    successful["execution_query"] = "RAG Agent"
    empty = service.answer("RAG Agent", top_k=2, filters={"city": "广州"})
    refused = base_case("empty-scope", "没有匹配，就明确拒答", "失败边界", "看不到证据时，不补出结论",
        "在这两条摘要中，只看广州的 RAG / Agent 岗位。", empty.answer, [],
        ["读取相同的两条深圳岗位摘要", "增加城市筛选：广州", "检索为空，返回明确的拒答结果"],
        [{"label": "筛选条件", "value": "广州"}, {"label": "匹配证据", "value": "0 条"},
         {"label": "系统结果", "value": "拒答"}],
        "没有匹配只针对本次两条摘要，不能解释成广州不存在相关岗位。该案例在生成回答之前结束。")
    refused["raw_result"] = asdict(empty)
    return [successful, refused]


def backend_case(source, runtime):
    from job_backend.contracts import JobInput, EvaluationRequest, ReviewRequest
    from job_backend.database import make_engine, migrate, sessions
    from job_backend.repository import Repository, Conflict
    from job_backend.evaluation import EvaluationService

    job = source["jobs"][0]
    runtime.mkdir(parents=True, exist_ok=True)
    database = runtime / ("demo-" + uuid4().hex + ".sqlite")
    engine = make_engine("sqlite:///" + str(database.resolve()))
    calls = 0

    class CountedExtractive(ExtractiveGroundedGenerator):
        def generate(self, prompt, results):
            nonlocal calls
            calls += 1
            return super().generate(prompt, results)

    config = {"mode": "extractive", "scope": "public-replay"}
    try:
        migrate(engine)
        repo = Repository(sessions(engine))
        item = JobInput(source_key=job["id"], source_url=job["source_url"], title=job["title"],
                        company=job["company"], city=job["city"], text=job["text"], skills=job["skills"],
                        source_grade="independent", collected_at=job["collected_at"])
        imported = repo.import_jobs([item])
        duplicate = repo.import_jobs([item])
        snapshot_id = imported["items"][0]["snapshot_id"]
        request = EvaluationRequest(snapshot_id=snapshot_id, profile={"cities": ["深圳"],
            "graduation_year": 2026, "work_experience_months": 0, "capabilities": [
                {"skill": "Python", "evidence": "示例能力：完成 Python 练习", "source": "演示输入"},
                {"skill": "RAG", "evidence": "示例能力：完成检索练习", "source": "演示输入"}]})
        evaluator = EvaluationService(repo, CountedExtractive, config)
        result, replayed_first = evaluator.evaluate("public-demo-evaluation", request)
        replay, replayed = evaluator.evaluate("public-demo-evaluation", request)
        reviewed = repo.add_review(result["id"], ReviewRequest(expected_version=0, decision="needs_confirmation",
                                                            note="演示确认：届别和岗位状态仍需自行核实。"))
        conflict = False
        try:
            repo.add_review(result["id"], ReviewRequest(expected_version=0, decision="consider", note="过期版本"))
        except Conflict:
            conflict = True
        engine.dispose()
        engine = make_engine("sqlite:///" + str(database.resolve()))
        restored_repo = Repository(sessions(engine))
        restored = restored_repo.get_evaluation(result["id"])
        restored_replay, replayed_after_reopen = EvaluationService(restored_repo, CountedExtractive, config).evaluate(
            "public-demo-evaluation", request)
        checks = {"duplicate_import_created_zero": duplicate["created_snapshots"] == 0,
                  "same_key_replayed": replayed and replay["id"] == result["id"] and not replayed_first,
                  "generator_calls": calls, "review_version": reviewed["review_version"],
                  "stale_review_conflict": conflict, "reopened_record_readable": restored["id"] == result["id"],
                  "reopened_request_replayed": replayed_after_reopen and restored_replay["id"] == result["id"]}
        if not all(v == (1 if k in {"generator_calls", "review_version"} else True) for k, v in checks.items()):
            raise RuntimeError("backend replay checks failed")
        comparison = result["result"]["comparison"]
        matched = "、".join(x["skill"] for x in comparison["supported_mentions"])
        missing = "、".join(comparison["mentions_without_profile_evidence"])
        case = base_case("saved-evaluation", "评估结果，可以复查", "后端流程", "重复请求与人工确认都有记录",
            "用一份示例能力清单评估所选岗位，再重复请求并添加复查意见。",
            f"已匹配到有示例能力证据的标签：{matched}。\n尚无示例能力证据的岗位标签：{missing}。\n届别、经验及岗位是否仍开放：需要另行核实。",
            [{**source_card(job), "number": 1}],
            ["保存所选岗位的历史快照", "对照示例能力，保存评估结果", "相同请求直接读取原结果", "追加复查意见，拒绝过期版本", "重新打开数据库后仍可读取结果"],
            [{"label": "生成执行", "value": f"{calls} 次"}, {"label": "重复请求", "value": "读取原结果"},
             {"label": "保存方式", "value": "本机 SQLite"}],
            "能力清单仅为演示输入，不包含个人简历。标签对照不是录用评分。此例实际执行本机 SQLite 流程；不代表公网或多用户验证。")
        case["checks"] = checks
        case["comparison"] = comparison
        return case
    finally:
        engine.dispose()


def pdf_case(source):
    r = source["pdf_recording"]
    evidence = [{"number": 1, "title": r["source_title"], "text": r["quote"],
                 "label": f'原文节选 · PDF 第 {r["pdf_page"]} 页 / 印刷页 {r["printed_page"]}',
                 "url": r["source_url"]}]
    case = base_case("pdf-citation", "文档回答，回到原文", "PDF 问答", "保留参数、默认条件与页码",
        r["question"], r["answer"], evidence,
        ["读取厂商技术文档", "检索到默认时钟与功能说明", "模型生成带引用的回答", "保留原文节选和页码供复核"],
        [{"label": "记录日期", "value": r["recorded_at"]}, {"label": "当时的模型", "value": r["model"]},
         {"label": "引用位置", "value": f'第 {r["pdf_page"]} 页'}],
        "这是此前真实模型调用的保存记录，本次未重新调用模型。仅展示必要原文节选，不附完整厂商 PDF。引用合法仍不等于语义自动核验。")
    case["recording"] = r
    return case


def build_payload(source, runtime):
    with without_network():
        cases = job_cases(source) + [backend_case(source, runtime), pdf_case(source)]
    code_files = ["demo_replay.py", "grounded_service.py", "retriever.py", "security.py", "prompt.py",
                  "job_backend/evaluation.py", "job_backend/repository.py"]
    payload = {"schema_version": 1, "built_at": datetime.now(timezone.utc).isoformat(),
               "title": "岗位需求 RAG 分析助手", "cases": cases,
               "source_sha256": sha256(json.dumps(source, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
               "code_sha256": {name: sha256((ROOT / name).read_bytes()).hexdigest() for name in code_files},
               "privacy": "只包含两条历史岗位摘要、一条 PDF 问答节选及演示能力输入；完整研究数据与个人资料未包含。",
               "mode": "local-static-replay", "new_model_calls": 0}
    check_public_text(json.dumps(payload, ensure_ascii=False))
    return payload


def render_html(payload, template=None):
    template = template if template is not None else (DEMO / "page.html").read_text(encoding="utf-8")
    if template.count("__DEMO_PAYLOAD__") != 1:
        raise ValueError("template requires exactly one payload slot")
    data = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    return template.replace("__DEMO_PAYLOAD__", data)


def main(argv=None):
    parser = argparse.ArgumentParser(description="生成无需服务器或模型调用的项目回放页面")
    parser.add_argument("--output", type=Path, default=DEMO)
    parser.add_argument("--runtime", type=Path, default=ROOT.parents[1] / "output" / "replay-runtime")
    args = parser.parse_args(argv)
    payload = build_payload(load_sources(), args.runtime)
    html = render_html(payload)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "index.html").write_text(html, encoding="utf-8")
    (args.output / "replay.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(payload["cases"]), "new_model_calls": 0, "output": str(args.output)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
