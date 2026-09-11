"""Build a career-aligned learning report from the verified job increment.

The 215-record enriched corpus is designed for retrieval experiments.  It
contains a balanced 150-card legacy sample, so it must not be used as a market
prevalence estimate.  This report uses the 68-record verified increment and
keeps application/Agent/RAG roles separate from model/algorithm roles.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

from job_dataset_pipeline import load_job_records, skill_frequencies


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
REPORT_DATE = "2026-08-24"

ALGORITHM_ROLE_TERMS = (
    "算法",
    "训练",
    "推理",
    "基座",
    "世界模型",
    "多模态",
    "感知",
    "VLA",
    "Infra",
    "预训练",
    "后训练",
    "模型研发",
)
APPLICATION_ROLE_TERMS = (
    "Agent",
    "智能体",
    "AI应用",
    "AI 应用",
    "RAG",
    "知识库",
    "大模型应用",
    "AI平台",
    "AI 平台",
    "AI开发",
    "AI 开发",
)
EARLY_CAREER_TERMS = (
    "经验不限",
    "无需经验",
    "应届",
    "校招",
    "校园招聘",
    "在校",
    "实习",
    "1年以内",
    "1-3年",
    "1－3年",
    "1—3年",
)
U10_SKILLS = (
    "Workflow",
    "State Management",
    "Context Management",
    "Memory",
    "Failure Handling",
    "Observability",
    "Testing",
)


def classify_role(record: dict[str, str]) -> str:
    """Classify by role title/category, preferring explicit algorithm terms."""

    title = f"{record.get('job_title', '')} {record.get('category', '')}"
    if any(term.lower() in title.lower() for term in ALGORITHM_ROLE_TERMS):
        return "algorithm_model"
    if any(term.lower() in title.lower() for term in APPLICATION_ROLE_TERMS):
        return "application_agent_rag"
    return "other"


def is_independent_page(record: dict[str, str]) -> bool:
    return record.get("verification_status") == "public_page_verified"


def is_early_career(record: dict[str, str]) -> bool:
    experience = record.get("experience", "")
    return any(term in experience for term in EARLY_CAREER_TERMS)


def tags(record: dict[str, str]) -> set[str]:
    return {
        item.strip()
        for item in record.get("normalized_skills", "").split(";")
        if item.strip()
    }


def subset_stats(records: Sequence[dict[str, str]]) -> dict[str, object]:
    return {
        "jobs": len(records),
        "skill_frequencies": skill_frequencies(records),
        "experience_counts": dict(
            sorted(Counter(record.get("experience", "未公开") for record in records).items())
        ),
    }


def specific_skill_counts(
    records: Sequence[dict[str, str]], skills: Iterable[str]
) -> list[dict[str, object]]:
    total = len(records)
    result = []
    for skill in skills:
        count = sum(skill in tags(record) for record in records)
        result.append(
            {
                "skill": skill,
                "jobs": count,
                "share": count / total if total else 0.0,
            }
        )
    return result


def analyze(records: list[dict[str, str]]) -> dict[str, object]:
    grouped = {
        group: [record for record in records if classify_role(record) == group]
        for group in ("application_agent_rag", "algorithm_model", "other")
    }
    verified_application = [
        record for record in grouped["application_agent_rag"] if is_independent_page(record)
    ]
    verified_algorithm = [
        record for record in grouped["algorithm_model"] if is_independent_page(record)
    ]
    early_application = [
        record for record in verified_application if is_early_career(record)
    ]

    stats: dict[str, object] = {
        "report_date": REPORT_DATE,
        "source_collected_at": max(record.get("collected_at", "") for record in records),
        "jobs": len(records),
        "independent_job_pages": sum(is_independent_page(record) for record in records),
        "aggregate_cards": sum(not is_independent_page(record) for record in records),
        "role_group_counts": {group: len(items) for group, items in grouped.items()},
        "application_agent_rag": subset_stats(grouped["application_agent_rag"]),
        "verified_application_agent_rag": subset_stats(verified_application),
        "early_career_verified_application": subset_stats(early_application),
        "algorithm_model": subset_stats(grouped["algorithm_model"]),
        "verified_algorithm_model": subset_stats(verified_algorithm),
        "u10_skill_counts": specific_skill_counts(verified_application, U10_SKILLS),
    }
    validate(stats)
    return stats


def validate(stats: dict[str, object]) -> None:
    groups = stats["role_group_counts"]
    if sum(groups.values()) != stats["jobs"]:
        raise ValueError("role groups do not add up to the source sample")
    if stats["independent_job_pages"] + stats["aggregate_cards"] != stats["jobs"]:
        raise ValueError("evidence levels do not add up to the source sample")
    for section_name in (
        "verified_application_agent_rag",
        "early_career_verified_application",
        "verified_algorithm_model",
    ):
        section = stats[section_name]
        total = section["jobs"]
        if any(item["jobs"] > total for item in section["skill_frequencies"]):
            raise ValueError(f"invalid skill count in {section_name}")


def top_skills(section: dict[str, object], limit: int = 12) -> list[dict[str, object]]:
    return list(section["skill_frequencies"][:limit])


def append_skill_table(lines: list[str], items: Sequence[dict[str, object]]) -> None:
    lines.extend(
        [
            "| 标准化技能 | 岗位数 | 占比 |",
            "|---|---:|---:|",
        ]
    )
    for item in items:
        lines.append(f"| {item['skill']} | {item['jobs']} | {item['share']:.1%} |")


def render_markdown(stats: dict[str, object]) -> str:
    groups = stats["role_group_counts"]
    verified_app = stats["verified_application_agent_rag"]
    verified_algorithm = stats["verified_algorithm_model"]
    early_app = stats["early_career_verified_application"]
    lines = [
        "# 2026-08-24 岗位需求与学习路线校准",
        "",
        "## 结论先行",
        "",
        "- 当前主项目仍然适合求职：目标应用岗位最常见的是 Python、RAG、Agent、工具调用和部署。",
        "- U10 没有偏题，但应只保留状态、分支、有限重试和事件记录的最小闭环；生产级恢复、幂等和原子 checkpoint 下放到进阶。",
        "- AI 应用岗位与模型算法岗位需要两条不同的能力画像，不能把 PyTorch/训练推理要求直接塞进当前 Agent/RAG 主线。",
        "- `learn-claude-code` 复现项目适合作为 Agent harness 选修实验，不替代当前原创求职项目。",
        "",
        "## 样本口径",
        "",
        f"- 分析日期：{stats['report_date']}；岗位数据采集日期：{stats['source_collected_at']}。",
        f"- 增量样本：{stats['jobs']} 条，其中独立职位页 {stats['independent_job_pages']} 条、聚合卡片 {stats['aggregate_cards']} 条。",
        f"- 标题分组：应用/Agent/RAG {groups['application_agent_rag']} 条，模型/算法 {groups['algorithm_model']} 条，其他 {groups['other']} 条。",
        "- 215 条去重语料用于 RAG 检索演示；其中旧 150 条按五类各 30 条采样，不能当作市场占比。",
        "- 下方主路线频率只使用独立职位页，聚合卡片仅作趋势发现。",
        "",
        f"## 目标应用岗位：{verified_app['jobs']} 条独立职位页",
        "",
    ]
    append_skill_table(lines, top_skills(verified_app, 16))
    lines.extend(
        [
            "",
            f"其中与校招、实习、经验不限或 1–3 年匹配的独立职位页有 {early_app['jobs']} 条。样本仍小，因此只用于检查方向，不单独宣称市场占比。",
            "",
            "## 模型/算法岗位：独立职位页",
            "",
            f"该组有 {verified_algorithm['jobs']} 条独立职位页，核心要求明显不同：",
            "",
        ]
    )
    append_skill_table(lines, top_skills(verified_algorithm, 14))
    lines.extend(
        [
            "",
            "这条路线更偏 LLM 原理、PyTorch、训练/后训练、推理部署、多模态和模型评测。它可以作为以后转算法岗的专项路线，但不应与当前 AI 应用工程主线同时推进。",
            "",
            "## U10 深度校准",
            "",
            f"分母为 {verified_app['jobs']} 条目标应用岗位独立职位页：",
            "",
            "| U10 相关标签 | 岗位数 | 占比 | 课程处理 |",
            "|---|---:|---:|---|",
        ]
    )
    decisions = {
        "Workflow": "保留：能画顺序、分支和终止状态",
        "State Management": "最小理解；具体状态对象不深挖",
        "Context Management": "放到 Agent 上下文专题",
        "Memory": "进阶选修",
        "Failure Handling": "保留：可重试与不可重试错误",
        "Observability": "保留事件/trace 概念，工具细节后置",
        "Testing": "与每个功能一起实践，不单独背术语",
    }
    for item in stats["u10_skill_counts"]:
        lines.append(
            f"| {item['skill']} | {item['jobs']} | {item['share']:.1%} | {decisions[item['skill']]} |"
        )
    lines.extend(
        [
            "",
            "当前 U10 的最低验收改为：看懂一张状态图；区分临时故障与业务拒答；解释当前状态和事件历史的区别；能读懂一个有限重试测试。checkpoint 的临时文件写法、崩溃中途恢复、外部副作用幂等只作进阶阅读。",
            "",
            "## 求职优先级",
            "",
            "### P0：当前主线必须完成",
            "",
            "1. Python 与软件工程基础：函数、类型、异常、测试、日志、HTTP。",
            "2. RAG：切分、Embedding、检索、引用、拒答和固定评测。",
            "3. Agent 与工具调用：Agent loop、schema、白名单、参数校验。",
            "4. 最小工作流：状态、分支、有限重试、事件记录。",
            "5. 可运行服务：FastAPI/CLI、配置、错误响应、README 与演示。",
            "",
            "### P1：主项目完成后提升竞争力",
            "",
            "- 用 LangGraph 重构同一条工作流，并复用同一组测试。",
            "- Docker + 数据库基础 + CI；是否加入 pgvector/Redis 由项目瓶颈决定。",
            "- 扩充困难 RAG 评测，增加 reranker 对照和真实模型小样本。",
            "- 用 trace、版本和成本记录解释一次真实失败。",
            "",
            "### P2：按目标岗位选择",
            "",
            "- 应用平台：MCP、Dify/RAGFlow、多 Agent。",
            "- 算法方向：PyTorch、微调/SFT、推理优化、vLLM、多模态。",
            "- 企业平台：Redis、异步任务、RBAC、多租户、Kubernetes。",
            "",
            "## 2026-08-24 网页抽查（不计入频率）",
            "",
            "- [海尔 AI 平台岗](https://maker.haier.net/client/job/detail/id/10228552/recommend_record/1)仍明确出现 Agent Runtime、工具调用、任务编排、RAG 和生产部署。",
            "- [百度深圳大模型算法岗](https://talent.baidu.com/jobs/detail/GRADUATE/ed7155b8-3af7-4ea4-adc4-c19d0f54aa21)仍明显偏预训练、后训练、强化学习和数据管线。",
            "- [近期校招 Agent 岗](https://www.nowcoder.com/jobs/detail/461361)继续强调 Python、RAG、Function Calling、框架和完整项目落地。",
            "",
            "## 证据边界",
            "",
            "- 这不是全市场普查；深圳样本占比高，且社招多于校招。",
            "- 技能出现频率不等于每个候选人都必须掌握，也不能推导薪资或录用概率。",
            "- `normalized_skills` 来自岗位摘要归一化，未出现的词不等于岗位完全不需要该能力。",
            "- 课程优先级同时考虑频率、可迁移性、当前基础和能否形成可验证项目证据。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze career-aligned AI job demand")
    parser.add_argument(
        "--input",
        default=str(ROOT / "data" / "raw" / "verified_jobs_2026-08-20.csv"),
    )
    parser.add_argument(
        "--json-output",
        default=str(ROOT / "data" / "processed" / "learning_alignment_2026-08-24.json"),
    )
    parser.add_argument(
        "--markdown-output",
        default=str(WORKSPACE / "notes" / "job-market" / "2026-08-24.md"),
    )
    args = parser.parse_args()

    records = load_job_records([args.input])
    stats = analyze(records)
    json_path = Path(args.json_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path = Path(args.markdown_output)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(stats), encoding="utf-8")
    print(
        json.dumps(
            {
                "jobs": stats["jobs"],
                "independent_job_pages": stats["independent_job_pages"],
                "role_group_counts": stats["role_group_counts"],
                "verified_application_jobs": stats["verified_application_agent_rag"]["jobs"],
                "verified_algorithm_jobs": stats["verified_algorithm_model"]["jobs"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"report: {markdown_path}")


if __name__ == "__main__":
    main()
