"""Deterministic full-corpus job-demand analysis with evidence citations.

This module answers corpus-level questions such as "哪些技能最常见".  It does
not ask an LLM to infer frequencies from Top-K retrieval results.  The default
denominator is the independently verified application/Agent/RAG job subset.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from threading import Lock
from time import perf_counter
from typing import Any, Protocol
from urllib.parse import urlsplit

from analyze_learning_alignment import classify_role, is_independent_page, tags
from grounded_service import (
    Citation,
    GroundedAnswer,
    REFUSAL_TEXT,
    rewrite_follow_up,
    validate_filters,
    validate_question,
)
from job_dataset_pipeline import ALLOWED_CITIES


class _ChunkLike(Protocol):
    chunk_id: str
    source: str
    text: str
    metadata: Mapping[str, str]


_MARKET_TERMS = (
    "高频",
    "中频",
    "低频",
    "占比",
    "频率",
    "频次",
    "统计",
    "整体",
    "总体",
    "全量",
    "全市场",
    "行业",
    "市场需求",
    "最常见",
    "常不常",
    "多少次",
    "多少个岗位",
    "几条岗位",
    "普遍",
    "硬性",
    "加分项",
    "核心技术",
    "核心能力",
)

_EVIDENCE_QUESTION_TERMS = (
    "哪个岗位",
    "哪些岗位",
    "哪一个岗位",
    "哪条岗位",
    "岗位原文",
    "职位原文",
    "原文证据",
)

_STATISTICAL_TERMS = (
    "高频",
    "中频",
    "低频",
    "占比",
    "频率",
    "频次",
    "统计",
    "最常见",
    "多少次",
    "多少个岗位",
    "几条岗位",
)

_FOCUS_SKILL_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("BM25", ("bm25",)),
    ("Hybrid Search", ("混合检索", "hybrid search", "hybrid retrieval")),
    ("Vector Search", ("向量检索", "vector search", "vector retrieval")),
    ("Vector Database", ("向量数据库", "向量库", "vector database")),
    ("Embedding", ("embedding", "嵌入模型", "向量化")),
    ("Reranking", ("rerank", "重排序", "重排")),
    ("GraphRAG", ("graphrag", "图检索增强")),
    ("RAG", ("rag", "检索增强生成")),
)

_WARNINGS = (
    "频次层级只描述技能在当前样本中的出现率，不等于硬性要求、优先项或加分项。",
    "当前结构化数据没有逐项保留“必须/优先/加分”等要求语气，因此不能可靠输出该分类。",
    "只统计 normalized_skills 中显式归一化的标签；未出现不代表岗位一定不需要。",
    "样本不是全市场普查，且地区与岗位来源存在偏差，结果只用于校准学习和求职方向。",
)


@dataclass(frozen=True)
class JobEvidence:
    record_key: str
    chunk_id: str
    source: str
    source_url: str
    job_title: str
    company: str
    city: str
    excerpt: str

    def to_dict(self) -> dict[str, str]:
        return {
            "record_key": self.record_key,
            "chunk_id": self.chunk_id,
            "source": self.source,
            "source_url": self.source_url,
            "job_title": self.job_title,
            "company": self.company,
            "city": self.city,
            "excerpt": self.excerpt,
        }


@dataclass(frozen=True)
class SkillFrequency:
    skill: str
    jobs: int
    share: float
    frequency_tier: str
    matched_record_keys: tuple[str, ...] = ()
    evidence: tuple[JobEvidence, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill": self.skill,
            "jobs": self.jobs,
            "share": self.share,
            "frequency_tier": self.frequency_tier,
            "matched_record_keys": list(self.matched_record_keys),
            "evidence": [item.to_dict() for item in self.evidence],
        }


@dataclass(frozen=True)
class MarketAnalysis:
    retrieval_corpus_jobs: int
    sample_size: int
    role_group: str
    evidence_scope: str
    filters: Mapping[str, str]
    skills: tuple[SkillFrequency, ...]
    sample_jobs: tuple[JobEvidence, ...]
    warnings: tuple[str, ...] = _WARNINGS

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrieval_corpus_jobs": self.retrieval_corpus_jobs,
            "sample_size": self.sample_size,
            "role_group": self.role_group,
            "evidence_scope": self.evidence_scope,
            "filters": dict(self.filters),
            "top_k_affects_statistics": False,
            "requirement_modality_available": False,
            "frequency_tiers": {
                "高频": "share >= 50%",
                "中频": "20% <= share < 50%",
                "低频": "0 < share < 20%",
                "未检出": "jobs == 0",
            },
            "skills": [item.to_dict() for item in self.skills],
            "sample_jobs": [item.to_dict() for item in self.sample_jobs],
            "warnings": list(self.warnings),
        }


class MarketAnalyzer:
    """Count normalized skills once per deduplicated job record."""

    def __init__(self, chunks: Sequence[_ChunkLike]) -> None:
        self._records = _deduplicate_records(chunks)

    @property
    def corpus_size(self) -> int:
        return len(self._records)

    def analyze(
        self,
        *,
        filters: Mapping[str, str] | None = None,
        focus_skills: Sequence[str] = (),
    ) -> MarketAnalysis:
        normalized_filters = validate_filters(filters)
        records = [
            record
            for record in self._records
            if is_independent_page(record)
            and _has_valid_source_url(record)
            and classify_role(record) == "application_agent_rag"
            and _matches_filters(record, normalized_filters)
        ]
        counts = Counter(
            skill
            for record in records
            for skill in tags(record)
        )
        for skill in focus_skills:
            counts.setdefault(skill, 0)

        total = len(records)
        skills = tuple(
            SkillFrequency(
                skill=skill,
                jobs=count,
                share=count / total if total else 0.0,
                frequency_tier=_frequency_tier(count / total if total else 0.0),
                matched_record_keys=tuple(
                    record["_record_key"] for record in records if skill in tags(record)
                ),
                evidence=_representative_evidence(records, skill),
            )
            for skill, count in sorted(
                counts.items(),
                key=lambda item: (-item[1], item[0].casefold()),
            )
        )
        return MarketAnalysis(
            retrieval_corpus_jobs=self.corpus_size,
            sample_size=total,
            role_group="application_agent_rag",
            evidence_scope="public_page_verified",
            filters=normalized_filters,
            skills=skills,
            sample_jobs=tuple(_job_evidence(record) for record in records),
        )


class MarketAwareJobService:
    """Route market-wide questions to analysis and evidence questions to RAG."""

    def __init__(self, rag_factory: Callable[[], Any], analyzer: MarketAnalyzer) -> None:
        self._rag_factory = rag_factory
        self._rag_service: Any | None = None
        self._rag_lock = Lock()
        self._analyzer = analyzer

    def _get_rag_service(self) -> Any:
        # Market-only requests never load an embedding model or API adapter.
        if self._rag_service is None:
            with self._rag_lock:
                if self._rag_service is None:
                    self._rag_service = self._rag_factory()
        return self._rag_service

    def answer(
        self,
        question: str,
        *,
        top_k: int = 3,
        filters: Mapping[str, str] | None = None,
        previous_query: str | None = None,
    ) -> GroundedAnswer:
        if not 1 <= top_k <= 5:
            raise ValueError("top_k must be between 1 and 5")
        cleaned_question = validate_question(question)
        normalized_filters = validate_filters(filters)
        resolved_query = rewrite_follow_up(cleaned_question, previous_query)
        if not is_market_analysis_question(resolved_query):
            return self._get_rag_service().answer(
                question,
                top_k=top_k,
                filters=normalized_filters,
                previous_query=previous_query,
            )

        started = perf_counter()
        mentioned_cities = sorted(city for city in ALLOWED_CITIES if city in resolved_query)
        if "city" not in normalized_filters:
            if len(mentioned_cities) > 1:
                raise ValueError("market analysis currently supports one city per request; set the city filter")
            if mentioned_cities:
                normalized_filters["city"] = mentioned_cities[0]
        focus_skills = extract_focus_skills(resolved_query)
        analysis = self._analyzer.analyze(
            filters=normalized_filters,
            focus_skills=focus_skills,
        )
        payload = analysis.to_dict()
        if analysis.sample_size == 0:
            return GroundedAnswer(
                question=cleaned_question,
                resolved_query=resolved_query,
                answer=f"{REFUSAL_TEXT} 当前筛选条件下没有符合统计口径的独立职位页。",
                refused=True,
                failure_type="market_sample_empty",
                latency_ms=(perf_counter() - started) * 1000,
                answer_mode="market_analysis",
                market_analysis=payload,
            )

        answer, citations = _render_market_answer(analysis, focus_skills)
        return GroundedAnswer(
            question=cleaned_question,
            resolved_query=resolved_query,
            answer=answer,
            refused=False,
            citations=citations,
            latency_ms=(perf_counter() - started) * 1000,
            answer_mode="market_analysis",
            market_analysis=payload,
        )


def is_market_analysis_question(question: str) -> bool:
    normalized = " ".join(question.lower().split())
    asks_for_specific_evidence = any(term in normalized for term in _EVIDENCE_QUESTION_TERMS)
    asks_for_statistics = any(term in normalized for term in _STATISTICAL_TERMS)
    if asks_for_specific_evidence and not asks_for_statistics:
        return False
    if any(term.lower() in normalized for term in _MARKET_TERMS):
        return True
    asks_for_skill_set = any(term in normalized for term in ("哪些技能", "什么技能", "哪些能力", "什么能力"))
    has_job_scope = any(term in normalized for term in ("岗位", "职位", "招聘"))
    return asks_for_skill_set and has_job_scope


def extract_focus_skills(question: str) -> tuple[str, ...]:
    normalized = question.casefold()
    found: list[str] = []
    for skill, aliases in _FOCUS_SKILL_ALIASES:
        if any(alias.casefold() in normalized for alias in aliases):
            found.append(skill)
    return tuple(found)


def _deduplicate_records(chunks: Sequence[_ChunkLike]) -> list[dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    for chunk in chunks:
        metadata = {key: str(value) for key, value in chunk.metadata.items()}
        path = metadata.get("path", "")
        record_id = metadata.get("record_id", "")
        record_key = (
            f"{path}#{record_id}" if path and record_id else chunk.source.split("#chunk-", 1)[0]
        )
        if record_key in records:
            existing_skills = tags(records[record_key])
            existing_skills.update(tags(metadata))
            records[record_key]["normalized_skills"] = ";".join(sorted(existing_skills))
            continue
        metadata.update(
            {
                "_record_key": record_key,
                "_chunk_id": chunk.chunk_id,
                "_source": chunk.source,
                "_text": chunk.text,
            }
        )
        records[record_key] = metadata
    return list(records.values())


def _matches_filters(record: Mapping[str, str], filters: Mapping[str, str]) -> bool:
    return all(record.get(key, "").casefold() == value.casefold() for key, value in filters.items())


def _frequency_tier(share: float) -> str:
    if share == 0:
        return "未检出"
    if share >= 0.5:
        return "高频"
    if share >= 0.2:
        return "中频"
    return "低频"


def _representative_evidence(
    records: Sequence[Mapping[str, str]],
    skill: str,
) -> tuple[JobEvidence, ...]:
    for record in records:
        if skill not in tags(dict(record)):
            continue
        return (_job_evidence(record),)
    return ()


def _job_evidence(record: Mapping[str, str]) -> JobEvidence:
    excerpt = record.get("requirements_summary", "") or record.get("_text", "")
    return JobEvidence(
        record_key=record.get("_record_key", ""),
        chunk_id=record.get("_chunk_id", ""),
        source=record.get("_source", ""),
        source_url=record.get("source_url", ""),
        job_title=record.get("job_title", ""),
        company=record.get("company", ""),
        city=record.get("city", ""),
        excerpt=" ".join(excerpt.split())[:260],
    )


def _has_valid_source_url(record: Mapping[str, str]) -> bool:
    if record.get("verification_status") != "public_page_verified":
        return False
    try:
        url = urlsplit(record.get("source_url", ""))
    except ValueError:
        return False
    if url.scheme not in {"http", "https"} or not url.netloc:
        return False
    return True


def _render_market_answer(
    analysis: MarketAnalysis,
    focus_skills: Sequence[str],
    *,
    display_limit: int = 12,
) -> tuple[str, tuple[Citation, ...]]:
    by_name = {item.skill: item for item in analysis.skills}
    displayed = list(analysis.skills[:display_limit])
    displayed_names = {item.skill for item in displayed}
    for skill in focus_skills:
        if skill in by_name and skill not in displayed_names:
            displayed.append(by_name[skill])
            displayed_names.add(skill)

    citation_numbers: dict[str, int] = {}
    citations: list[Citation] = []
    lines = [
        (
            f"统计口径：当前 {analysis.retrieval_corpus_jobs} 条检索语料中，仅使用 "
            f"{analysis.sample_size} 条“公开独立职位页 + 应用/Agent/RAG 岗位”，并按岗位去重；"
            "Top-K 不参与频次计算。"
        ),
        "筛选条件：" + ("；".join(f"{key}={value}" for key, value in analysis.filters.items()) or "未指定城市等额外条件"),
        "",
        "技能频次（岗位数/统计样本）：",
    ]
    for item in displayed:
        citation_suffix = ""
        if item.evidence:
            evidence = item.evidence[0]
            if evidence.record_key not in citation_numbers:
                number = len(citations) + 1
                citation_numbers[evidence.record_key] = number
                citations.append(
                    Citation(
                        number=number,
                        chunk_id=evidence.chunk_id,
                        source=evidence.source,
                        source_url=evidence.source_url,
                    )
                )
            citation_suffix = f" [{citation_numbers[evidence.record_key]}]"
        lines.append(
            f"- {item.frequency_tier}｜{item.skill}：{item.jobs}/{analysis.sample_size} "
            f"({item.share:.1%}){citation_suffix}"
        )
    lines.extend(
        [
            "",
            "分层规则：高频 ≥ 50%，中频 ≥ 20% 且 < 50%，低频 > 0 且 < 20%；0 次为未检出。",
            "重要边界：频次不等于硬性要求、优先项或加分项；当前数据没有逐项保留这些要求语气。",
            "这是有来源偏差的本地样本，不是全市场普查；标签未检出不代表业内不用。",
            "方括号仅指代表岗位；完整样本名单、技能列表与每项计数名单位于 market_analysis 结构化字段中。",
        ]
    )
    return "\n".join(lines), tuple(citations)
