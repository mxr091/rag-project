"""A selected immutable snapshot is the only job evidence for an assessment."""
from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
import sys
from threading import BoundedSemaphore

from document import Chunk
from grounded_service import ExtractiveGroundedGenerator, GroundedRAGService, ModelGroundedGenerator
from vector_retriever import VectorSearchResult

from .repository import digest


class CapacityExceeded(Exception):
    pass


class SnapshotRetriever:
    def __init__(self, snapshot):
        text = snapshot["text"]
        self.result = VectorSearchResult(chunk=Chunk(
            chunk_id=snapshot["id"], source=snapshot["source_key"], text=text,
            start=0, end=len(text), metadata={"city": snapshot["city"],
                                           "source_url": snapshot["source_url"]}), score=1.0)

    def search(self, query, *, top_k=3, filters=None):
        return [self.result]


def environment_generator():
    mode = os.getenv("BACKEND_GENERATOR_MODE", "extractive")
    if mode == "extractive":
        return ExtractiveGroundedGenerator, {"mode": mode, "assessment_version": 2}
    if mode != "model":
        raise ValueError("BACKEND_GENERATOR_MODE must be extractive or model")
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "experiments" /
                           "02-openai-compatible-adapter"))
    from openai_compatible import OpenAICompatibleModel
    key, endpoint, name = (os.getenv(k, "") for k in ("MODEL_API_KEY", "MODEL_ENDPOINT", "MODEL_NAME"))
    if not all((key, endpoint, name)):
        raise ValueError("model configuration incomplete")
    # A fresh generator holds usage for one request. No implicit retries or unbounded output.
    thinking_type = os.getenv("MODEL_THINKING_TYPE") or None
    model = OpenAICompatibleModel(endpoint=endpoint, api_key=key, model=name,
                                  timeout_seconds=30, max_retries=0, max_tokens=900,
                                  thinking_type=thinking_type)
    return lambda: ModelGroundedGenerator(model), {
        "mode": mode, "model": name, "endpoint_hash": digest(endpoint),
        "max_tokens": 900, "timeout_seconds": 30, "max_retries": 0,
        "thinking_type": thinking_type, "assessment_version": 2}


def assess(snapshot, profile, generator):
    declared = {c.skill.casefold(): c for c in profile.capabilities}
    mentioned = {s.casefold(): s for s in snapshot["skills"]}
    supported = [{"skill": mentioned[key], "evidence": declared[key].evidence,
                  "source": declared[key].source} for key in sorted(mentioned.keys() & declared.keys())]
    missing = [mentioned[key] for key in sorted(mentioned.keys() - declared.keys())]
    # Only ask the model to explain the job: profile comparison below is deterministic.
    # User-supplied capability claims are retained, not promoted to verified employment history.
    answer = GroundedRAGService(SnapshotRetriever(snapshot), generator).answer(
        "只提取这一个岗位证据中明确写出的职责和技能，分两点简述，总计不超过200字，每点附[1]。"
        "沿用原文要求的语气，不把技能标签变成硬性要求。未提到的条件直接省略；"
        "毕业届别、工作经验是否匹配和当前是否招聘由后端另行标记，本次无需判断。", top_k=1)
    return {"snapshot_id": snapshot["id"], "generator": asdict(answer),
            "comparison": {"city_matches": snapshot["city"] in profile.cities,
                           "supported_mentions": supported, "mentions_without_profile_evidence": missing,
                           "skill_match_method": "exact_casefolded_tag_overlap_not_eligibility_score",
                           "experience_requirement_text": snapshot["experience_text"],
                           "education_requirement_text": snapshot["education_text"],
                           "graduation_eligibility": "needs_confirmation",
                           "experience_eligibility": "needs_confirmation",
                           "currently_hiring": "unverified"},
            "source": {k: snapshot[k] for k in ("source_grade", "collected_at", "checked_at", "source_url")},
            "limits": ["能力证据由请求方提供，未自动核验", "技能标签出现不等于必须条件",
                       "引用检查仅覆盖编号和范围，未验证语义蕴含", "历史快照不证明岗位仍开放"]}


class EvaluationService:
    def __init__(self, repository, generator_factory, generator_config, max_concurrency=2):
        self.repository = repository
        self.factory = generator_factory
        self.config = generator_config
        self.capacity = BoundedSemaphore(max_concurrency)

    def evaluate(self, key, request):
        payload = request.model_dump(mode="json") | {"generator_config": self.config}
        request_hash = digest(payload)
        prior = self.repository.lookup_request(key, request_hash)
        if prior:
            return prior, True
        snapshot = self.repository.get_snapshot(request.snapshot_id)
        if not self.capacity.acquire(blocking=False):
            raise CapacityExceeded("evaluation_capacity_exceeded")
        try:
            record, created = self.repository.start_evaluation(key, request_hash, payload)
            if not created:
                return record, True
            try:
                result = assess(snapshot, request.profile, self.factory())
            except Exception as exc:
                # No raw exception/API credential/provider body crosses the storage/API boundary.
                failure_type = "output_truncated" if getattr(exc, "failure_type", "") == "output_truncated" else "generation_failed"
                usage = getattr(exc, "usage", None)
                return self.repository.finish_evaluation(record["id"], failure_type=failure_type,
                    result={"model_usage": usage} if isinstance(usage, dict) else None), False
            return self.repository.finish_evaluation(record["id"], result=result), False
        finally:
            self.capacity.release()
