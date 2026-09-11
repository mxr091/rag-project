"""Input bounds and explicit source/proficiency boundaries."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from urllib.parse import urlsplit
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class JobInput(Contract):
    source_key: str = Field(min_length=1, max_length=512)
    source_url: str = Field(max_length=2048)
    title: str = Field(min_length=1, max_length=300)
    company: str = Field(min_length=1, max_length=300)
    city: str = Field(min_length=1, max_length=100)
    text: str = Field(min_length=1, max_length=20000)
    skills: list[str] = Field(default_factory=list, max_length=100)
    experience_text: str = Field(default="", max_length=300)
    education_text: str = Field(default="", max_length=300)
    source_grade: Literal["independent", "aggregate", "legacy", "unknown"] = "unknown"
    collected_at: str = Field(default="", max_length=100)
    checked_at: datetime | None = None

    @field_validator("source_url")
    @classmethod
    def valid_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("source_url must be a public-style HTTP(S) URL without credentials")
        return value

    @field_validator("skills")
    @classmethod
    def normalize_skills(cls, values: list[str]) -> list[str]:
        if any(not v.strip() or len(v.strip()) > 80 for v in values):
            raise ValueError("skills must contain nonempty strings of at most 80 characters")
        return sorted(set(v.strip() for v in values), key=lambda value: (value.casefold(), value))

    @field_validator("checked_at")
    @classmethod
    def checked_time(cls, value: datetime | None) -> datetime | None:
        if value is not None:
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("checked_at requires a timezone")
            if value > datetime.now(timezone.utc):
                raise ValueError("checked_at cannot be in the future")
        return value


class ImportRequest(Contract):
    jobs: list[JobInput] = Field(min_length=1, max_length=250)


class Capability(Contract):
    skill: str = Field(min_length=1, max_length=80)
    evidence: str = Field(min_length=1, max_length=300)
    source: str = Field(min_length=1, max_length=300)


class CandidateProfile(Contract):
    cities: list[str] = Field(min_length=1, max_length=5)
    graduation_year: int = Field(ge=2000, le=2100)
    work_experience_months: int = Field(ge=0, le=600)
    capabilities: list[Capability] = Field(min_length=1, max_length=20)

    @field_validator("capabilities")
    @classmethod
    def distinct_capabilities(cls, values: list[Capability]) -> list[Capability]:
        if len({v.skill.casefold() for v in values}) != len(values):
            raise ValueError("capabilities must have distinct skill names")
        return values

    @field_validator("cities")
    @classmethod
    def cities_valid(cls, values: list[str]) -> list[str]:
        if any(not v.strip() or len(v) > 100 for v in values):
            raise ValueError("invalid city")
        return list(dict.fromkeys(v.strip() for v in values))


class EvaluationRequest(Contract):
    snapshot_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    profile: CandidateProfile


class ReviewRequest(Contract):
    expected_version: int = Field(ge=0)
    decision: Literal["consider", "skip", "needs_confirmation"]
    note: str = Field(default="", max_length=2000)
