"""Import the existing historical corpus without upgrading source verification."""
import json
from pathlib import Path

from .contracts import JobInput


def load_corpus(path):
    jobs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        meta = row["metadata"]
        status = meta.get("verification_status", "")
        grade = {"public_page_verified": "independent",
                 "aggregate_page_verified": "aggregate"}.get(status, "legacy" if not status else "unknown")
        jobs.append(JobInput(
            source_key=row["source"], source_url=meta.get("source_url", ""),
            title=meta.get("job_title") or "未标注", company=meta.get("company") or "未标注",
            city=meta.get("city") or "未标注", text=row["text"],
            skills=[s for s in meta.get("normalized_skills", "").split(";") if s.strip()],
            experience_text=meta.get("experience", ""), education_text=meta.get("education", ""),
            source_grade=grade, collected_at=meta.get("collected_at", ""), checked_at=None))
    return jobs
