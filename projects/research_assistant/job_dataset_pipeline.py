"""Validated job-data ingestion, deduplication, corpus building, and quality reporting."""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from document import Document, chunk_document, load_job_csv_documents, write_chunks_jsonl

ALLOWED_CITIES = frozenset({"上海", "深圳", "广州"})
REQUIRED_FIELDS = (
    "id", "city", "category", "job_title", "company",
    "skills_summary", "source_url", "source_type",
)
QUALITY_FIELDS = REQUIRED_FIELDS + ("salary", "experience")
VERIFICATION_STATUSES = frozenset({"public_page_verified", "aggregate_page_verified", "legacy_aggregate"})
_VERIFICATION_RANK = {"legacy_aggregate": 1, "aggregate_page_verified": 2, "public_page_verified": 3}


def load_job_records(paths: Sequence[str | Path]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for input_path in paths:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames:
                raise ValueError(f"CSV has no header: {path}")
            missing_columns = [field for field in REQUIRED_FIELDS if field not in reader.fieldnames]
            if missing_columns:
                raise ValueError(f"CSV missing columns in {path}: {', '.join(missing_columns)}")
            for line_number, raw in enumerate(reader, start=2):
                record = {key: (value or "").strip() for key, value in raw.items()}
                _validate_record(record, path, line_number)
                record_id = record["id"]
                if record_id in seen_ids:
                    raise ValueError(f"duplicate job id across inputs: {record_id}")
                seen_ids.add(record_id)
                if not record.get("verification_status"):
                    record["verification_status"] = "legacy_aggregate"
                records.append(record)
    return records


def build_enriched_corpus(
    input_paths: Sequence[str | Path],
    *,
    chunks_path: str | Path,
    manifest_path: str | Path,
    chunk_size: int = 500,
    overlap: int = 80,
) -> dict[str, object]:
    raw_records = load_job_records(input_paths)
    raw_documents = [
        document
        for input_path in input_paths
        for document in load_job_csv_documents(input_path)
    ]
    if len(raw_records) != len(raw_documents):
        raise ValueError("record/document count mismatch")
    records, documents, superseded = _deduplicate(raw_records, raw_documents)
    chunks = [
        chunk
        for document in documents
        for chunk in chunk_document(document, chunk_size=chunk_size, overlap=overlap)
    ]
    write_chunks_jsonl(chunks, chunks_path)
    manifest = build_quality_manifest(
        records,
        chunk_count=len(chunks),
        input_paths=input_paths,
        input_record_count=len(raw_records),
        superseded_record_ids=superseded,
    )
    target = Path(manifest_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def build_quality_manifest(
    records: Sequence[dict[str, str]],
    *,
    chunk_count: int,
    input_paths: Sequence[str | Path],
    input_record_count: int | None = None,
    superseded_record_ids: Sequence[str] = (),
) -> dict[str, object]:
    if not records:
        raise ValueError("job dataset must not be empty")
    duplicate_keys = _duplicate_keys(records)
    source_urls = [record["source_url"] for record in records]
    independent = sum(record.get("verification_status") == "public_page_verified" for record in records)
    completeness = {
        field: round(sum(bool(record.get(field)) for record in records) / len(records), 4)
        for field in QUALITY_FIELDS
    }
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_files": [str(Path(path)) for path in input_paths],
        "input_records": input_record_count if input_record_count is not None else len(records),
        "records_after_deduplication": len(records),
        "superseded_record_ids": list(superseded_record_ids),
        "chunks": chunk_count,
        "independently_verified_records": independent,
        "aggregate_or_legacy_records": len(records) - independent,
        "unique_source_urls": len(set(source_urls)),
        "unique_source_url_ratio": round(len(set(source_urls)) / len(records), 4),
        "remaining_duplicate_record_keys": duplicate_keys,
        "remaining_duplicate_record_count": len(duplicate_keys),
        "city_counts": dict(sorted(Counter(record["city"] for record in records).items())),
        "category_counts": dict(sorted(Counter(record["category"] for record in records).items())),
        "source_type_counts": dict(sorted(Counter(record["source_type"] for record in records).items())),
        "verification_status_counts": dict(
            sorted(Counter(record.get("verification_status", "legacy_aggregate") for record in records).items())
        ),
        "field_completeness": completeness,
    }


def skill_frequencies(records: Sequence[dict[str, str]]) -> list[dict[str, object]]:
    counts: Counter[str] = Counter()
    for record in records:
        tags = {tag.strip() for tag in record.get("normalized_skills", "").split(";") if tag.strip()}
        counts.update(tags)
    total = len(records)
    return [
        {
            "skill": skill,
            "jobs": count,
            "share": round(count / total, 4),
            "confidence": "高" if count >= 5 else "中" if count >= 3 else "低",
        }
        for skill, count in sorted(counts.items(), key=lambda item: (-item[1], item[0].casefold()))
    ]


def _deduplicate(
    records: Sequence[dict[str, str]],
    documents: Sequence[Document],
) -> tuple[list[dict[str, str]], list[Document], list[str]]:
    selected: dict[tuple[str, str, str], tuple[dict[str, str], Document]] = {}
    superseded: list[str] = []
    for record, document in zip(records, documents, strict=True):
        key = (record["city"].casefold(), record["company"].casefold(), record["job_title"].casefold())
        current = selected.get(key)
        if current is None:
            selected[key] = (record, document)
            continue
        current_record = current[0]
        new_rank = _VERIFICATION_RANK[record["verification_status"]]
        current_rank = _VERIFICATION_RANK[current_record["verification_status"]]
        if new_rank > current_rank or (new_rank == current_rank and int(record["id"]) > int(current_record["id"])):
            superseded.append(current_record["id"])
            selected[key] = (record, document)
        else:
            superseded.append(record["id"])
    ordered = sorted(selected.values(), key=lambda pair: int(pair[0]["id"]))
    return [pair[0] for pair in ordered], [pair[1] for pair in ordered], sorted(superseded, key=int)


def _validate_record(record: dict[str, str], path: Path, line_number: int) -> None:
    missing = [field for field in REQUIRED_FIELDS if not record.get(field)]
    if missing:
        raise ValueError(f"missing values at {path}:{line_number}: {', '.join(missing)}")
    if not record["id"].isdigit():
        raise ValueError(f"job id must be numeric at {path}:{line_number}")
    if record["city"] not in ALLOWED_CITIES:
        raise ValueError(f"unsupported city at {path}:{line_number}: {record['city']}")
    parsed = urlparse(record["source_url"])
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"invalid source_url at {path}:{line_number}")
    status = record.get("verification_status")
    if status and status not in VERIFICATION_STATUSES:
        raise ValueError(f"invalid verification_status at {path}:{line_number}: {status}")


def _duplicate_keys(records: Sequence[dict[str, str]]) -> list[str]:
    counts: Counter[tuple[str, str, str]] = Counter(
        (record["city"].casefold(), record["company"].casefold(), record["job_title"].casefold())
        for record in records
    )
    return [" | ".join(key) for key, count in sorted(counts.items()) if count > 1]
