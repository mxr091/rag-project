"""Document loading, deterministic chunking, and JSONL export utilities."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Document:
    source: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    text: str
    start: int
    end: int
    metadata: dict[str, str] = field(default_factory=dict)


def load_text_documents(directory: str | Path) -> list[Document]:
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(root)
    documents: list[Document] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}:
            documents.append(
                Document(
                    source=path.name,
                    text=path.read_text(encoding="utf-8"),
                    metadata={"path": str(path), "suffix": path.suffix.lower()},
                )
            )
    return documents


def load_job_csv_documents(csv_path: str | Path) -> list[Document]:
    """Convert one job row from the project's CSV into one source document."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() != ".csv":
        raise ValueError("job data must be a CSV file")

    documents: list[Document] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row_number, row in enumerate(csv.DictReader(file), start=2):
            values = {key: (value or "").strip() for key, value in row.items()}
            record_id = values.get("id") or str(row_number)
            fields = (
                ("岗位名称", values.get("job_title", "")),
                ("公司", values.get("company", "")),
                ("城市", values.get("city", "")),
                ("岗位类别", values.get("category", "")),
                ("薪资", values.get("salary", "")),
                ("经验要求", values.get("experience", "")),
                ("技能要求", values.get("skills_summary", "")),
                ("学历要求", values.get("education", "")),
                ("职责摘要", values.get("duties_summary", "")),
                ("任职要求摘要", values.get("requirements_summary", "")),
                ("标准化技能", values.get("normalized_skills", "")),
            )
            text = "\n".join(f"{label}：{value}" for label, value in fields if value)
            if not text:
                raise ValueError(f"job row {row_number} has no usable text")

            metadata = {
                "path": str(path),
                "suffix": ".csv",
                "record_id": record_id,
                "city": values.get("city", ""),
                "category": values.get("category", ""),
                "job_title": values.get("job_title", ""),
                "company": values.get("company", ""),
                "salary": values.get("salary", ""),
                "experience": values.get("experience", ""),
                "source_url": values.get("source_url", ""),
                "source_type": values.get("source_type", ""),
                "verification_note": values.get("verification_note", ""),
                "education": values.get("education", ""),
                "duties_summary": values.get("duties_summary", ""),
                "requirements_summary": values.get("requirements_summary", ""),
                "normalized_skills": values.get("normalized_skills", ""),
                "collected_at": values.get("collected_at", ""),
                "published_at": values.get("published_at", ""),
                "verification_status": values.get("verification_status", ""),
            }
            documents.append(
                Document(
                    source=f"{path.name}#job-{record_id}",
                    text=text,
                    metadata=metadata,
                )
            )
    return documents


def chunk_document(document: Document, chunk_size: int = 500, overlap: int = 80) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1")

    text = " ".join(document.text.split())
    chunks: list[Chunk] = []
    start = 0
    index = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        content = text[start:end].strip()
        if content:
            chunks.append(
                Chunk(
                    chunk_id=f"{document.source}#chunk-{index}",
                    source=document.source,
                    text=content,
                    start=start,
                    end=end,
                    metadata=document.metadata,
                )
            )
            index += 1
        if end == len(text):
            break
        start = end - overlap
    return chunks


def write_chunks_jsonl(chunks: list[Chunk], output_path: str | Path) -> Path:
    """Write validated chunks as one JSON object per line."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    seen_chunk_ids: set[str] = set()

    with path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            if not chunk.chunk_id:
                raise ValueError("chunk_id must not be empty")
            if chunk.chunk_id in seen_chunk_ids:
                raise ValueError(f"duplicate chunk_id: {chunk.chunk_id}")
            if not chunk.source:
                raise ValueError(f"chunk {chunk.chunk_id} has no source")
            if not chunk.text.strip():
                raise ValueError(f"chunk {chunk.chunk_id} has no text")

            seen_chunk_ids.add(chunk.chunk_id)
            record = {
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "text": chunk.text,
                "start": chunk.start,
                "end": chunk.end,
                "metadata": chunk.metadata,
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    return path
