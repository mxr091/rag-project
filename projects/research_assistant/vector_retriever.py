"""U6 embedding retrieval: persistent vectors, metadata filtering, and retrieval records."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from document import Chunk

FILTERABLE_METADATA_FIELDS = frozenset({"city", "category", "job_title", "company"})


class EmbeddingModel(Protocol):
    name: str

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    """Real local multilingual embeddings; import is delayed until use."""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "Install requirements-embedding.txt before using semantic embeddings."
            ) from error
        self.name = f"sentence-transformers:{model_name}"
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return [[float(value) for value in vector] for vector in vectors]


@dataclass(frozen=True)
class VectorSearchResult:
    chunk: Chunk
    score: float


def load_chunks_jsonl(input_path: str | Path) -> list[Chunk]:
    """Load U5 JSONL chunks while preserving text, source and metadata."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(path)

    chunks: list[Chunk] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                chunk = Chunk(
                    chunk_id=record["chunk_id"],
                    source=record["source"],
                    text=record["text"],
                    start=record["start"],
                    end=record["end"],
                    metadata=record["metadata"],
                )
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid chunk JSONL at {path}:{line_number}") from error
            if not chunk.chunk_id or not chunk.source or not chunk.text.strip():
                raise ValueError(f"invalid chunk fields at {path}:{line_number}")
            if chunk.chunk_id in seen_ids:
                raise ValueError(f"duplicate chunk_id in {path}: {chunk.chunk_id}")
            if not isinstance(chunk.metadata, dict) or not all(
                isinstance(key, str) and isinstance(value, str) for key, value in chunk.metadata.items()
            ):
                raise ValueError(f"invalid metadata at {path}:{line_number}")
            seen_ids.add(chunk.chunk_id)
            chunks.append(chunk)
    return chunks


class JsonVectorStore:
    """Small local vector index for U6, persisted as JSON under data/index/."""

    def __init__(self, index_path: str | Path) -> None:
        self.path = Path(index_path)
        self.embedding_model: str | None = None
        self._records: dict[str, tuple[Chunk, tuple[float, ...]]] = {}
        if self.path.exists():
            self._load()

    @property
    def count(self) -> int:
        return len(self._records)

    def upsert_chunks(self, chunks: Sequence[Chunk], embedder: EmbeddingModel) -> int:
        """Embed only new or changed chunks and replace same-ID records."""
        model_name = _model_name(embedder)
        if self.embedding_model is not None and self.embedding_model != model_name:
            raise ValueError("embedding model changed; use rebuild() instead of mixing vectors")
        prepared = _prepare_records(chunks, embedder)
        if not prepared:
            return 0
        self._records.update(prepared)
        self.embedding_model = model_name
        self._save()
        return len(prepared)

    def rebuild(self, chunks: Sequence[Chunk], embedder: EmbeddingModel) -> int:
        """Recreate every vector after a model or chunking-rule change."""
        self._records = _prepare_records(chunks, embedder)
        self.embedding_model = _model_name(embedder)
        self._save()
        return self.count

    def search(
        self,
        query: str,
        *,
        embedder: EmbeddingModel,
        top_k: int = 3,
        filters: Mapping[str, str] | None = None,
    ) -> list[VectorSearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not query.strip() or not self._records:
            return []
        if self.embedding_model != _model_name(embedder):
            raise ValueError("query embedder must match the index embedding model")

        normalized_filters = _validate_filters(filters)
        query_vector = _validate_vector(embedder.embed_query(query))
        results = [
            VectorSearchResult(chunk=chunk, score=cosine_similarity(query_vector, vector))
            for chunk, vector in self._records.values()
            if _matches_filters(chunk.metadata, normalized_filters)
        ]
        results.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
        return results[:top_k]

    def _load(self) -> None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid vector index JSON: {self.path}") from error
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError(f"unsupported vector index format: {self.path}")
        model_name, records = payload.get("embedding_model"), payload.get("records")
        if not isinstance(model_name, str) or not isinstance(records, list):
            raise ValueError(f"invalid vector index structure: {self.path}")

        loaded: dict[str, tuple[Chunk, tuple[float, ...]]] = {}
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get("chunk"), dict):
                raise ValueError(f"invalid vector record in {self.path}")
            data = record["chunk"]
            try:
                chunk = Chunk(
                    chunk_id=data["chunk_id"], source=data["source"], text=data["text"],
                    start=data["start"], end=data["end"], metadata=data["metadata"],
                )
                vector = _validate_vector(record["vector"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid vector record in {self.path}") from error
            if chunk.chunk_id in loaded:
                raise ValueError(f"duplicate chunk_id in vector index: {chunk.chunk_id}")
            loaded[chunk.chunk_id] = (chunk, vector)
        self.embedding_model = model_name
        self._records = loaded

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        records = []
        for chunk_id in sorted(self._records):
            chunk, vector = self._records[chunk_id]
            records.append(
                {
                    "chunk": {
                        "chunk_id": chunk.chunk_id, "source": chunk.source, "text": chunk.text,
                        "start": chunk.start, "end": chunk.end, "metadata": chunk.metadata,
                    },
                    "vector": list(vector),
                }
            )
        payload = {"schema_version": 1, "embedding_model": self.embedding_model, "records": records}
        temporary_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary_path.replace(self.path)


def append_retrieval_record(
    output_path: str | Path,
    *,
    question: str,
    filters: Mapping[str, str] | None,
    top_k: int,
    results: Sequence[VectorSearchResult],
    observation: str,
) -> Path:
    """Record a U6 query and manual observation; later chapters evaluate it formally."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "question": question,
        "filters": _validate_filters(filters),
        "top_k": top_k,
        "results": [
            {
                "chunk_id": result.chunk.chunk_id,
                "source": result.chunk.source,
                "score": round(result.score, 6),
                "city": result.chunk.metadata.get("city", ""),
                "category": result.chunk.metadata.get("category", ""),
            }
            for result in results
        ],
        "observation": observation,
    }
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return path


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("vectors must be non-empty and share one dimension")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("vectors must not have zero magnitude")
    return dot / (left_norm * right_norm)


def _prepare_records(chunks: Sequence[Chunk], embedder: EmbeddingModel) -> dict[str, tuple[Chunk, tuple[float, ...]]]:
    materialized = list(chunks)
    vectors = embedder.embed_documents([chunk.text for chunk in materialized])
    if len(vectors) != len(materialized):
        raise ValueError("embedder returned a different vector count than input chunks")
    prepared: dict[str, tuple[Chunk, tuple[float, ...]]] = {}
    for chunk, vector in zip(materialized, vectors, strict=True):
        if chunk.chunk_id in prepared:
            raise ValueError(f"duplicate chunk_id: {chunk.chunk_id}")
        prepared[chunk.chunk_id] = (chunk, _validate_vector(vector))
    return prepared


def _model_name(embedder: EmbeddingModel) -> str:
    return str(getattr(embedder, "name", type(embedder).__name__))


def _validate_vector(vector: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in vector)
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("embedding must contain finite numeric values")
    return values


def _validate_filters(filters: Mapping[str, str] | None) -> dict[str, str]:
    if filters is None:
        return {}
    normalized: dict[str, str] = {}
    for field, value in filters.items():
        if field not in FILTERABLE_METADATA_FIELDS:
            allowed = ", ".join(sorted(FILTERABLE_METADATA_FIELDS))
            raise ValueError(f"unsupported metadata filter: {field}; allowed: {allowed}")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"metadata filter {field} must be a non-empty string")
        normalized[field] = value.strip().casefold()
    return normalized


def _matches_filters(metadata: Mapping[str, str], filters: Mapping[str, str]) -> bool:
    return all(metadata.get(field, "").strip().casefold() == value for field, value in filters.items())

