"""Small in-memory lexical retriever used as the BM25 baseline."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

from document import Chunk

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
FILTERABLE_METADATA_FIELDS = frozenset({"city", "category", "job_title", "company"})


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


class InMemoryRetriever:
    """BM25-style lexical baseline with deterministic ranking and metadata filters."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = list(chunks)
        self.term_frequencies = [Counter(tokenize(chunk.text)) for chunk in self.chunks]
        self.lengths = [sum(freq.values()) for freq in self.term_frequencies]
        self.avg_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        document_frequency: Counter[str] = Counter()
        for frequencies in self.term_frequencies:
            document_frequency.update(frequencies.keys())
        self.document_frequency = document_frequency

    def search(
        self,
        query: str,
        top_k: int = 3,
        *,
        filters: Mapping[str, str] | None = None,
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        normalized_filters = _validate_filters(filters)
        query_terms = set(tokenize(query))
        if not query_terms or not self.chunks:
            return []

        total = len(self.chunks)
        results: list[SearchResult] = []
        for index, frequencies in enumerate(self.term_frequencies):
            chunk = self.chunks[index]
            if not _matches_filters(chunk.metadata, normalized_filters):
                continue
            score = 0.0
            for term in query_terms:
                term_frequency = frequencies.get(term, 0)
                if not term_frequency:
                    continue
                document_frequency = self.document_frequency[term]
                idf = math.log(1 + (total - document_frequency + 0.5) / (document_frequency + 0.5))
                k1, b = 1.5, 0.75
                length_factor = 1 - b + b * self.lengths[index] / max(self.avg_length, 1.0)
                score += idf * (term_frequency * (k1 + 1)) / (term_frequency + k1 * length_factor)
            if score > 0:
                results.append(SearchResult(chunk, score))

        results.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
        return results[:top_k]


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
