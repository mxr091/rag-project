"""Lexical retrieval with literal technical identifiers/numbers ahead of generic words."""
import re
import unicodedata

from retriever import InMemoryRetriever, SearchResult
from .parser import compact

RETRIEVAL_VERSION = "literal-anchor-v1"


def anchors(query):
    identifiers = re.findall(r"[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*", query)
    identifiers = [value.lower() for value in identifiers if len(value) >= 3 and any(c.isdigit() for c in value)]
    numbers = re.findall(r"(?<![A-Za-z0-9.])\d+(?:\.\d+)?(?![A-Za-z0-9.])", query)
    return set(identifiers + numbers)


class PdfLexicalRetriever:
    def __init__(self, chunks):
        self.baseline = InMemoryRetriever(chunks)

    def search(self, query, *, top_k=3, filters=None):
        wanted = anchors(query)
        if not wanted:
            return self.baseline.search(query, top_k=top_k, filters=filters)
        candidates = self.baseline.search(query, top_k=max(len(self.baseline.chunks),1), filters=filters)
        scale = max((result.score for result in candidates), default=0) + 1
        ranked = []
        for result in candidates:
            text = unicodedata.normalize("NFKC", result.chunk.text).lower()
            hits = sum(bool(re.search(r"(?<![a-z0-9])"+re.escape(value)+r"(?![a-z0-9])", text)) for value in wanted)
            ranked.append(SearchResult(result.chunk, hits * scale + result.score))
        ranked.sort(key=lambda result:(-result.score,result.chunk.chunk_id))
        return ranked[:top_k]
