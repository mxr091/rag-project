"""Page-scoped evidence chunks with physical table geometry and neighboring context."""
from __future__ import annotations

import json
from document import Chunk
from .parser import PdfError

CHUNK_VERSION = "page-evidence-v1"


def build_chunks(document, pages=None):
    chunks = []
    for page in document["pages"]:
        if page["status"] != "ready" or (pages and page["number"] not in pages):
            continue
        blocks = {b["id"]: b for b in page["blocks"]}
        heading = ""
        for block in page["blocks"]:
            if block["kind"] == "heading":
                heading = block["text"]
            ids = [block["id"]]
            texts = [block["text"]]
            if block["kind"] == "table":
                ids += block["context_ids"]
                context = "\n".join(blocks[key]["text"] for key in block["context_ids"])
                # Small tables remain whole. Large tables repeat the first two raw header rows.
                # Header detection is deliberately not presented as semantic schema inference.
                table_rows = [" | ".join(value if value is not None else "[未识别单元格]" for value in row)
                              for row in block["matrix"]]
                if len(block["text"]) > 2200:
                    texts = ["\n".join(table_rows[:2] + [row]) for row in table_rows[2:]]
                texts = ["表格（按物理合并单元格展开）\n" + t +
                         "\n同页邻近说明（适用关系需阅读确认）\n" + context for t in texts]
            elif heading and heading != block["text"]:
                texts = [heading + "\n" + block["text"]]
                # Heading text is provenance too, not an invented chunk label.
                ids += [b["id"] for b in page["blocks"] if b["text"] == heading][:1]
            for part, text in enumerate(texts):
                if not text.strip():
                    continue
                # Never silently truncate evidence or detach a table's conditions.
                if len(text) > 12000:
                    raise PdfError("evidence_chunk_too_large")
                evidence_blocks = [{"id": key, "kind": blocks[key]["kind"],
                                    "bbox": blocks[key]["bbox"], "text": blocks[key]["text"]}
                                   for key in dict.fromkeys(ids)]
                metadata = {"doc_id": document["id"], "revision": document["revision"],
                            "pdf_page": str(page["number"]), "printed_page": page["printed_page"] or "",
                            "block_kind": block["kind"], "chunk_version": CHUNK_VERSION,
                            "evidence_blocks": json.dumps(evidence_blocks, ensure_ascii=False),
                            "source_url": f"/v1/documents/{document['id']}/source#page={page['number']}"}
                chunks.append(Chunk(f"{document['id']}-{document['revision']}-{CHUNK_VERSION}-p{page['number']}-{block['order']}-{part}",
                                    f"{document['filename']} | PDF 第 {page['number']} 页", text, 0, len(text), metadata))
    return chunks
