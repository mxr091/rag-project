"""Bounded native PDF parser. Coordinates are PDF points, origin at top left."""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
from statistics import median
import re
import unicodedata

import pdfplumber

SCHEMA = "native-pdf-v1"
MAX_BYTES = 25 * 1024 * 1024
MAX_PAGES = 150
MAX_CHARS_PER_PAGE = 50000
PARSER_VERSION = SCHEMA + ":pdfplumber-" + version("pdfplumber")
PARSER_CODE_SHA = sha256(Path(__file__).read_bytes()).hexdigest()
REVISION = sha256((PARSER_VERSION + PARSER_CODE_SHA).encode()).hexdigest()[:16]


class PdfError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def compact(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text or ""))


def box(items):
    return [round(min(x[0] for x in items), 2), round(min(x[1] for x in items), 2),
            round(max(x[2] for x in items), 2), round(max(x[3] for x in items), 2)]


def word_box(word):
    return (word["x0"], word["top"], word["x1"], word["bottom"])


def inside(inner, outer):
    x, y = (inner[0] + inner[2]) / 2, (inner[1] + inner[3]) / 2
    return outer[0] - 0.2 <= x <= outer[2] + 0.2 and outer[1] - 0.2 <= y <= outer[3] + 0.2


def quality_code(text):
    if not text.strip():
        return "no_native_text"
    bad = sum(unicodedata.category(c) in {"Co", "Cs", "Cn"} or c == "\ufffd" for c in text)
    if "(cid:" in text or bad / max(len(text), 1) > 0.02:
        return "unreliable_encoding"
    # This is an explicit heuristic, not language-independent OCR confidence.
    return None


def lines_from_words(words):
    lines = []
    for word in sorted(words, key=lambda w: (round(w["top"] / 3), w["x0"])):
        center = (word["top"] + word["bottom"]) / 2
        candidates = [line for line in lines[-4:] if abs(line["center"] - center) < 5]
        if candidates:
            candidates[-1]["words"].append(word)
        else:
            lines.append({"center": center, "words": [word]})
    return sorted(lines, key=lambda line: line["center"])


def line_record(words, column):
    words = sorted(words, key=lambda w: w["x0"])
    text = " ".join(w["text"] for w in words)
    text = re.sub(r"(?<=[\u4e00-\u9fff]) (?=[\u4e00-\u9fff])", "", text)
    return {"text": text, "bbox": box([word_box(w) for w in words]), "column": column,
            "font_size": median(w["bottom"] - w["top"] for w in words)}


def ordered_lines(words, width):
    """One/two column heuristic; spanning lines split independent column regions."""
    middle = width / 2
    pending = [[], []]
    result = []

    def flush():
        for column in pending:
            result.extend(column)
            column.clear()

    for group in lines_from_words(words):
        row = sorted(group["words"], key=lambda w: w["x0"])
        splits = [i for i in range(1, len(row))
                  if row[i]["x0"] - row[i - 1]["x1"] >= 14
                  and row[i - 1]["x1"] < middle + width * .08
                  and row[i]["x0"] > middle - width * .08]
        if splits:
            cut = min(splits, key=lambda i: abs((row[i]["x0"] + row[i-1]["x1"]) / 2 - middle))
            pending[0].append(line_record(row[:cut], "left"))
            pending[1].append(line_record(row[cut:], "right"))
        elif row[-1]["x1"] < middle + 6:
            pending[0].append(line_record(row, "left"))
        elif row[0]["x0"] > middle - 6:
            pending[1].append(line_record(row, "right"))
        else:
            flush()
            result.append(line_record(row, "full"))
    flush()
    return result


def paragraphs(words, width):
    result = []
    for line in ordered_lines(words, width):
        heading = bool(re.match(r"^\d+(?:\.\d+)*\s+\D", line["text"])) and line["font_size"] >= 14
        footnote = bool(re.match(r"^\d+\s+\D", line["text"])) and not heading
        new_item = bool(re.match(r"^[•●>]|^\d+\s", line["text"]))
        prev = result[-1] if result else None
        same_flow = prev and (prev["column"] == line["column"] or (
            "full" in {prev["column"], line["column"]} and abs(prev["bbox"][0]-line["bbox"][0]) < 4))
        can_join = (prev and same_flow and prev["kind"] in {"paragraph", "footnote"}
                    and not heading and not new_item and 0 <= line["bbox"][1] - prev["bbox"][3] < 8
                    and abs(prev["font_size"] - line["font_size"]) < 2 and len(prev["text"]) < 1200)
        if can_join:
            prev["text"] += "\n" + line["text"]
            prev["bbox"] = box([prev["bbox"], line["bbox"]])
        else:
            result.append({**line, "kind": "heading" if heading else "footnote" if footnote else "paragraph"})
    return result


def extract_table(table, page):
    """Recover merged cells from rectangle coverage, never from forward-filling nulls."""
    raw = table.extract(x_tolerance=2, y_tolerance=3)
    xs = sorted({b[0] for b in table.cells} | {b[2] for b in table.cells})
    ys = sorted({b[1] for b in table.cells} | {b[3] for b in table.cells})
    values = {}
    for row, values_row in zip(table.rows, raw):
        for bounds, text in zip(row.cells, values_row):
            if bounds is not None:
                values[bounds] = text or ""
    cells = []
    for bounds in table.cells:
        r0, r1 = ys.index(bounds[1]), ys.index(bounds[3])
        c0, c1 = xs.index(bounds[0]), xs.index(bounds[2])
        cells.append({"row": r0, "column": c0, "row_span": r1-r0, "column_span": c1-c0,
                      "text": values.get(bounds, ""), "bbox": [round(v, 2) for v in bounds]})
    matrix = [[None for _ in range(len(xs)-1)] for _ in range(len(ys)-1)]
    for cell in cells:
        for r in range(cell["row"], cell["row"] + cell["row_span"]):
            for c in range(cell["column"], cell["column"] + cell["column_span"]):
                matrix[r][c] = cell["text"]
    text = "\n".join(" | ".join(value if value is not None else "[未识别单元格]" for value in row)
                     for row in matrix)
    return {"kind": "table", "text": text, "bbox": [round(v, 2) for v in table.bbox],
            "matrix": matrix, "cells": cells, "row_count": len(matrix),
            "column_count": len(xs)-1, "context_ids": [], "context_relation": "same_page_proximity"}


def parse_pdf(data: bytes, filename="document.pdf"):
    if not data.startswith(b"%PDF-"):
        raise PdfError("invalid_pdf")
    if len(data) > MAX_BYTES:
        raise PdfError("file_too_large")
    doc_id = sha256(data).hexdigest()
    pages = []
    try:
        with pdfplumber.open(BytesIO(data), unicode_norm="NFKC") as pdf:
            if not 1 <= len(pdf.pages) <= MAX_PAGES:
                raise PdfError("page_limit")
            for page in pdf.pages:
                if len(page.chars) > MAX_CHARS_PER_PAGE:
                    raise PdfError("page_character_limit")
                text = page.extract_text() or ""
                code = quality_code(text)
                if code == "no_native_text" and page.images:
                    code = "ocr_required"
                words = page.extract_words(x_tolerance=2, y_tolerance=3)
                footer = [w["text"] for w in words if w["top"] > page.height * .94
                          and page.width*.4 < w["x0"] < page.width*.6 and re.fullmatch(r"\d+", w["text"])]
                blocks = []
                if code is None:
                    # Marginal text remains in raw PDF; omit header/footer from retrieval content.
                    words = [w for w in words if page.height*.045 < w["top"] < page.height*.94]
                    tables = sorted([t for t in page.find_tables() if len(t.rows) >= 2
                                     and len(t.rows[0].cells) >= 2], key=lambda t: t.bbox[1])
                    remaining = [w for w in words if not any(inside(word_box(w), t.bbox) for t in tables)]
                    previous = -1.0
                    for table in tables:
                        section = [w for w in remaining if previous <= w["top"] < table.bbox[1]]
                        blocks.extend(paragraphs(section, page.width))
                        blocks.append(extract_table(table, page))
                        previous = table.bbox[3]
                    blocks.extend(paragraphs([w for w in remaining if w["top"] >= previous], page.width))
                    for order, block in enumerate(blocks):
                        block["id"] = f"{doc_id[:12]}-p{page.page_number}-b{order}"
                        block["order"] = order
                    for i, block in enumerate(blocks):
                        if block["kind"] == "table":
                            # Preserve context, but don't claim that every neighboring sentence is a footnote.
                            block["context_ids"] = [b["id"] for b in blocks
                                if b["kind"] != "table" and b["bbox"][1] < block["bbox"][3] + 155
                                and b["bbox"][3] > block["bbox"][1] - 160]
                if any(b["kind"] == "table" and any(len(c["text"]) > 1500 for c in b["cells"]) for b in blocks):
                    code = "complex_table_layout"
                if code is None and not blocks:
                    code = "no_body_content"
                pages.append({"number": page.page_number, "width": page.width, "height": page.height,
                              "printed_page": footer[0] if len(footer) == 1 else None,
                              "status": code or "ready", "blocks": blocks,
                              "warnings": ([code] if code else []) +
                              (["native_layout_heuristic_requires_review"] if blocks else [])})
                page.close()
    except PdfError:
        raise
    except Exception as exc:
        raise PdfError("unreadable_or_encrypted_pdf") from exc
    ready = sum(p["status"] == "ready" for p in pages)
    return {"schema": SCHEMA, "id": doc_id, "sha256": doc_id, "filename": filename,
            "parser": PARSER_VERSION, "revision": REVISION, "page_count": len(pages),
            "parser_code_sha256": PARSER_CODE_SHA,
            "status": "ready" if ready == len(pages) else "partial" if ready else "quarantined",
            "page_status_counts": dict(Counter(p["status"] for p in pages)), "pages": pages,
            "limits": ["native_text_only", "one_or_two_columns_heuristic", "ruled_tables_only",
                       "same_page_context_links", "no_cross_page_table_merge", "no_semantic_certification"]}
