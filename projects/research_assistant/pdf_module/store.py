"""Content-addressed local storage; single-process publication, atomic file replacement."""
from __future__ import annotations

from hashlib import sha256
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
from threading import RLock
from uuid import uuid4

from .parser import PdfError, REVISION, parse_pdf
from .chunks import CHUNK_VERSION, build_chunks


def safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise PdfError("invalid_document_id")
    return value


def safe_filename(value):
    name = str(value).replace("\\", "/").rsplit("/", 1)[-1]
    name = re.sub(r'[\x00-\x1f\x7f<>:"|?*]', "_", name)[:150]
    return name if name.lower().endswith(".pdf") else "document.pdf"


class PdfStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = RLock()

    def _write(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + "." + uuid4().hex + ".tmp")
        try:
            with tmp.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

    def _json(self, path, obj):
        self._write(path, json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8"))

    def ingest(self, data, filename="document.pdf"):
        doc_id = sha256(data).hexdigest()
        with self.lock:
            if (self.root / doc_id / f"{REVISION}.json").exists():
                existing = self.get(doc_id)
                if existing.get("chunk_version") == CHUNK_VERSION:
                    return existing, True
        document = parse_pdf(data, safe_filename(filename))
        chunks = build_chunks(document)
        document.update(chunk_version=CHUNK_VERSION, chunk_count=len(chunks))
        chunk_bytes = "".join(json.dumps(asdict(c), ensure_ascii=False) + "\n" for c in chunks).encode("utf-8")
        with self.lock:
            if (self.root / doc_id / f"{REVISION}.json").exists():
                existing = self.get(doc_id)
                if existing.get("chunk_version") == CHUNK_VERSION:
                    return existing, True
            self._write(self.root / doc_id / "source.pdf", data)
            self._write(self.root / doc_id / f"chunks-{REVISION}-{CHUNK_VERSION}.jsonl", chunk_bytes)
            self._json(self.root / doc_id / f"{REVISION}.json", document)
        return document, False

    def get(self, doc_id):
        doc_id = safe_id(doc_id)
        try:
            result = json.loads((self.root / doc_id / f"{REVISION}.json").read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise PdfError("document_not_found") from exc
        if result.get("id") != doc_id or result.get("revision") != REVISION:
            raise PdfError("document_integrity_error")
        return result

    def source(self, doc_id):
        self.get(doc_id)
        path = self.root / doc_id / "source.pdf"
        if not path.exists() or sha256(path.read_bytes()).hexdigest() != doc_id:
            raise PdfError("document_integrity_error")
        return path

    def list(self):
        return [summary(self.get(path.parent.name))
                for path in sorted(self.root.glob(f"*/{REVISION}.json"))
                if re.fullmatch(r"[a-f0-9]{64}", path.parent.name)]

    def save_run(self, result):
        run_id = uuid4().hex
        result = json.loads(json.dumps({**result, "run_id": run_id}, ensure_ascii=False))
        with self.lock:
            self._json(self.root / "runs" / (run_id + ".json"), result)
        return result

    def get_run(self, run_id):
        if not re.fullmatch(r"[a-f0-9]{32}", run_id):
            raise PdfError("invalid_run_id")
        try:
            return json.loads((self.root / "runs" / (run_id + ".json")).read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise PdfError("run_not_found") from exc


def summary(document):
    return {key: value for key, value in document.items() if key != "pages"}
