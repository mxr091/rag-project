"""Single-operator PDF API. Launch with uvicorn pdf_api:create_app --factory."""
from __future__ import annotations

from contextlib import asynccontextmanager
import hmac
import os
from pathlib import Path
from threading import BoundedSemaphore

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, StrictInt
from starlette.concurrency import run_in_threadpool

from pdf_module.config import configured_model
from pdf_module.parser import MAX_BYTES, PdfError
from pdf_module.service import PdfService
from pdf_module.store import PdfStore, summary


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=500)
    document_ids: list[str] = Field(min_length=1, max_length=5)
    pages: list[StrictInt] | None = Field(default=None, max_length=150)
    top_k: StrictInt = Field(default=3, ge=1, le=5)
    strategy: str = "lexical"
    mode: str = "extractive"


def create_app(*, store=None, token=None, service=None):
    token = token or os.getenv("PDF_API_TOKEN", "")
    if len(token.encode()) < 32:
        raise ValueError("PDF_API_TOKEN must contain at least 32 bytes")
    root = Path(__file__).resolve().parents[2] / "output" / "pdf-runtime"
    store = store or PdfStore(os.getenv("PDF_STORE_DIR", str(root)))
    service = service or PdfService(store, model_factory=configured_model
                                   if os.getenv("PDF_ENABLE_MODEL") == "1" else None)
    app = FastAPI(title="PDF Evidence RAG", version="1.0", docs_url=None, redoc_url=None, openapi_url=None)
    slots = BoundedSemaphore(2)
    bearer = HTTPBearer(auto_error=False)

    def authenticate(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
        if credentials is None or not hmac.compare_digest(credentials.credentials.encode(), token.encode()):
            raise HTTPException(401, "unauthorized", headers={"WWW-Authenticate": "Bearer"})

    @app.exception_handler(PdfError)
    async def pdf_error(request, exc):
        status = 404 if exc.code.endswith("not_found") else 413 if exc.code in {
            "file_too_large", "page_limit", "page_character_limit", "evidence_chunk_too_large"} else 422
        return JSONResponse({"detail": exc.code}, status_code=status)

    @asynccontextmanager
    async def capacity():
        if not slots.acquire(blocking=False):
            raise HTTPException(429, "capacity_exceeded", headers={"Retry-After": "2"})
        try:
            yield
        finally:
            slots.release()

    @app.get("/health")
    def health():
        return {"status": "ok", "module": "pdf-evidence", "scope": "single_process_local"}

    @app.get("/openapi.json", dependencies=[Depends(authenticate)])
    def schema():
        return app.openapi()

    @app.post("/v1/documents", dependencies=[Depends(authenticate)])
    async def ingest(request: Request, filename: str = Query(default="document.pdf", max_length=150)):
        if request.headers.get("content-type", "").split(";")[0].strip() != "application/pdf":
            raise HTTPException(415, "application/pdf_required")
        async with capacity():
            body = bytearray()
            async for part in request.stream():
                if len(body) + len(part) > MAX_BYTES:
                    raise PdfError("file_too_large")
                body.extend(part)
            document, cached = await run_in_threadpool(store.ingest, bytes(body), filename)
            return {"document": summary(document), "cached": cached}

    @app.get("/v1/documents", dependencies=[Depends(authenticate)])
    def documents():
        return {"documents": store.list()}

    @app.get("/v1/documents/{doc_id}", dependencies=[Depends(authenticate)])
    def document(doc_id: str):
        return summary(store.get(doc_id))

    @app.get("/v1/documents/{doc_id}/pages/{page}", dependencies=[Depends(authenticate)])
    def page(doc_id: str, page: int):
        doc = store.get(doc_id)
        if not 1 <= page <= doc["page_count"]:
            raise PdfError("page_not_found")
        return doc["pages"][page-1]

    @app.get("/v1/documents/{doc_id}/source", dependencies=[Depends(authenticate)])
    def source(doc_id: str):
        return FileResponse(store.source(doc_id), media_type="application/pdf",
                            filename=store.get(doc_id)["filename"], content_disposition_type="inline")

    @app.post("/v1/pdf/ask", dependencies=[Depends(authenticate)])
    async def ask(payload: AskRequest):
        if not payload.question.strip():
            raise HTTPException(422, "question_required")
        async with capacity():
            return await run_in_threadpool(service.ask, **payload.model_dump())

    @app.get("/v1/pdf/runs/{run_id}", dependencies=[Depends(authenticate)])
    def run(run_id: str):
        return store.get_run(run_id)

    return app
