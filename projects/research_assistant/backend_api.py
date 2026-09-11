"""Authenticated persistent backend. Start with uvicorn backend_api:create_app --factory."""
from __future__ import annotations

from contextlib import asynccontextmanager
import os
import secrets
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from job_backend.contracts import EvaluationRequest, ImportRequest, ReviewRequest
from job_backend.database import make_engine, require_schema, sessions
from job_backend.evaluation import CapacityExceeded, EvaluationService, environment_generator
from job_backend.repository import Conflict, NotFound, Repository


def create_app(*, database_url=None, token=None, engine=None, generator_factory=None,
               generator_config=None, max_concurrency=2):
    token = token or os.getenv("BACKEND_API_TOKEN", "")
    if len(token) < 32 or not token.isascii():
        raise ValueError("BACKEND_API_TOKEN requires at least 32 ASCII characters")
    owns_engine = engine is None
    if engine is None:
        url = database_url or os.getenv("DATABASE_URL", "")
        if not url:
            raise ValueError("DATABASE_URL is required; run migrations before startup")
        engine = make_engine(url)
    require_schema(engine)
    if generator_factory is None:
        generator_factory, generator_config = environment_generator()
    repo = Repository(sessions(engine))
    service = EvaluationService(repo, generator_factory, generator_config or {"mode": "injected"},
                                max_concurrency=max_concurrency)

    @asynccontextmanager
    async def lifespan(app):
        yield
        if owns_engine:
            engine.dispose()

    app = FastAPI(title="岗位证据评估后端", version="1.0", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.state.repository, app.state.evaluator = repo, service

    def authorize(authorization: Annotated[str | None, Header()] = None):
        if authorization is None or not secrets.compare_digest(
                authorization.encode(), f"Bearer {token}".encode()):
            raise HTTPException(401, "unauthorized", headers={"WWW-Authenticate": "Bearer"})

    # Do not reflect request bodies or SQL parameters in errors (profile content may be private).
    from fastapi.exceptions import RequestValidationError

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        return JSONResponse(status_code=422, content={"detail": "invalid_request",
            "errors": [{"loc": list(e["loc"]), "type": e["type"]} for e in exc.errors()]})

    @app.exception_handler(Conflict)
    async def conflict(request, exc):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(NotFound)
    async def missing(request, exc):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(CapacityExceeded)
    async def capacity(request, exc):
        return JSONResponse(status_code=503, content={"detail": str(exc)}, headers={"Retry-After": "2"})

    @app.exception_handler(SQLAlchemyError)
    async def database_error(request, exc):
        return JSONResponse(status_code=503, content={"detail": "database_unavailable"})

    @app.get("/health")
    def health():
        from sqlalchemy import text
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return {"status": "ok", "schema": "backend_0001", "generator": service.config["mode"]}

    auth = [Depends(authorize)]

    @app.post("/v1/jobs/import", dependencies=auth)
    def import_jobs(body: ImportRequest):
        return repo.import_jobs(body.jobs)

    @app.get("/v1/jobs", dependencies=auth)
    def list_jobs(city: str | None = Query(None, min_length=1, max_length=100),
                  title: str | None = Query(None, min_length=1, max_length=300),
                  company: str | None = Query(None, min_length=1, max_length=300),
                  limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0, le=10000)):
        return repo.list_jobs(city=city, title=title, company=company, limit=limit, offset=offset)

    @app.get("/v1/snapshots/{snapshot_id}", dependencies=auth)
    def get_snapshot(snapshot_id: str):
        return repo.get_snapshot(snapshot_id)

    @app.post("/v1/evaluations", dependencies=auth)
    def evaluate(body: EvaluationRequest,
                 idempotency_key: Annotated[str, Header(min_length=8, max_length=128,
                                                       pattern=r"^[A-Za-z0-9_.:-]+$")]):
        record, replayed = service.evaluate(idempotency_key, body)
        return JSONResponse(status_code=202 if record["status"] == "running" else (200 if replayed else 201),
                            content={**record, "replayed": replayed})

    @app.get("/v1/evaluations", dependencies=auth)
    def list_evaluations(limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0, le=10000)):
        return repo.list_evaluations(limit=limit, offset=offset)

    @app.get("/v1/evaluations/{evaluation_id}", dependencies=auth)
    def get_evaluation(evaluation_id: str):
        return repo.get_evaluation(evaluation_id)

    @app.post("/v1/evaluations/{evaluation_id}/reviews", dependencies=auth)
    def review(evaluation_id: str, body: ReviewRequest):
        return repo.add_review(evaluation_id, body)

    return app
