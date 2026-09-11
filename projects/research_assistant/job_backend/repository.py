"""Short transactions; snapshots and original model results are immutable."""
from __future__ import annotations

import hashlib
import json
from uuid import uuid4

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError

from .models import Evaluation, EvaluationReview, Job, JobSnapshot, utcnow


class Conflict(Exception):
    pass


class NotFound(Exception):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def snapshot_dict(row):
    return {"id": row.id, "job_id": row.job_id, "revision": row.revision,
            "created_at": row.created_at.isoformat(), **row.content}


def evaluation_dict(row):
    return {key: getattr(row, key) for key in (
        "id", "snapshot_id", "request", "status", "result", "failure_type", "review_version"
    )} | {"created_at": row.created_at.isoformat(), "updated_at": row.updated_at.isoformat()}


class Repository:
    def __init__(self, session_factory):
        self.sessions = session_factory

    def import_jobs(self, jobs):
        # A duplicate source in one batch is ambiguous; reject before opening a transaction.
        if len({job.source_key for job in jobs}) != len(jobs):
            raise Conflict("duplicate_source_in_batch")
        created, unchanged, items = 0, 0, []
        with self.sessions.begin() as session:
            insert = pg_insert if session.bind.dialect.name == "postgresql" else sqlite_insert
            # Consistent row-lock order prevents crossing batches from deadlocking.
            for item in sorted(jobs, key=lambda job: job.source_key):
                content = item.model_dump(mode="json")
                job_id, snapshot_id = digest(item.source_key), digest(content)
                session.execute(insert(Job).values(id=job_id, source_key=item.source_key, revision=0)
                                .on_conflict_do_nothing(index_elements=["source_key"]))
                job = session.scalar(select(Job).where(Job.id == job_id).with_for_update())
                snapshot = session.get(JobSnapshot, snapshot_id)
                if snapshot is None:
                    job.revision += 1
                    snapshot = JobSnapshot(id=snapshot_id, job_id=job.id, revision=job.revision,
                                           city=item.city, title=item.title, company=item.company,
                                           content=content)
                    session.add(snapshot)
                    session.flush()
                    created += 1
                else:
                    unchanged += 1
                items.append({"job_id": job.id, "snapshot_id": snapshot.id,
                              "revision": snapshot.revision, "current_revision": job.revision})
        return {"created_snapshots": created, "unchanged": unchanged, "items": items}

    def list_jobs(self, *, city=None, title=None, company=None, limit=20, offset=0):
        query = select(JobSnapshot).join(Job, JobSnapshot.job_id == Job.id).where(
            JobSnapshot.revision == Job.revision)
        if city:
            query = query.where(JobSnapshot.city == city)
        if company:
            query = query.where(JobSnapshot.company == company)
        if title:
            query = query.where(JobSnapshot.title.contains(title, autoescape=True))
        with self.sessions() as session:
            total = session.scalar(select(func.count()).select_from(query.subquery()))
            rows = session.scalars(query.order_by(JobSnapshot.job_id).offset(offset).limit(limit))
            return {"total": total, "limit": limit, "offset": offset,
                    "items": [snapshot_dict(row) for row in rows]}

    def get_snapshot(self, snapshot_id):
        with self.sessions() as session:
            row = session.get(JobSnapshot, snapshot_id)
            if row is None:
                raise NotFound("snapshot_not_found")
            return snapshot_dict(row)

    def get_evaluation(self, evaluation_id):
        with self.sessions() as session:
            row = session.get(Evaluation, evaluation_id)
            if row is None:
                raise NotFound("evaluation_not_found")
            result = evaluation_dict(row)
            reviews = session.scalars(select(EvaluationReview).where(
                EvaluationReview.evaluation_id == evaluation_id).order_by(EvaluationReview.version))
            result["reviews"] = [{"id": r.id, "version": r.version, "decision": r.decision,
                                  "note": r.note, "created_at": r.created_at.isoformat()} for r in reviews]
            return result

    def list_evaluations(self, *, limit=20, offset=0):
        with self.sessions() as session:
            total = session.scalar(select(func.count()).select_from(Evaluation))
            rows = session.scalars(select(Evaluation).order_by(
                Evaluation.created_at.desc(), Evaluation.id).offset(offset).limit(limit))
            return {"total": total, "limit": limit, "offset": offset,
                    "items": [evaluation_dict(row) for row in rows]}

    def lookup_request(self, key, request_hash):
        with self.sessions() as session:
            row = session.scalar(select(Evaluation).where(Evaluation.idempotency_key == key))
            if row is None:
                return None
            if row.request_hash != request_hash:
                raise Conflict("idempotency_key_reused_with_different_request")
            return evaluation_dict(row)

    def start_evaluation(self, key, request_hash, request):
        evaluation_id = str(uuid4())
        try:
            with self.sessions.begin() as session:
                session.add(Evaluation(id=evaluation_id, idempotency_key=key,
                                       request_hash=request_hash, snapshot_id=request["snapshot_id"],
                                       request=request, status="running"))
            return self.get_evaluation(evaluation_id), True
        except IntegrityError:
            # Unique constraint is the cross-thread/process idempotency arbiter.
            existing = self.lookup_request(key, request_hash)
            if existing is None:
                raise
            return existing, False

    def finish_evaluation(self, evaluation_id, *, result=None, failure_type=None):
        with self.sessions.begin() as session:
            changed = session.execute(update(Evaluation).where(
                Evaluation.id == evaluation_id, Evaluation.status == "running"
            ).values(status="failed" if failure_type else "succeeded", result=result,
                     failure_type=failure_type, updated_at=utcnow()))
            if changed.rowcount != 1:
                raise Conflict("evaluation_already_finished")
        return self.get_evaluation(evaluation_id)

    def add_review(self, evaluation_id, request):
        with self.sessions.begin() as session:
            row = session.get(Evaluation, evaluation_id)
            if row is None:
                raise NotFound("evaluation_not_found")
            if row.status != "succeeded":
                raise Conflict("evaluation_not_succeeded")
            changed = session.execute(update(Evaluation).where(
                Evaluation.id == evaluation_id, Evaluation.review_version == request.expected_version
            ).values(review_version=request.expected_version + 1, updated_at=utcnow()))
            if changed.rowcount != 1:
                raise Conflict("review_version_conflict")
            session.add(EvaluationReview(id=str(uuid4()), evaluation_id=evaluation_id,
                                         version=request.expected_version + 1,
                                         decision=request.decision, note=request.note))
        return self.get_evaluation(evaluation_id)
