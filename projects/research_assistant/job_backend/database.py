"""Explicit migrations and per-operation sessions; never share a Session."""
from __future__ import annotations

from pathlib import Path
from sqlalchemy import create_engine, event, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from alembic import command
from alembic.config import Config

ROOT = Path(__file__).resolve().parents[1]


def make_engine(url: str):
    options = {"pool_pre_ping": True, "hide_parameters": True}
    if url.startswith("sqlite"):
        options["connect_args"] = {"check_same_thread": False, "timeout": 20}
        if url.endswith(":memory:") or url == "sqlite://":
            options["poolclass"] = StaticPool
    engine = create_engine(url, **options)
    if engine.dialect.name not in {"sqlite", "postgresql"}:
        engine.dispose()
        raise ValueError("backend supports PostgreSQL and SQLite only")
    if engine.dialect.name == "sqlite":
        @event.listens_for(engine, "connect")
        def configure_sqlite(connection, record):
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA busy_timeout=20000")
    return engine


def migrate(engine, revision: str = "head") -> None:
    config = Config(str(ROOT / "backend-alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "backend_migrations"))
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, revision)


def require_schema(engine) -> None:
    from sqlalchemy import text
    if "alembic_version" not in inspect(engine).get_table_names():
        raise RuntimeError("run backend_cli.py migrate before serving requests")
    with engine.connect() as connection:
        if connection.execute(text("SELECT version_num FROM alembic_version")).scalar() != "backend_0001":
            raise RuntimeError("backend database schema is not current")


def sessions(engine):
    return sessionmaker(engine, expire_on_commit=False)
