"""Explicit schema migration; never create database schema during API startup."""
import argparse
import os

from job_backend.database import make_engine, migrate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["migrate"])
    parser.parse_args()
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        parser.error("DATABASE_URL is required")
    engine = make_engine(url)
    try:
        migrate(engine)
        print("Backend schema is at backend_0001")
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
