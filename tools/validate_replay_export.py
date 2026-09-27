"""Validate an exported ZIP in a separate checkout using current Python dependencies."""
from __future__ import annotations

import argparse
import ast
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from uuid import uuid4
import zipfile

from export_replay_project import verify_export

ROOT = Path(__file__).resolve().parents[1]


def validate(archive: Path):
    scratch = ROOT / "output/replay-validation/clean-room" / uuid4().hex
    checkout = scratch / "checkout"
    checkout.mkdir(parents=True)
    with zipfile.ZipFile(archive) as bundle:
        for name in bundle.namelist():
            if not (checkout / name).resolve().is_relative_to(checkout.resolve()):
                raise ValueError("archive path escapes checkout")
        bundle.extractall(checkout)
    before = verify_export(checkout)
    for path in checkout.rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
    links = re.findall(r"\]\(([^)]+)\)", (checkout / "README.md").read_text(encoding="utf-8"))
    assert all((checkout / link).exists() for link in links if not link.startswith("https:"))
    environment = os.environ.copy()
    isolated_temp = scratch / "temp"
    isolated_temp.mkdir()
    environment.update(
        PYTHONPATH="",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUTF8="1",
        TEMP=str(isolated_temp),
        TMP=str(isolated_temp),
        TMPDIR=str(isolated_temp),
    )
    for name in ["MODEL_API_KEY", "MODEL_ENDPOINT", "MODEL_NAME"]:
        environment.pop(name, None)
    project = checkout / "projects/research_assistant"
    commands = [
        ("public unit tests", ["-m", "unittest", "test_demo_replay", "test_personal_live_cli",
                               "test_personal_live_web", "test_live_web_scenarios"]),
        ("regenerate replay", ["demo_replay.py", "--output", str(scratch / "rebuilt"),
                               "--runtime", str(scratch / "runtime")]),
        ("entrypoint imports", ["-c", """import agent_loop,assistant,grounded_service,service_factory,backend_api,pdf_api,langgraph_workflow,personal_live_cli,personal_live_web,run_live_web_scenarios
from pathlib import Path
assert all(Path(m.__file__).resolve().is_relative_to(Path.cwd().parents[1]) for m in [agent_loop,assistant,grounded_service,service_factory,backend_api,pdf_api,langgraph_workflow,personal_live_cli,personal_live_web,run_live_web_scenarios])
"""]),
        ("paid scenario gate", ["-c", """import run_live_web_scenarios as runner
try:
    runner.main([])
except SystemExit as error:
    assert '--execute-real-api' in str(error)
else:
    raise AssertionError('paid scenario suite ran without explicit acknowledgement')
"""]),
    ]
    # Exercise the README's CSV-to-CLI route with selected excerpts, not the private corpus.
    prepare = """import csv,json,sys
from pathlib import Path
from document import load_job_csv_documents,chunk_document,write_chunks_jsonl
base=Path(sys.argv[1]); source=json.loads(Path('demo/selected_sources.json').read_text(encoding='utf-8'))
path=base/'jobs.csv'
with path.open('w',encoding='utf-8',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=['id','job_title','company','city','duties_summary','source_url']);writer.writeheader()
 for j in source['jobs']: writer.writerow(dict(id=j['id'],job_title=j['title'],company=j['company'],city=j['city'],duties_summary=j['text'],source_url=j['source_url']))
write_chunks_jsonl([c for d in load_job_csv_documents(path) for c in chunk_document(d)],base/'chunks.jsonl')
"""
    commands.extend([
        ("own CSV conversion", ["-c", prepare, str(scratch)]),
        ("documented lexical CLI", ["cli.py", "RAG Agent", "--strategy", "lexical", "--generator",
                                    "extractive", "--chunks-path", str(scratch / "chunks.jsonl"),
                                    "--trace-path", str(scratch / "trace.jsonl"), "--json"]),
    ])
    results = []
    for index, (label, args) in enumerate(commands):
        result = subprocess.run([sys.executable, "-B", *args], cwd=project, env=environment,
                                capture_output=True, text=True, encoding="utf-8")
        (scratch / f"check-{index}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        results.append({"check": label, "exit_code": result.returncode})
        if result.returncode:
            raise RuntimeError(f"{label} failed; inspect {scratch / f'check-{index}.log'}")
        if label == "documented lexical CLI":
            answer = json.loads(result.stdout)
            assert not answer["refused"] and answer["citations"]
    assert before == verify_export(checkout)
    probe = checkout / "private-data.csv"
    probe.write_text("disposable validation probe", encoding="utf-8")
    try:
        verify_export(checkout)
    except ValueError:
        pass
    else:
        raise AssertionError("unexpected file not detected")
    finally:
        probe.unlink()
    return {"schema_version": 1, "passed": True, "archive": archive.relative_to(ROOT).as_posix(),
            "archive_sha256": sha256(archive.read_bytes()).hexdigest(), "package": verify_export(checkout),
            "readme_links_exist": True, "python_syntax_passed": True, "clean_room_runs": results,
            "extra_file_rejected": True, "new_model_calls": 0, "real_api_calls": 0,
            "browser_interaction_tested": False,
            "environment": "Existing Python dependencies reused; project imports resolved within extracted source; temporary files isolated under the clean-room workspace; no fresh dependency installation."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    report = validate(args.archive.resolve())
    output = ROOT / "output/replay-validation/package-validation.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))


if __name__ == "__main__":
    main()
