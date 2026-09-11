"""Copy a named set of project sources and replay assets; never copy the workspace tree."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PROJECT = "projects/research_assistant/"
CORE = """api.py assistant.py agent_loop.py cli.py document.py grounded_service.py
service_factory.py market_analysis.py analyze_learning_alignment.py retriever.py vector_retriever.py retrieval_pipeline.py
reranker.py prompt.py security.py rag_observability.py observability.py rag_workflow.py
langgraph_workflow.py rag_evaluation.py rerank_evaluation.py job_vector_cli.py
job_dataset_pipeline.py backend_api.py backend_cli.py pdf_api.py pdf_cli.py demo_replay.py
test_demo_replay.py backend-alembic.ini requirements-api.txt requirements-backend.txt
requirements-embedding.txt requirements-workflow.txt requirements-pdf.txt requirements-pdf-test.txt""".split()
SUBMODULES = {
    "job_backend": "__init__.py contracts.py corpus.py database.py evaluation.py models.py repository.py".split(),
    "pdf_module": "__init__.py chunks.py config.py parser.py retrieval.py service.py store.py".split(),
    "backend_migrations": ["env.py", "versions/backend_0001.py"],
    "demo": ["index.html", "replay.json", "selected_sources.json", "page.html", "README.md"],
}


def public_mapping():
    result = {PROJECT + name: PROJECT + name for name in CORE}
    for folder, names in SUBMODULES.items():
        result.update({PROJECT + folder + "/" + name: PROJECT + folder + "/" + name for name in names})
    result.update({"README.md": PROJECT + "demo/PUBLIC_README.md",
                   "LICENSE": PROJECT + "demo/LICENSE_CODE.txt",
                   "OPEN_DEMO.html": PROJECT + "demo/index.html",
                   "tools/export_replay_project.py": "tools/export_replay_project.py",
                   "experiments/01-minimal-agent/agent_loop.py": "experiments/01-minimal-agent/agent_loop.py",
                   "experiments/02-openai-compatible-adapter/openai_compatible.py":
                       "experiments/02-openai-compatible-adapter/openai_compatible.py"})
    # Retain source docs so the exported checkout can produce another export.
    result.update({PROJECT + "demo/" + name: PROJECT + "demo/" + name
                   for name in ["PUBLIC_README.md", "LICENSE_CODE.txt"]})
    return result


def scan_file(path):
    text = path.read_text(encoding="utf-8")
    # Scan concrete secrets/contacts, not the words API_KEY in legitimate source.
    expressions = {
        "credential": r"sk-[A-Za-z0-9_-]{24,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----",
        "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        "mobile": r"(?<![A-Za-z0-9])1[3-9]\d{9}(?![A-Za-z0-9])",
    }
    for label, pattern in expressions.items():
        if re.search(pattern, text):
            raise ValueError(f"review required: {label} in {path.name}")


def export_project(destination, source_root=ROOT):
    source_root = source_root.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError("use a new destination; existing files are preserved")
    mapping = public_mapping()
    # Read and check everything before creating a publishable directory.
    contents = {}
    for relative, origin in mapping.items():
        path = source_root / origin
        if path.is_symlink() or not path.resolve().is_relative_to(source_root):
            raise ValueError("source must be an ordinary file inside the project")
        scan_file(path)
        contents[relative] = path.read_bytes()
    index = contents["OPEN_DEMO.html"].decode("utf-8")
    embedded = index.split('<script type="application/json" id="demo-data">', 1)[1].split("</script>", 1)[0]
    replay = json.loads(contents[PROJECT + "demo/replay.json"])
    if json.loads(embedded) != replay:
        raise ValueError("HTML and replay.json must come from the same build")
    selected = json.loads(contents[PROJECT + "demo/selected_sources.json"])
    digest = sha256(json.dumps(selected, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    if replay["source_sha256"] != digest:
        raise ValueError("selected sources changed; regenerate the replay first")
    for name, expected in replay["code_sha256"].items():
        if sha256(contents[PROJECT + name]).hexdigest() != expected:
            raise ValueError("recorded source changed; regenerate the replay first")
    contents[".gitignore"] = b".venv/\n__pycache__/\n*.py[cod]\n.env\n.env.*\ndata/local/\ndata/raw/\ndata/index/\nlogs/\noutput/\n"
    manifest = {"schema_version": 1, "scope": "selected code and local replay; no original dataset or git history",
                "files": {key: {"sha256": sha256(data).hexdigest(), "bytes": len(data)}
                          for key, data in sorted(contents.items())}}
    contents["PUBLIC_FILES.json"] = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    destination.mkdir(parents=True)
    for name, content in contents.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return manifest


def verify_export(directory):
    manifest = json.loads((directory / "PUBLIC_FILES.json").read_text(encoding="utf-8"))
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    expected = set(manifest["files"]) | {"PUBLIC_FILES.json"}
    if actual != expected:
        raise ValueError("export contains missing or extra files")
    for name, entry in manifest["files"].items():
        if sha256((directory / name).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("export file changed: " + name)
    return {"files": len(actual), "bytes": sum(p.stat().st_size for p in directory.rglob("*") if p.is_file()),
            "manifest_verified": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description="导出仅含项目源码和精选回放材料的目录与 ZIP")
    parser.add_argument("destination", type=Path)
    args = parser.parse_args(argv)
    destination = args.destination.resolve()
    archive = destination.with_name(destination.name + ".zip")
    if archive.exists():
        raise FileExistsError("archive already exists")
    export_project(destination)
    report = verify_export(destination)
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as output:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                output.write(path, path.relative_to(destination).as_posix())
    report.update(archive=str(archive), archive_sha256=sha256(archive.read_bytes()).hexdigest())
    print(json.dumps(report, ensure_ascii=True))


if __name__ == "__main__":
    main()
