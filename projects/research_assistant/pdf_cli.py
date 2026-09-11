"""Local PDF module CLI. Paid calls require --mode model; default is source excerpts."""
import argparse
import json
from pathlib import Path
import sys

from pdf_module.config import configured_model
from pdf_module.parser import MAX_BYTES, PdfError
from pdf_module.service import PdfService
from pdf_module.store import PdfStore, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="PDF 结构化入库与证据问答")
    parser.add_argument("--store", type=Path, default=Path(__file__).resolve().parents[2] / "output/pdf-runtime")
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest")
    ingest.add_argument("path", type=Path)
    sub.add_parser("list")
    show = sub.add_parser("show")
    show.add_argument("document_id")
    show.add_argument("--page", type=int)
    ask = sub.add_parser("ask")
    ask.add_argument("--document", action="append", required=True)
    ask.add_argument("--question", required=True)
    ask.add_argument("--page", type=int, action="append")
    ask.add_argument("--top-k", type=int, default=3)
    ask.add_argument("--strategy", choices=["lexical", "hybrid"], default="lexical")
    ask.add_argument("--mode", choices=["extractive", "model"], default="extractive")
    args = parser.parse_args(argv)
    store = PdfStore(args.store)
    try:
        if args.command == "ingest":
            if args.path.stat().st_size > MAX_BYTES:
                raise PdfError("file_too_large")
            doc, cached = store.ingest(args.path.read_bytes(), args.path.name)
            result = {"document": summary(doc), "cached": cached}
        elif args.command == "list":
            result = {"documents": store.list()}
        elif args.command == "show":
            doc = store.get(args.document_id)
            if args.page is not None and not 1 <= args.page <= doc["page_count"]:
                raise PdfError("page_not_found")
            result = doc["pages"][args.page-1] if args.page else summary(doc)
        else:
            service = PdfService(store, model_factory=configured_model if args.mode == "model" else None)
            result = service.ask(args.question, args.document, pages=args.page, top_k=args.top_k,
                                 strategy=args.strategy, mode=args.mode)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2 if result.get("refused") else 0
    except (PdfError, OSError, ValueError) as exc:
        print(json.dumps({"error": exc.code if isinstance(exc, PdfError) else "invalid_input_or_storage"}))
        return 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
