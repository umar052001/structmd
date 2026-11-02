"""Reproducible extraction-quality benchmark for structmd.

Usage::

    python -m benchmarks.run   # against benchmarks/corpus/*.pdf
    python -m benchmarks.run --corpus docs/papers -o report.json
    python -m benchmarks.run --no-live   # score only docs already in cache

The run reads the same page cache as the CLI: a corpus scored once is
re-scored offline in seconds on subsequent runs. When pages are missing
from the cache the pipeline calls the configured Ollama model (unless
``--no-live``), so a cold benchmark needs a reachable VLM endpoint.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from benchmarks import gt, metrics
from structmd.cache.manager import CacheManager
from structmd.config import load_config
from structmd.converters.pdf import PDFConverter
from structmd.core import ElementType, ExtractedDocument
from structmd.pipeline import StructMDPipeline


@dataclass
class DocumentReport:
    name: str
    pages: int = 0
    content: Dict[str, float] = field(default_factory=dict)
    reading_order: float = 0.0
    heading_text_absence: float = 0.0
    tables: Dict[str, Any] = field(default_factory=dict)
    figures: Dict[str, Any] = field(default_factory=dict)


def score_document(name: str, extracted: ExtractedDocument, pdf_path: str) -> DocumentReport:
    report = DocumentReport(name=name, pages=len(extracted.pages))

    truth = gt.extract_ground_truth(pdf_path, [p.page_number for p in extracted.pages])
    if not extracted.pages or not truth.strip():
        raise ValueError(f"{name}: scoring requires extracted pages and a PDF text layer")
    extracted_text = "\n".join(gt.extracted_text_pages(extracted))
    report.content = dict(
        zip(("precision", "recall", "f1"), metrics.content_f1(extracted_text, truth))
    )
    report.reading_order = metrics.reading_order_similarity(extracted_text, truth)

    headings = [
        el.text
        for page in extracted.pages
        for el in page.elements
        if el.type is ElementType.HEADING and el.text
    ]
    report.heading_text_absence = metrics.heading_text_absence_rate(headings, truth)

    tables = [
        el for page in extracted.pages for el in page.elements if el.type is ElementType.TABLE
    ]
    consistency = [
        m
        for m in (
            metrics.table_width_consistency(
                None,
                t.table_data,
            )
            for t in tables
            if t.table_data
        )
        if m is not None
    ]
    report.tables = {
        "count": len(tables),
        "rows": sum(len(t.table_data or []) for t in tables),
        "width_consistent": (
            metrics.aggregate([1.0 if ok else 0.0 for ok in consistency]) if consistency else None
        ),
    }

    images = [
        el for page in extracted.pages for el in page.elements if el.type is ElementType.IMAGE
    ]
    report.figures = {
        "detected": len(images),
        "asset_evaluation": "not_evaluated",
    }
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="benchmarks/corpus", help="directory of PDFs to score")
    parser.add_argument("-o", "--output", default="benchmarks/report.json")
    parser.add_argument(
        "--no-live", action="store_true", help="score only documents fully available from cache"
    )
    parser.add_argument("--model", help="Ollama model (default: config)")
    parser.add_argument("--url", help="Ollama base URL (default: config)")
    parser.add_argument("--dpi", type=int, help="render DPI (default: config)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)
    corpus = Path(args.corpus)
    pdfs = sorted(corpus.glob("*.pdf"))
    if not pdfs:
        print(
            f"no PDFs found under {corpus} (run benchmarks/fetch_papers.sh first)", file=sys.stderr
        )
        return 2

    base = load_config()
    if args.model:
        base.ollama_model = args.model
    if args.url:
        base.ollama_url = args.url
    if args.dpi:
        base.dpi = args.dpi

    pipeline = None if args.no_live else StructMDPipeline(base)
    cache = CacheManager(base.cache_dir)
    reports: List[DocumentReport] = []
    started = time.monotonic()
    try:
        for pdf in pdfs:
            if pipeline is None:
                count = PDFConverter().get_page_count(str(pdf))
                cached_pages = [cache.load_page(str(pdf), n) for n in range(1, count + 1)]
                if any(page is None for page in cached_pages):
                    print(f"skipping {pdf.name}: incomplete page cache", file=sys.stderr)
                    continue
                doc = ExtractedDocument(
                    source_path=str(pdf),
                    page_count=count,
                    pages=[page for page in cached_pages if page is not None],
                )
            else:
                doc = pipeline.extract_only(str(pdf))
            reports.append(score_document(pdf.name, doc, str(pdf)))
    except Exception as exc:  # noqa: BLE001 - report clearly instead of dying mid-corpus
        print(f"error scoring corpus: {exc}", file=sys.stderr)
        if not args.no_live:
            print(
                "hint: start Ollama or run once with a warm cache, or use --no-live",
                file=sys.stderr,
            )
        raise SystemExit(1) from exc
    finally:
        if pipeline is not None:
            pipeline.close()

    if not reports:
        print("no documents scored", file=sys.stderr)
        return 2

    summary = {
        "documents": len(reports),
        "documents_skipped": len(pdfs) - len(reports),
        "content_f1": metrics.merge_f1(
            [(r.content["precision"], r.content["recall"], r.content["f1"]) for r in reports]
        ),
        "reading_order": metrics.aggregate([r.reading_order for r in reports]),
        "heading_text_absence": metrics.aggregate([r.heading_text_absence for r in reports]),
        "table_width_consistent": metrics.aggregate(
            [
                r.tables["width_consistent"]
                for r in reports
                if r.tables["width_consistent"] is not None
            ]
        ),
        "figures_detected": sum(r.figures["detected"] for r in reports),
        "elapsed_s": round(time.monotonic() - started, 1),
    }

    payload = {
        "meta": {
            "requested_model": base.ollama_model,
            "cached_model_provenance_verified": False,
            "dpi": base.dpi,
            "corpus": str(corpus),
            "live": not args.no_live,
            "metric_version": 2,
            "limitations": [
                "PDF text layer is a proxy reference, not annotated ground truth",
                "heading_text_absence measures missing text, not structural correctness",
                "figure assets are not generated in extraction-only mode",
                "reading_order uses word sequences with autojunk disabled",
            ],
        },
        "summary": summary,
        "documents": [asdict(r) for r in reports],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    row = lambda name, pages, f1, order, hall, tables: (  # noqa: E731
        f"{name:<22} {pages:>5} {f1:>6.3f} {order:>10.3f} {hall:>9.3f} {tables:>4}"
    )
    print(f"{'document':<22} {'pages':>5} {'f1':>6} {'order':>10} {'hdr-miss':>9} {'tbl':>4}")
    print("-" * 60)
    for r in reports:
        print(
            row(
                r.name,
                r.pages,
                r.content.get("f1", 0.0),
                r.reading_order,
                r.heading_text_absence,
                r.tables["count"],
            )
        )
    print("-" * 60)
    s = summary
    print(
        row(
            "AVERAGE",
            "",
            s["content_f1"]["f1"],
            s["reading_order"],
            s["heading_text_absence"],
            "—",
        )
    )
    print(
        f"\nfigures detected: {s['figures_detected']} (asset export not evaluated)   "
        f"elapsed: {s['elapsed_s']}s\nreport: {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
