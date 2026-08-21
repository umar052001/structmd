#!/usr/bin/env python3
"""Example: Convert a document to Markdown using structmd.

Run against a local Ollama server with a vision model installed:

    ollama pull qwen2-vl:2b
    python examples/convert.py document.pdf
"""

import sys

from structmd import StructMDPipeline


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "document.pdf"

    pipeline = StructMDPipeline()

    # Single document: full two-stage run.
    result = pipeline.process(
        path,
        output_json="extraction.json",  # keep Stage 1 output for inspection
        output_md="output.md",
    )
    print(f"Title: {result.title}")
    print(f"Pages: {result.metadata.get('page_count')}")
    print("Wrote output.md and extraction.json")

    # Batch processing: pages from all documents share one async worker pool.
    batch_paths = ["doc1.pdf", "doc2.docx"]
    results = pipeline.process_batch(batch_paths)
    for r in results:
        print(f"Converted: {r.title}")

    pipeline.close()


if __name__ == "__main__":
    main()
