# structmd

[![CI](https://github.com/umar052001/structmd/actions/workflows/ci.yml/badge.svg)](https://github.com/umar052001/structmd/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/structmd?color=6d5cff)](https://pypi.org/project/structmd/)
[![Python](https://img.shields.io/pypi/pyversions/structmd)](https://pypi.org/project/structmd/)
[![Docs](https://img.shields.io/badge/docs-github.io-00b3e6)](https://umar052001.github.io/structmd/)
[![License: MIT](https://img.shields.io/badge/license-MIT-ff4f9a)](LICENSE)

**Convert PDFs, Office documents, and images into structured Markdown using Vision-Language Models served by [Ollama](https://ollama.com) — local or cloud.**

structmd is built around a strict two-stage architecture:

```
┌─────────────────────────────┐      ┌──────────────────────────────────┐
│ Stage 1 — Extraction (VLM)  │      │ Stage 2 — Building (deterministic)│
│                             │      │                                  │
│  PDF/DOCX/PNG ─▶ page images │ ──▶ │  extraction JSON ─▶ Markdown     │
│  Ollama /api/chat per page   │      │  pure algorithms, zero ML        │
│  output: structured JSON     │      │  byte-for-byte reproducible      │
└─────────────────────────────┘      └──────────────────────────────────┘
```

**JSON is the single source of truth.** The VLM never writes Markdown directly. It produces an inspectable, editable, cacheable JSON description of the document layout; a deterministic builder turns that JSON into Markdown. Same JSON in → identical Markdown out, every time.

## Why this architecture?

- **Small-model friendly.** 2B–3B VLMs are bad at writing clean Markdown but decent at describing layout as JSON. structmd plays to that strength.
- **Debuggable.** Bad conversion? Open the JSON and see exactly what the model saw. Fix it by hand and rebuild without re-running inference.
- **Cheap to iterate.** Extraction is cached by file hash + mtime — including *per page*. Re-tune Markdown output (heading levels, table captions, column handling) instantly from cached JSON.
- **Deterministic output.** The builder is pure code: no sampling, no randomness, no hidden state.

## Installation

Requires Python ≥ 3.9 and a running [Ollama](https://ollama.com) server.

```bash
# with uv (recommended)
uv add "structmd[pdf]"

# or pip
pip install "structmd[pdf]"
```

Extras:

| Extra | Installs | Needed for |
|---|---|---|
| `[pdf]` | PyMuPDF | PDF rendering (also powers figure extraction) |
| `[office]` | PyMuPDF | Office documents (converted through LibreOffice → PDF) |
| `[all]` | everything above | one-stop install |

Office documents additionally require **LibreOffice** (`soffice`) on your PATH:

```bash
sudo apt install libreoffice        # Debian/Ubuntu
brew install --cask libreoffice     # macOS
```

## Ollama setup

structmd works with **both local and cloud models** — the same API, the same code path.

### Local models

```bash
ollama pull qwen2-vl:2b          # ~1.6 GB, good default
# alternatives:
ollama pull smolvlm              # very light
ollama pull llama3.2-vision      # larger, stronger
```

### Cloud models (no GPU needed)

Ollama can transparently offload larger vision models to [ollama.com](https://ollama.com) while your tooling keeps talking to `localhost:11434`:

```bash
ollama signin                    # one-time account link
ollama pull gemma4:cloud         # registers the cloud model (no big download)

structmd scan.pdf --model gemma4:cloud -o output.md
```

Notes:

- **Throughput**: cloud models are typically much faster than CPU-bound local inference. In our benchmarks, five arXiv papers (77 pages) converted in ~13 minutes with `gemma4:cloud` at 3 workers (~10 s/page effective).
- **Timeouts**: large frontier models can take longer per page; raise the budget with `STRUCTMD_OLLAMA_TIMEOUT=300` or the YAML key `ollama.timeout`.
- **Privacy**: pages are sent to Ollama's cloud service. For sensitive documents, stick to local models — structmd treats both identically.

Verify your endpoint is up:

```bash
curl http://localhost:11434/api/tags
```

structmd auto-detects model tags (`qwen2-vl` resolves to `qwen2-vl:latest`; `gemma4:cloud` is used verbatim) and raises a clear error with the exact `ollama pull …` command if the model is missing.

## Quick start

### CLI

```bash
# simplest form
structmd input.pdf -o output.md

# keep the intermediate JSON too
structmd input.pdf --json extraction.json --md output.md

# tune the run
structmd input.pdf --model qwen2-vl:2b --workers 8 --dpi 200

# rebuild Markdown from existing JSON — no VLM needed
structmd --from-json extraction.json -o output.md

# batch: many documents through one async worker pool
structmd batch doc1.pdf doc2.docx doc3.png -o output_dir/

# whole directories work too (walked recursively)
structmd batch ./papers/ -o output_dir/

# page selection (1-indexed, ranges allowed)
structmd input.pdf --pages 1,3,5-10 -o output.md

# extract figures as PNGs and link them in the Markdown
structmd input.pdf --save-assets -o output.md   # -> ./figures/*.png

# custom config file
structmd --config ~/.structmd.yaml input.pdf -o out.md
```

### Python

```python
from structmd import StructMDPipeline

with StructMDPipeline() as pipeline:
    result = pipeline.process(
        "document.pdf",
        output_json="extraction.json",   # optional: keep Stage 1 output
        output_md="output.md",           # optional: write final Markdown
    )

print(result.title)                      # extracted from the first heading
print(result.content[:200])              # the Markdown itself
print(result.metadata["page_count"])     # 12
```

## Guide: convert a folder full of PDFs

The most common request. A complete, runnable script:

```python
"""Convert every PDF in a folder (recursively) to Markdown."""
from pathlib import Path

from structmd import StructMDPipeline
from structmd.config import StructMDConfig

# 1. Configure once — these settings apply to every document.
config = StructMDConfig(
    ollama_model="gemma4:cloud",   # or a local tag, e.g. "qwen2-vl:2b"
    ollama_max_workers=3,          # pages processed concurrently
    save_assets=True,              # crop figures out as PNGs
)

# 2. Collect inputs. rglob walks subfolders; use glob for top-level only.
pdfs = sorted(Path("papers").rglob("*.pdf"))
print(f"Found {len(pdfs)} PDFs")

# 3. Run. All pages of all documents share one async worker pool,
#    and finished pages are cached individually.
with StructMDPipeline(config) as pipeline:
    documents = pipeline.process_batch(
        [str(p) for p in pdfs],
        output_dir="markdown",     # anchors figure assets at markdown/figures/
    )

# 4. Write the Markdown files.
for doc in documents:
    stem = Path(doc.metadata["source_path"]).stem
    target = Path("markdown") / f"{stem}.md"
    doc.save(str(target))
    print(f"{target}  ({doc.metadata['page_count']} pages)")
```

What each import gives you:

| Import | Why |
|---|---|
| `pathlib.Path` | stdlib — walking the folder and building output paths |
| `structmd.StructMDPipeline` | the orchestrator: converters + extractor + builder + cache |
| `structmd.config.StructMDConfig` | typed configuration dataclass; every CLI/env option is a field |

Behavior you get for free:

- **Parallelism** — all pages from all PDFs flow through one worker pool (`ollama_max_workers`), not one PDF at a time.
- **Resumability** — each extracted page is cached under `~/.cache/structmd`. Interrupted a 500-page run at page 200? Re-run it; only pages 201+ hit the model.
- **Failure isolation** — one corrupt PDF logs an error and is skipped; the rest of the batch completes.
- **Mixed inputs** — pass `.docx`, `.pptx`, images, anything supported; they ride the same pool.

CLI equivalent:

```bash
structmd batch ./papers/ -o markdown/ --save-assets
```

## The two-stage workflow

Extract once, then iterate on the Markdown forever:

```python
from structmd import StructMDPipeline

with StructMDPipeline() as pipeline:
    # Stage 1 only: VLM runs here (slow, cached afterwards)
    doc = pipeline.extract_only("report.pdf", output_json="report.json")

    # ... inspect / hand-edit report.json ...
    # e.g. fix a heading level, correct a table cell, drop a stray footer.

    # Stage 2 only: deterministic rebuild (instant, no VLM)
    md = pipeline.build_from_json("report.json", output_md="report.md")
```

Or from the shell:

```bash
structmd report.pdf --json report.json -o report.md   # full run
vim report.json                                       # fix the JSON
structmd --from-json report.json -o report.md         # instant rebuild
```

## Figure extraction

Set `save_assets=True` (or pass `--save-assets`) and every region the VLM flagged as an image is clip-rendered from the source PDF at `assets_dpi` (default 200) into `<output_dir>/figures/`. The Markdown links the real files:

```markdown
![Diagram of the Vision Transformer architecture](figures/2010.11929_p03_1.png)
```

Design notes:

- **Clip-rendering, not object extraction** — captures vector graphics and text labels, which is what most academic figures actually are.
- **Blank-crop guard** — VLM boxes that land on empty space are detected and dropped instead of producing broken images.
- **PDF-only for now** — Office/image inputs skip asset extraction gracefully.

For standalone use (e.g. attaching assets to a cached extraction):

```python
from structmd import attach_assets

attach_assets("paper.pdf", extracted_document, "output/report.md", dpi=200)
# -> writes output/figures/*.png and annotates the document's elements
```

## Batch processing

All pages of all documents flow through one shared `asyncio` worker pool:

```python
import asyncio
from pathlib import Path

from structmd import StructMDPipeline

async def main():
    with StructMDPipeline() as pipeline:
        docs = await pipeline.process_batch_async(
            [str(p) for p in Path("papers").glob("*.pdf")]
        )
        for d in docs:
            print(d.metadata["source_path"], d.metadata["page_count"], "pages")

asyncio.run(main())
```

`process_batch(...)` is the synchronous twin — safe to call from scripts *and* inside Jupyter (it detects a running event loop and delegates to a helper thread).

Need progress visibility? Drop to `BatchProcessor` for callbacks:

```python
import asyncio
from structmd import BatchProcessor, OllamaExtractor
from structmd.config import StructMDConfig

config = StructMDConfig(ollama_model="gemma4:cloud")

def on_doc(doc_id, document):
    print(f"done: {document.source_path} ({document.page_count} pages)")

async def main():
    processor = BatchProcessor(
        OllamaExtractor(config),
        max_workers=config.ollama_max_workers,
        cache=None,                     # pass a CacheManager to enable caching
    )
    documents = await processor.process_batch(
        ["a.pdf", "b.pdf"],
        on_doc_complete=on_doc,         # also: on_page_complete(doc_id, n, page)
    )

asyncio.run(main())
```

A tqdm progress bar renders out of the box; failures on one file or page are logged and isolated.

## API reference

Everything below is importable from the package root: `from structmd import …`.

### `StructMDPipeline(config=None)`

The high-level facade. Manages converters, extractor, builder, and cache. Supports use as a context manager (`with StructMDPipeline() as p:`) which releases HTTP resources on exit.

| Method | Returns | Description |
|---|---|---|
| `process(input_path, output_json=None, output_md=None, pages=None, force=False)` | `MarkdownDocument` | Full pipeline: convert → extract → build. Writes files when paths given. |
| `extract_only(input_path, output_json=None, pages=None, force=False)` | `ExtractedDocument` | Stage 1 only. Page-level cached. |
| `build_from_json(json_path, output_md=None)` | `MarkdownDocument` | Stage 2 only. No VLM, no source file needed. |
| `process_batch(paths, pages=None, force=False, output_dir=None)` | `list[MarkdownDocument]` | Sync batch over a shared worker pool. |
| `process_batch_async(...)` | same | Async variant for running event loops. |

`pages` is a list of 1-indexed page numbers, e.g. `[1, 3, 5, 6, 7]`. Selected pages keep their true numbers throughout extraction and caching.

### `MarkdownDocument`

| Field | Type | Description |
|---|---|---|
| `title` | `str \| None` | Document title (first H1 candidate). |
| `content` | `str` | The final Markdown. |
| `metadata` | `dict` | Includes `source_path`, `page_count`, `model`, `dpi`. |

`.save(path)` writes the file (prepends `# title` when appropriate) and returns the `Path`.

### `ExtractedDocument` / `ExtractedPage` / `DocumentElement`

The Stage 1 JSON model. `ExtractedDocument` holds `pages: list[ExtractedPage]`, each holding `elements: list[DocumentElement]`. Every element carries:

| Field | Type | Description |
|---|---|---|
| `type` | `ElementType` | `heading`, `paragraph`, `table`, `list_item`, `caption`, `image`, `code_block`, `blockquote`, `footnote`, `header`, `footer`, `page_number`, `horizontal_rule` |
| `text` | `str` | Element content (for tables see `table_data`). |
| `bbox` | `BoundingBox \| None` | Pixel coordinates on the rendered page, top-left origin. |
| `page_number` | `int` | True 1-indexed page number. |
| `table_data` | `list[list[str]] \| None` | Rows for `table` elements. |
| `heading_level` | `int \| None` | Depth for `heading` elements. |
| `confidence` | `float` | VLM self-reported confidence. |
| `metadata` | `dict` | Extensible — figure assets record `asset_path` here. |

All three serialize cleanly: `.to_dict()` / `.from_dict()` / `.to_json()` / `.from_json()`, plus `ExtractedDocument.save_json(path)`.

### `BatchProcessor(extractor, max_workers=4, dpi=150, converters=None, cache=None, force=False)`

Lower-level async engine used by the pipeline. `await process_batch(paths, on_page_complete=None, on_doc_complete=None, pages=None)` returns `list[ExtractedDocument]`. Callbacks receive `(doc_id, document)` / `(doc_id, page_number, page)`.

### `AssetExtractor(dpi=200)` / `attach_assets(...)`

Figure cropping (see [Figure extraction](#figure-extraction)). `extract_assets(source_path, document, output_dir, dirname="figures")` returns the list of written relative paths and annotates image elements in place.

### `CacheManager(cache_dir="~/.cache/structmd")`

Content-addressed JSON cache. Keys are `sha256(abs_path + mtime_ns + size)`: modifying a file invalidates its entries automatically; moving it simply starts a fresh entry. Document-level (`load`/`save`) and page-level (`load_page`/`save_page`) APIs, atomic writes, `invalidate(file_path)` for explicit eviction.

### Exceptions

All derive from `structmd.core.StructMDError`: `OllamaConnectionError`, `ModelNotFoundError`, `ConversionError`, `CacheError`.

## Configuration

Precedence (highest wins): **CLI flags → env vars → `./.structmd.yaml` → `~/.config/structmd/config.yaml` → defaults**.

Every field of `StructMDConfig` is settable in all four places. The YAML file accepts nested sections or flat keys:

```yaml
# .structmd.yaml
ollama:
  url: "http://localhost:11434"
  model: "qwen2-vl:2b"
  timeout: 120          # seconds per chat call
  max_workers: 4        # concurrent page extractions

processing:
  dpi: 150
  detect_columns: true            # heuristic multi-column reading order
  merge_continued_paragraphs: true
  normalize_headings: true        # remap [1,3,3] -> [1,2,2]

output:
  include_page_numbers: true
  page_number_format: "\n<!-- Page {page} -->\n"
  table_caption_position: "before"  # or "after"

cache:
  dir: "~/.cache/structmd"

# flat keys work too, e.g.:
# cache_enabled: true
# save_assets: true
# assets_dirname: "figures"
# assets_dpi: 200
```

| Field | Default | Env var |
|---|---|---|
| `ollama_url` | `http://localhost:11434` | `STRUCTMD_OLLAMA_URL` |
| `ollama_model` | `qwen2-vl:2b` | `STRUCTMD_OLLAMA_MODEL` |
| `ollama_timeout` | `120` | `STRUCTMD_OLLAMA_TIMEOUT` |
| `ollama_max_workers` | `4` | `STRUCTMD_OLLAMA_MAX_WORKERS` |
| `dpi` | `150` | `STRUCTMD_DPI` |
| `include_page_numbers` | `True` | `STRUCTMD_INCLUDE_PAGE_NUMBERS` |
| `merge_continued_paragraphs` | `True` | `STRUCTMD_MERGE_CONTINUED_PARAGRAPHS` |
| `detect_columns` | `True` | `STRUCTMD_DETECT_COLUMNS` |
| `normalize_headings` | `True` | `STRUCTMD_NORMALIZE_HEADINGS` |
| `table_caption_position` | `before` | `STRUCTMD_TABLE_CAPTION_POSITION` |
| `cache_dir` | `~/.cache/structmd` | `STRUCTMD_CACHE_DIR` |
| `cache_enabled` | `True` | `STRUCTMD_CACHE_ENABLED` |
| `save_assets` | `False` | `STRUCTMD_SAVE_ASSETS` |
| `assets_dirname` | `figures` | `STRUCTMD_ASSETS_DIRNAME` |
| `assets_dpi` | `200` | `STRUCTMD_ASSETS_DPI` |
| `verbose` | `False` | `STRUCTMD_VERBOSE` |

See [`.structmd.yaml.example`](.structmd.yaml.example) for a ready-to-copy template.

## Caching

Extraction results live under `~/.cache/structmd`, keyed by `sha256(path + mtime + size)`:

- Modify the file → automatic miss.
- Move or copy the file → the new path starts with a fresh entry on first run.
- **Page-level entries** (`{hash}_page{N}.json`) support partial reuse: re-running a 20-page paper after adding one paragraph re-pays for exactly one page.

Measured on our benchmark: a 77-page, 5-paper run took **12m37s cold and 3.0s warm**.

Force a fresh run with `--force` / `force=True` (re-extracts and refreshes entries), or disable reads entirely with `--no-cache` / `cache_enabled=False`.

## Supported inputs & models

| Input | How | Notes |
|---|---|---|
| `.pdf` | PyMuPDF rendering at configurable DPI | page selection + figure extraction supported |
| `.docx .pptx .xlsx .odt .ods .odp .doc .ppt .xls` | LibreOffice headless → PDF | requires `soffice` |
| `.png .jpg .jpeg .tiff .bmp .webp` | direct passthrough | single page |

| Model family | Prompt template | Notes |
|---|---|---|
| Qwen2-VL / Qwen2.5-VL / Qwen3-VL | `qwen2-vl` | recommended local default (`qwen2-vl:2b`) |
| SmolVLM | `smolvlm` | tuned for short outputs |
| PaliGemma | `paligemma` | terse prompt style |
| Llama 3.2 Vision | `llama3.2-vision` | system-style instructions |
| anything else (incl. `gemma4`, `qwen3.5`, `:cloud` tags) | `default` | generic JSON contract; verified with `gemma4:cloud` and `gemma4:31b` locally |

## How it compares

| | structmd | MinerU | Marker | py-zerox | LlamaParse |
|---|---|---|---|---|---|
| Runs fully local | ✅ | ✅ | ✅ | ✅ | ❌ (cloud API) |
| Backend | any Ollama VLM (2B+) | custom OCR + layout models | Surya OCR + LLM optional | GPT-4o(-mini) via LiteLLM | proprietary |
| GPU required | ❌ (CPU-friendly small models) | recommended | recommended | ❌ (API) | ❌ |
| Intermediate format | editable JSON | MD/JSON | MD/JSON/HTML | MD | MD/JSON |
| Deterministic builder stage | ✅ | partial | partial | ❌ | ❌ |
| Per-page caching & resume | ✅ | ❌ | ❌ | ❌ | ❌ |
| Cost | free | free | free | API tokens | paid |
| Office documents | ✅ via LibreOffice | ❌ | ❌ | ❌ | limited |
| Multi-column heuristics | ✅ coordinate-based | ✅ ML | ✅ ML | ❌ | ✅ |

Pick structmd when you want **local, cheap, auditable** conversion with small models — and when being able to hand-fix the intermediate JSON matters more than squeezing out state-of-the-art accuracy on gnarly scans.

## Docker usage

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libreoffice \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "structmd[all]"

ENTRYPOINT ["structmd"]
```

Build and point it at a host Ollama:

```bash
docker build -t structmd .
docker run --rm --network host -v "$PWD:/data" -w /data \
  structmd input.pdf --url http://127.0.0.1:11434 -o output.md
```

Or use the official Ollama container alongside:

```bash
docker run -d --name ollama -v ollama:/root/.ollama -p 11434:11434 ollama/ollama
docker exec ollama ollama pull qwen2-vl:2b
```

## Development

```bash
git clone https://github.com/umar052001/structmd && cd structmd
uv sync --extra dev --extra all
uv run pytest                 # full suite (offline; HTTP mocked)
uv run black . && uv run ruff check .
uv run mypy structmd
```

Project layout follows the two stages:

```
structmd/
├── core.py           # data models: DocumentElement, BoundingBox, ExtractedDocument…
├── config.py         # layered YAML/env configuration
├── pipeline.py       # orchestrator
├── assets.py         # figure crop extraction
├── cli.py            # Click CLI
├── extractors/       # Stage 1: BaseExtractor, OllamaExtractor
├── builders/         # Stage 2: deterministic MarkdownBuilder
├── converters/       # pdf / office / image → PIL pages
├── batch/            # async worker pool processor
└── cache/            # hash-keyed JSON cache
```

## License

MIT
