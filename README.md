# structmd

[![CI](https://github.com/umar052001/structmd/actions/workflows/ci.yml/badge.svg)](https://github.com/umar052001/structmd/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/structmd?color=6d5cff)](https://pypi.org/project/structmd/)
[![Python](https://img.shields.io/pypi/pyversions/structmd)](https://pypi.org/project/structmd/)
[![Docs](https://img.shields.io/badge/docs-github.io-00b3e6)](https://umar052001.github.io/structmd/)
[![License: MIT](https://img.shields.io/badge/license-MIT-ff4f9a)](LICENSE)

**Convert PDFs, Office documents, and images into structured Markdown using small Vision-Language Models served by [Ollama](https://ollama.com).**

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

**JSON is the single source of truth.** The VLM never writes Markdown directly. It produces an inspectable, editable, cacheable JSON description of the document layout; a deterministic builder ("the cutter") turns that JSON into Markdown. Same JSON in → identical Markdown out, every time.

## Why this architecture?

- **Small-model friendly.** 2B–3B VLMs are bad at writing clean Markdown but decent at describing layout as JSON. structmd plays to that strength.
- **Debuggable.** Bad conversion? Open the JSON and see exactly what the model saw. Fix it by hand and rebuild without re-running inference.
- **Cheap to iterate.** Extraction is cached by file hash + mtime. Re-tune Markdown output (heading levels, table captions, column handling) instantly from cached JSON.
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
- `[pdf]` — PyMuPDF for PDF rendering
- `[office]` — PyMuPDF (Office docs go through LibreOffice → PDF)
- `[all]` — everything above

Office documents additionally require **LibreOffice** (`soffice`) on your PATH:

```bash
sudo apt install libreoffice        # Debian/Ubuntu
brew install --cask libreoffice     # macOS
```

## Ollama setup

structmd works with **both local and cloud models** — the same API, the same code path.

### Local (on-prem) models

```bash
# install ollama, then pull a small vision model:
ollama pull qwen2-vl:2b          # ~1.6 GB, good default
# alternatives:
ollama pull smolvlm              # very light
ollama pull llama3.2-vision      # larger, stronger
```

### Cloud models (no GPU needed)

Ollama can transparently offload larger vision models to [ollama.com](https://ollama.com) while your tooling keeps talking to `localhost:11434`. Sign in once, pull the cloud tag, and use it like any local model:

```bash
ollama signin                    # one-time account link
ollama pull gemma4:cloud         # registers the cloud model (no big download)

structmd scan.pdf --model gemma4:cloud -o output.md
```

Cloud vision models currently include `gemma4:cloud`, `qwen3.5:*-cloud`, `kimi-k2.6:cloud`, and friends — see the [cloud catalog](https://ollama.com/search?c=cloud). Notes:

- **Throughput**: cloud models are typically much faster than CPU-bound local inference (a 3-page PDF took ~18s via `gemma4:cloud` vs >120s/page locally on CPU).
- **Timeouts**: large frontier models can take longer per page; raise the budget with `--timeout`-style config (`STRUCTMD_OLLAMA_TIMEOUT=300`) or the YAML key `ollama.timeout`.
- **Privacy**: pages are sent to Ollama's cloud service. For sensitive documents, stick to local models — structmd treats both identically.

Verify whatever endpoint you use is up:

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

# page selection (1-indexed, ranges allowed)
structmd input.pdf --pages 1,3,5-10 -o output.md

# custom config file
structmd --config ~/.structmd.yaml input.pdf -o out.md
```

### Python API

```python
from structmd import StructMDPipeline

pipeline = StructMDPipeline()

result = pipeline.process(
    "document.pdf",
    output_json="extraction.json",   # optional: keep Stage 1 output
    output_md="output.md",           # optional: write final Markdown
)

print(result.title)                          # extracted from first heading
print(result.metadata["page_count"])         # 12
```

## The two-stage workflow

This is where structmd's design pays off. Extract once, then iterate on the Markdown forever:

```python
from structmd import StructMDPipeline

pipeline = StructMDPipeline()

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

## Supported inputs & models

| Input | How | Notes |
|---|---|---|
| `.pdf` | PyMuPDF rendering at configurable DPI | page selection supported |
| `.docx .pptx .xlsx .odt .ods .odp .doc .ppt .xls` | LibreOffice headless → PDF | requires `soffice` |
| `.png .jpg .jpeg .tiff .bmp .webp` | direct passthrough | single page |

| Model family | Prompt template | Notes |
|---|---|---|
| Qwen2-VL / Qwen2.5-VL / Qwen3-VL | `qwen2-vl` | recommended local default (`qwen2-vl:2b`) |
| SmolVLM | `smolvlm` | tuned for short outputs |
| PaliGemma | `paligemma` | terse prompt style |
| Llama 3.2 Vision | `llama3.2-vision` | system-style instructions |
| anything else (incl. `gemma4`, `qwen3.5`, `:cloud` tags) | `default` | generic JSON contract; verified with `gemma4:cloud` and `gemma4:31b` locally |

## Configuration

Precedence (highest wins): **env vars → `./.structmd.yaml` → `~/.config/structmd/config.yaml` → defaults → CLI flags** (CLI flags always win at runtime).

```yaml
# .structmd.yaml
ollama:
  url: "http://localhost:11434"
  model: "qwen2-vl:2b"
  timeout: 120          # seconds per chat call
  max_workers: 4        # async workers for batch processing

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
```

Every key can also be set via environment variables: `STRUCTMD_OLLAMA_URL`, `STRUCTMD_OLLAMA_MODEL`, `STRUCTMD_DPI`, `STRUCTMD_CACHE_DIR`, `STRUCTMD_VERBOSE`, …

See [`.structmd.yaml.example`](.structmd.yaml.example) for a ready-to-copy template.

## Caching

Extraction results are cached under `~/.cache/structmd` keyed by `sha256(path + mtime + size)`:

- Move the file → cache still valid.
- Modify the file → automatic miss.
- Page-level entries (`{hash}_page{N}.json`) support partial reuse.

Force a fresh run with `structmd` after touching the file, or clear the cache directory.

## Batch processing

All pages of all documents flow through one shared `asyncio` worker pool:

```python
import asyncio
from structmd import StructMDPipeline

async def main():
    pipeline = StructMDPipeline()
    docs = await pipeline.process_batch_async(["a.pdf", "b.docx", "c.png"])
    for d in docs:
        print(d.title, d.metadata["page_count"])

asyncio.run(main())
```

Callbacks are available on the lower-level `BatchProcessor` (`on_page_complete`, `on_doc_complete`), with tqdm progress out of the box.

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

## How it compares

| | structmd | MinerU | Marker | py-zerox | LlamaParse |
|---|---|---|---|---|---|
| Runs fully local | ✅ | ✅ | ✅ | ✅ | ❌ (cloud API) |
| Backend | any Ollama VLM (2B+) | custom OCR + layout models | Surya OCR + LLM optional | GPT-4o(-mini) via LiteLLM | proprietary |
| GPU required | ❌ (CPU-friendly small models) | recommended | recommended | ❌ (API) | ❌ |
| Intermediate format | editable JSON | MD/JSON | MD/JSON/HTML | MD | MD/JSON |
| Deterministic builder stage | ✅ | partial | partial | ❌ | ❌ |
| Cost | free | free | free | API tokens | paid |
| Office documents | ✅ via LibreOffice | ❌ | ❌ | ❌ | limited |
| Multi-column heuristics | ✅ coordinate-based | ✅ ML | ✅ ML | ❌ | ✅ |

Pick structmd when you want **local, cheap, auditable** conversion with small models — and when being able to hand-fix the intermediate JSON matters more than squeezing out state-of-the-art accuracy on gnarly scans.

## Development

```bash
git clone <repo> && cd structmd
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
├── cli.py            # Click CLI
├── extractors/       # Stage 1: BaseExtractor, OllamaExtractor
├── builders/         # Stage 2: deterministic MarkdownBuilder
├── converters/       # pdf / office / image → PIL pages
├── batch/            # async worker pool processor
└── cache/            # hash-keyed JSON cache
```

## License

MIT
