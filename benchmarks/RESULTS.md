# Local benchmark results

These are five-paper proxy measurements, **not** an OmniDocBench result or a
claim that structmd outperforms other parsers. Metric definitions and limitations
are in [README.md](README.md). Metric version: **2**.

## structmd cached extraction

| arXiv paper | Pages | Content F1 | Word-order similarity | Heading text absence |
|---|---:|---:|---:|---:|
| 1301.3781 | 12 | 0.962 | 0.956 | 0.045 |
| 1512.03385 | 12 | 0.862 | 0.766 | 0.217 |
| 1706.03762 | 15 | 0.912 | 0.908 | 0.000 |
| 1810.04805 | 16 | 0.882 | 0.835 | 0.083 |
| 2010.11929 | 22 | 0.897 | 0.885 | 0.000 |
| **Macro average** | **77** | **0.903** | **0.870** | **0.069** |

The cached pages came from earlier runs; their model/configuration provenance
has not been verified. These numbers characterize those saved outputs, not a
fresh evaluation of the default model. No new model inference was performed.

The previous 0.743 "heading hallucination" number was primarily a metric bug:
it required exact matches to individual PDF text lines. Version 2 ignores line
wrapping and labels this only as heading **text absence**. The reading-order
metric also changed from characters to words with `autojunk=False`; its version
1 and version 2 numbers must not be compared as extraction improvements.

## Warm-cache performance check

Machine: Intel Core i5-10310U, 4 cores / 8 threads, Linux, Python 3.11.
One local measurement on the same 77 pages, at 150 DPI:

| Workload | Seconds |
|---|---:|
| Previous algorithm: render all pages, then read cached pages | 2.777 |
| Updated `extract_only`: read cached pages without rendering | 0.059 |

This is about **47× faster warm extraction replay**, not faster inference.
The comparison directly exercised the old render-then-cache workload versus the
new extraction path, not two full application releases. Metric calculation was
excluded; a separate complete offline scoring run took 3.2 seconds.
Timings are a single-machine observation, not a controlled throughput claim.

Regression tests enforce the important invariant: warm PDF pages never render;
partially cached documents render only missing pages; `force=True` still renders
and extracts all requested pages.

## Real figure-cache smoke test

On the same five cached documents, `attach_assets` plus the Markdown builder
produced **31 accepted crops and zero missing asset files**. Using a fresh
temporary crop cache, the first pass rendered 32 candidate regions in 1.627s.
The warm pass took 0.112s and rendered only the one blank region rejected on
both passes; all 31 accepted crops were reused. Rejected crops are deliberately
not cached. Both passes verified that every returned asset path exists.

## PyMuPDF4LLM baseline

PyMuPDF4LLM **1.28.2**, PyMuPDF **1.28.2**, Python 3.12, same machine and
five PDFs. Legacy text-layer mode (`use_layout(False)`), no OCR, no image
extraction; `force_text=True`, `margins=0`, `table_strategy="lines_strict"`,
page chunks enabled, default automatic heading detection.

| arXiv paper | Conversion seconds | Content F1 | Word-order similarity |
|---|---:|---:|---:|
| 1301.3781 | 4.17 | 0.9749 | 0.9334 |
| 1512.03385 | 8.11 | 0.9133 | 0.8863 |
| 1706.03762 | 4.00 | 0.9186 | 0.9013 |
| 1810.04805 | 5.62 | 0.9117 | 0.9005 |
| 2010.11929 | 11.48 | 0.8779 | 0.8542 |
| **Total / macro average** | **33.38** | **0.9193** | **0.8951** |

Markdown was parsed with markdown-it-py 4.2.0 before applying the same v2 text
metrics. Visible text, code, link labels and table cells were retained; markup,
link URLs and image alt text were omitted. This differs from structmd's
prose-and-table IR projection, so these are approximate text-preservation
baselines, not equivalent structural-accuracy scores. The reference is itself a
PyMuPDF text extraction, which favors text-layer-preserving approaches.

On these proxies the baseline's averages exceed structmd's cached outputs
(F1 0.903, order 0.870). The biggest gap is the ResNet paper (1512.03385);
inspect its tables and reading order before changing the model or prompt.
Structmd scores higher on ViT (2010.11929). Neither observation establishes
overall parser superiority. Scans, formulas, figure recall and table cell
accuracy were not evaluated.

Conversion timing includes PDF opening and header scanning, excludes output
writes and scoring, and had no warmup. Import initialization took another
0.37s. This fresh conversion must not be compared to structmd's cached replay
as an inference speed result; repeat-run variance was not measured.

Reproduction scripts, source PDF hashes, exact settings, metric-source
snapshots, raw Markdown and scores are saved in the separate lab under
`structmd-lab/comparison/pymupdf4llm/`. No competitor package was added to
structmd's dependencies.

### Docling attempt

No Docling result is available: CPU-only PyTorch installation timed out twice
before conversion (240s and 180s download attempts; **0/5 PDFs processed**).
The CPU benchmark script and attempt manifest are saved under
`structmd-lab/comparison/docling/`. This is an environment/download blocker,
not evidence about Docling's speed or quality.
