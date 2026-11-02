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
