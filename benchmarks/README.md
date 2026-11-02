# Extraction benchmark

From the repository's development environment:

```sh
bash benchmarks/fetch_papers.sh
python -m benchmarks.run --corpus benchmarks/corpus -o benchmarks/report.json
python -m benchmarks.run --no-live --corpus benchmarks/corpus -o benchmarks/report.json
```

The first command downloads five arXiv papers (77 pages). A live run can call
the configured Ollama endpoint and incur provider costs. `--no-live` reads only
cached pages: it never creates a model client or renders PDF pages, skips PDFs
with incomplete page caches, and exits with status 2 if nothing can be scored.
Moving a PDF changes its cache key; use the same source paths across runs.

## What the numbers mean

Metric version 2 is recorded in report metadata. Version 1 scores are not
comparable for reading order or heading grounding.

- **Content F1:** multiset word overlap against the PDF text layer. This is a
  proxy reference, not hand-annotated ground truth; punctuation, math, footnotes,
  hyphenation and the text layer's own mistakes affect the score.
- **Reading-order similarity:** word-level `SequenceMatcher`, with `autojunk`
  disabled so frequent words are not silently excluded. The PDF text layer is
  not necessarily the correct human reading order.
- **Heading text absence:** fraction of distinct normalized heading texts
  absent from normalized source text, ignoring line wrapping. This is **not**
  proof of hallucination or a heading-classification accuracy metric.
- **Table width consistency:** rectangularity, not cell accuracy. Missing table
  data is excluded. Merged cells may legitimately produce nonrectangular rows.
- **Figures:** detection count only. Asset export is not evaluated by this
  extraction-only harness; no figure-recall claim is possible without labels.

Scans without a text layer are outside this benchmark's scope. Cache identity
currently excludes model/configuration: reports record the requested model, not
verified provenance of every cached page. Use a separate cache directory for
each model when comparing configurations.

Cold model inference, warm extraction, metric computation and competitor model
initialization are different workloads. Report their timings separately; never
compare a cached structmd run with a competitor's cold conversion as throughput.
