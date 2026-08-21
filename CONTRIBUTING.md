# Contributing to structmd

Thanks for your interest in improving structmd! This guide gets you from clone to merged PR.

## Project philosophy

structmd enforces a **strict two-stage architecture**:

1. **Stage 1 (extraction):** a VLM describes page images as typed, positioned JSON.
2. **Stage 2 (building):** a deterministic, pure-Python builder converts that JSON to Markdown.

The VLM never writes Markdown directly. PRs that blur this boundary will be asked to restructure —
it is the property that makes output reproducible and auditable.

## Development setup

We use [uv](https://docs.astral.sh/uv/) for everything (envs, deps, builds). One command bootstraps:

```bash
git clone https://github.com/umar052001/structmd.git
cd structmd
uv sync --extra all --extra dev
```

Verify you are green before touching anything:

```bash
uv run pytest          # 154 tests, fully offline
```

> **The entire test suite runs without Ollama.** HTTP traffic is mocked with `respx`,
> extraction is stubbed, and fixtures are generated locally. Never write a test that
> requires a live model server — CI has none.

### Optional local services

- **LibreOffice** — needed only for `tests/test_converters.py` office tests; they skip automatically if absent.
- **A real VLM** — for manual validation only (`ollama pull gemma4:cloud` works well); never required by tests.

## Making changes

### 1. Branch

```bash
git checkout -b feat/my-feature   # or fix/..., docs/...
```

### 2. Code standards

All four gates must pass — CI enforces them on every push:

```bash
uv run ruff check .       # linting
uv run black .            # formatting
uv run mypy structmd      # strict-ish typing
uv run pytest             # full suite
```

House rules:

- **Python 3.9 compatible syntax** in `structmd/` (we test 3.9–3.13). Use `from __future__ import annotations`.
- **Type hints everywhere** in library code; tests included where practical.
- **No new dependencies** without discussion — this library stays lean on purpose.
- **Docstrings** on every public class/function, with a short example for API surface.
- Keep prompts for new model families in `extractors/ollama.py::PROMPTS` — add a family key,
  extend `detect_model_family()`, and add unit tests with mocked responses.

### 3. Tests

- New feature → new tests, same PR. Bug fix → regression test that fails without the fix.
- Use `respx` for HTTP mocking (see `tests/test_ollama_extractor.py`) and `CliRunner` for CLI behavior.
- Determinism matters: if your change affects `MarkdownBuilder` output, prove byte-stability
  (see `tests/test_markdown_builder.py` patterns).

### 4. Commit & PR

- Small, focused commits; imperative subject lines (`fix: suppress page break when paragraphs merge`).
- PRs should describe *what* and *why*; the diff shows *how*.
- Link related issues. One feature per PR keeps review fast.

## Reporting issues

Great bug reports include:

- structmd version (`structmd --version`) and Python version
- The **extraction JSON** for the failing document (this is exactly why we produce it)
- The command run and full stderr (`-v` output)
- A minimal public document reproducing it, if possible

For model-quality issues (bad headings, missed tables), note the exact model tag — prompt
profiles differ per family.

## Release process (maintainers)

1. Update version in `pyproject.toml` (`uv` will pick it up) and bump `schema_version`
   only if the extraction JSON format changed.
2. Merge to `main` — CI must be green.
3. Tag and push:
   ```bash
   git tag v0.x.y && git push origin v0.x.y
   ```
4. The `Publish` workflow re-runs the full suite, builds sdist+wheel, checks metadata with
   `twine`, and publishes via PyPI Trusted Publishing (OIDC — no tokens in the repo).
5. Docs deploy automatically on pushes to `main` under `docs/`.

## License

By contributing you agree your work is released under the repository's MIT License.
