"""CLI tests: argument parsing, from-json flow, and a fully mocked E2E run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

pytest.importorskip("pymupdf", reason="PyMuPDF extra not installed")

from structmd.cli import main, parse_pages  # noqa: E402


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


class TestParsePages:
    def test_single(self) -> None:
        assert parse_pages("3") == [3]

    def test_comma_separated(self) -> None:
        assert parse_pages("1,3,5") == [1, 3, 5]

    def test_range(self) -> None:
        assert parse_pages("5-10") == [5, 6, 7, 8, 9, 10]

    def test_mixed(self) -> None:
        assert parse_pages("1,3,5-10") == [1, 3, 5, 6, 7, 8, 9, 10]

    @pytest.mark.parametrize("bad", ["0", "abc", "5-2", "1-", "", "-3"])
    def test_invalid_specs_raise(self, bad: str) -> None:
        from click import BadParameter

        with pytest.raises(BadParameter):
            parse_pages(bad)


class TestBasicCli:
    def test_version(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "structmd" in result.output

    def test_no_args_shows_help(self, runner: CliRunner) -> None:
        result = runner.invoke(main, [])
        assert result.exit_code == 2
        assert "Usage" in result.output

    def test_missing_input_file(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["/no/such/file.pdf", "-o", "out.md"])
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_bad_pages_spec(self, runner: CliRunner, tmp_path: Path) -> None:
        f = tmp_path / "x.pdf"
        f.write_bytes(b"%PDF-fake")
        result = runner.invoke(main, [str(f), "--pages", "0", "-o", "o.md"])
        assert result.exit_code == 2  # UsageError


class TestFromJson:
    def test_from_json_builds_markdown_without_vlm(self, runner: CliRunner, tmp_path: Path) -> None:
        sample = json.loads(
            (Path(__file__).parent / "fixtures" / "sample_extraction.json").read_text(
                encoding="utf-8"
            )
        )
        extraction = tmp_path / "extraction.json"
        extraction.write_text(json.dumps(sample), encoding="utf-8")
        out_md = tmp_path / "report.md"

        result = runner.invoke(main, ["--from-json", str(extraction), "-o", str(out_md)])
        assert result.exit_code == 0, result.output
        content = out_md.read_text(encoding="utf-8")
        assert "# Quarterly Business Report" in content
        assert "| Metric | Q1 | Q2 |" in content

    def test_from_json_missing_file(self, runner: CliRunner, tmp_path: Path) -> None:
        result = runner.invoke(main, ["--from-json", str(tmp_path / "nope.json"), "-o", "o.md"])
        assert result.exit_code != 0


class TestEndToEndWithMockedOllama:
    @pytest.fixture()
    def one_page_pdf(self, tmp_path: Path) -> Path:
        import pymupdf

        path = tmp_path / "tiny.pdf"
        doc = pymupdf.open()
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 100), "Hello Structured World", fontsize=20)
        doc.save(str(path))
        doc.close()
        return path

    def test_convert_pdf_to_markdown_file(
        self, runner: CliRunner, tmp_path: Path, one_page_pdf: Path, monkeypatch
    ) -> None:
        import httpx
        import respx

        extraction = {
            "elements": [
                {
                    "type": "heading",
                    "bbox": [70, 80, 400, 110],
                    "text": "Hello Structured World",
                    "level": 1,
                },
                {
                    "type": "paragraph",
                    "bbox": [70, 130, 500, 160],
                    "text": "Body text extracted by the mock.",
                },
            ]
        }
        with respx.mock:
            respx.get("http://localhost:11434/api/tags").mock(
                return_value=httpx.Response(200, json={"models": [{"name": "qwen2-vl:2b"}]})
            )
            route = respx.post("http://localhost:11434/api/chat").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(extraction),
                        }
                    },
                )
            )
            out_md = tmp_path / "nested" / "out.md"
            result = runner.invoke(
                main,
                [
                    str(one_page_pdf),
                    "-o",
                    str(out_md),
                    "--json",
                    str(tmp_path / "extraction.json"),
                    "--no-page-numbers",
                ],
            )

        assert result.exit_code == 0, result.output
        assert out_md.is_file()
        content = out_md.read_text(encoding="utf-8")
        assert "# Hello Structured World" in content
        assert "Body text extracted by the mock." in content
        # The chat request carried an image and stream:false.
        body = json.loads(route.calls.last.request.read())
        assert body["stream"] is False
        assert body["messages"][0]["images"]

    def test_model_not_available_gives_pull_hint(
        self, runner: CliRunner, tmp_path: Path, one_page_pdf: Path
    ) -> None:
        import httpx
        import respx

        with respx.mock:
            respx.get("http://localhost:11434/api/tags").mock(
                return_value=httpx.Response(200, json={"models": [{"name": "gemma4:latest"}]})
            )
            result = runner.invoke(main, [str(one_page_pdf), "-o", str(tmp_path / "o.md")])

        assert result.exit_code != 0
        assert "ollama pull" in result.output


class TestBatchCommand:
    def test_batch_writes_markdown_files(
        self, runner: CliRunner, tmp_path: Path, monkeypatch
    ) -> None:
        """Batch is tested with a stubbed pipeline (HTTP covered elsewhere)."""
        from structmd.core import MarkdownDocument
        from structmd.pipeline import StructMDPipeline

        files = []
        for name in ("a.pdf", "b.docx"):
            f = tmp_path / name
            f.write_bytes(b"x")
            files.append(str(f))

        calls = {"n": 0}

        def fake_process_batch(self, paths, pages=None):
            calls["n"] += 1
            return [
                MarkdownDocument(
                    title=f"T{i}", content=f"# T{i}\n\nbody {i}", metadata={"source_path": p}
                )
                for i, p in enumerate(paths)
            ]

        monkeypatch.setattr(StructMDPipeline, "process_batch", fake_process_batch)

        outdir = tmp_path / "mds"
        result = runner.invoke(main, ["batch", *files, "-o", str(outdir)])
        assert result.exit_code == 0, result.output
        assert calls["n"] == 1
        assert (outdir / "a.md").is_file()
        assert (outdir / "b.md").is_file()

    def test_batch_missing_file_fails_fast(self, runner: CliRunner, tmp_path: Path) -> None:
        result = runner.invoke(main, ["batch", str(tmp_path / "ghost.pdf")])
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_batch_pages_reach_pipeline(
        self, runner: CliRunner, tmp_path: Path, monkeypatch
    ) -> None:
        """--pages given to batch must reach process_batch (not be ignored)."""
        from structmd.core import MarkdownDocument
        from structmd.pipeline import StructMDPipeline

        f = tmp_path / "a.pdf"
        f.write_bytes(b"x")
        seen = {}

        def fake_process_batch(self, paths, pages=None):
            seen["pages"] = pages
            return [MarkdownDocument(metadata={"source_path": p}) for p in paths]

        monkeypatch.setattr(StructMDPipeline, "process_batch", fake_process_batch)
        result = runner.invoke(main, ["batch", str(f), "--pages", "1,3", "-o", str(tmp_path / "o")])
        assert result.exit_code == 0, result.output
        assert seen["pages"] == [1, 3]
