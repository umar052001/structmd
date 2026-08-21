"""Pipeline tests: cache hits, JSON round-trips, batch wiring — all offline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import pytest
from PIL import Image

import structmd
from structmd.config import StructMDConfig
from structmd.core import ExtractedDocument, ExtractedPage
from structmd.pipeline import StructMDPipeline


class StubExtractor:
    """Counts calls; returns one paragraph per page."""

    def __init__(self, config=None) -> None:
        self.config = config or StructMDConfig()
        self.call_count = 0
        self.page_calls: List[int] = []

    def extract_document(self, images, source_path=None) -> ExtractedDocument:
        self.call_count += 1
        pages = [
            ExtractedPage(
                page_number=i + 1,
                width=float(img.width),
                height=float(img.height),
            )
            for i, img in enumerate(images)
        ]
        return ExtractedDocument(source_path=source_path, page_count=len(pages), pages=pages)

    def extract_page(self, image, page_number: int) -> ExtractedPage:
        self.call_count += 1
        self.page_calls.append(page_number)
        return ExtractedPage(
            page_number=page_number, width=float(image.width), height=float(image.height)
        )

    async def extract_page_async(self, image, page_number):
        return ExtractedPage(
            page_number=page_number, width=float(image.width), height=float(image.height)
        )

    def close(self) -> None:
        pass


class SinglePageConverter:
    """Serves a fixed image for any .stub path."""

    _supported_extensions = [".stub"]

    def __init__(self) -> None:
        pass

    def supports(self, path: str) -> bool:
        return path.endswith(".stub")

    def convert(
        self, path: str, pages: Optional[List[int]] = None
    ) -> List[Tuple[int, Image.Image]]:
        numbers = pages if pages is not None else [1]
        return [(n, Image.new("RGB", (100, 120))) for n in numbers]


@pytest.fixture()
def pipeline(tmp_path: Path, monkeypatch) -> StructMDPipeline:
    cfg = StructMDConfig(cache_dir=str(tmp_path / "cache"))
    pipe = StructMDPipeline(config=cfg)
    pipe.extractor = StubExtractor(cfg)
    pipe.converters = [SinglePageConverter()]
    yield pipe
    pipe.close()


@pytest.fixture()
def stub_file(tmp_path: Path) -> Path:
    f = tmp_path / "doc.stub"
    f.write_bytes(b"stub")
    return f


class TestProcess:
    def test_end_to_end(self, pipeline: StructMDPipeline, stub_file: Path) -> None:
        md = pipeline.process(str(stub_file))
        assert md.metadata["page_count"] == 1

    def test_cache_hit_skips_extraction(self, pipeline: StructMDPipeline, stub_file: Path) -> None:
        pipeline.process(str(stub_file))
        first_calls = pipeline.extractor.call_count
        assert first_calls == 1
        # Second run must not invoke the extractor.
        pipeline.process(str(stub_file))
        assert pipeline.extractor.call_count == 1

    def test_page_selection_uses_page_cache(
        self, pipeline: StructMDPipeline, stub_file: Path
    ) -> None:
        """Page subsets cache per page: the re-run must not re-extract."""
        pipeline.process(str(stub_file), pages=[1])
        assert pipeline.extractor.call_count == 1
        pipeline.process(str(stub_file), pages=[1])
        assert pipeline.extractor.call_count == 1  # served from page cache

    def test_force_re_extracts_and_refreshes_cache(
        self, pipeline: StructMDPipeline, stub_file: Path
    ) -> None:
        pipeline.process(str(stub_file), pages=[1])
        assert pipeline.extractor.call_count == 1
        pipeline.process(str(stub_file), pages=[1], force=True)
        assert pipeline.extractor.call_count == 2  # bypassed cache read

    def test_no_cache_config_disables_caching(
        self, pipeline: StructMDPipeline, stub_file: Path
    ) -> None:
        pipeline.config.cache_enabled = False
        pipeline.process(str(stub_file))
        pipeline.process(str(stub_file))
        assert pipeline.extractor.call_count == 2

    def test_output_json_written(
        self, pipeline: StructMDPipeline, stub_file: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out" / "extraction.json"
        pipeline.process(str(stub_file), output_json=str(out))
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["source_path"] == str(stub_file)

    def test_output_md_written(
        self, pipeline: StructMDPipeline, stub_file: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out" / "result.md"
        pipeline.process(str(stub_file), output_md=str(out))
        assert out.is_file()


class TestExtractOnlyAndBuildFromJson:
    def test_extract_only(
        self, pipeline: StructMDPipeline, stub_file: Path, tmp_path: Path
    ) -> None:
        json_out = tmp_path / "extraction.json"
        doc = pipeline.extract_only(str(stub_file), output_json=str(json_out))
        assert doc.page_count == 1
        assert json_out.is_file()

    def test_build_from_json_roundtrip(
        self, pipeline: StructMDPipeline, stub_file: Path, tmp_path: Path
    ) -> None:
        json_out = tmp_path / "extraction.json"
        pipeline.extract_only(str(stub_file), output_json=str(json_out))

        # No extractor needed at all for the build stage.
        pipeline.extractor.call_count = 0
        pipeline.build_from_json(str(json_out), output_md=str(tmp_path / "out.md"))
        assert (tmp_path / "out.md").is_file()
        assert pipeline.extractor.call_count == 0

    def test_build_from_invalid_json_raises(
        self, pipeline: StructMDPipeline, tmp_path: Path
    ) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid", encoding="utf-8")
        with pytest.raises(ValueError, match="not a valid structmd extraction"):
            pipeline.build_from_json(str(bad))


class TestBatch:
    async def test_process_batch_async(
        self, pipeline: StructMDPipeline, stub_file: Path, tmp_path: Path
    ) -> None:
        files = []
        for i in range(2):
            f = tmp_path / f"b{i}.stub"
            f.write_bytes(b"x")
            files.append(str(f))
        docs = await pipeline.process_batch_async(files)
        assert len(docs) == 2
        assert all(d.metadata["page_count"] == 1 for d in docs)

    def test_sync_process_batch(self, pipeline: StructMDPipeline, tmp_path: Path) -> None:
        files = [str(tmp_path / "s.stub")]
        (tmp_path / "s.stub").write_bytes(b"x")
        docs = pipeline.process_batch(files)
        assert len(docs) == 1


class TestPublicApi:
    def test_version_and_exports(self) -> None:
        assert structmd.__version__
        for name in (
            "StructMDPipeline",
            "OllamaExtractor",
            "MarkdownBuilder",
            "ExtractedDocument",
            "BoundingBox",
        ):
            assert hasattr(structmd, name)

    def test_default_config_used_when_none(self, monkeypatch, tmp_path) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path))  # isolate global config
        pipe = StructMDPipeline()
        assert pipe.config.ollama_url == "http://localhost:11434"
