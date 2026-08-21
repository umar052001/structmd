"""Tests for BatchProcessor using a fake extractor (no HTTP)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import List, Tuple

import pytest
from PIL import Image

from structmd.batch.processor import BatchProcessor
from structmd.converters.base import BaseConverter
from structmd.core import ExtractedPage


class FakeExtractor:
    """Deterministic fake: one element per page, records call order."""

    def __init__(self, delay: float = 0.0, fail_pages: set = frozenset()) -> None:
        self.delay = delay
        self.fail_pages = set(fail_pages)
        self.calls: List[int] = []
        self.lock = asyncio.Lock()

    async def extract_page_async(self, image: Image.Image, page_number: int) -> ExtractedPage:
        async with self.lock:
            self.calls.append(page_number)
        if page_number in self.fail_pages:
            raise RuntimeError(f"injected failure on page {page_number}")
        if self.delay:
            await asyncio.sleep(self.delay)
        return ExtractedPage(
            page_number=page_number,
            width=float(image.width),
            height=float(image.height),
        )

    def extract_page(self, image, page_number):
        raise NotImplementedError

    def extract_document(self, images, source_path=None):
        raise NotImplementedError


class StubConverter(BaseConverter):
    """Serves pre-rendered images regardless of path; honors page selection."""

    _supported_extensions = [".stub"]

    def __init__(self, pages_per_doc: int = 3) -> None:
        self.pages_per_doc = pages_per_doc
        self.last_pages_seen = None

    def supports(self, path: str) -> bool:
        return path.endswith(".stub")

    def convert(self, path: str, pages=None) -> List[Tuple[int, Image.Image]]:
        self.last_pages_seen = pages
        numbers = pages if pages is not None else list(range(1, self.pages_per_doc + 1))
        return [(n, Image.new("RGB", (10 + i, 20), "white")) for i, n in enumerate(numbers)]


@pytest.fixture()
def stub_files(tmp_path: Path) -> List[str]:
    return [str(tmp_path / f"doc{i}.stub").replace(".stub", ".stub") for i in range(1, 4)]


def make_processor(extractor, workers: int = 2) -> BatchProcessor:
    return BatchProcessor(
        extractor,
        max_workers=workers,
        converters=[StubConverter(pages_per_doc=3)],
    )


class TestBatchProcessing:
    async def test_all_documents_assembled_in_order(self, stub_files) -> None:
        processor = make_processor(FakeExtractor())
        docs = await processor.process_batch(stub_files)
        assert len(docs) == 3
        assert [d.source_path for d in docs] == stub_files
        for doc in docs:
            assert doc.page_count == 3
            assert [p.page_number for p in doc.pages] == [1, 2, 3]

    async def test_worker_concurrency_respected(self, stub_files) -> None:
        """With 4 workers and slow pages, several pages run simultaneously."""
        concurrent = {"now": 0, "peak": 0}

        class TrackingExtractor(FakeExtractor):
            async def extract_page_async(self, image, page_number):
                concurrent["now"] += 1
                concurrent["peak"] = max(concurrent["peak"], concurrent["now"])
                try:
                    return await super().extract_page_async(image, page_number)
                finally:
                    concurrent["now"] -= 1

        processor = make_processor(TrackingExtractor(delay=0.05), workers=4)
        await processor.process_batch(stub_files)
        assert concurrent["peak"] > 1

    async def test_page_failures_do_not_sink_batch(self, stub_files) -> None:
        extractor = FakeExtractor(fail_pages={2})
        processor = make_processor(extractor)
        docs = await processor.process_batch(stub_files)
        assert len(docs) == 3
        # Page 2 missing everywhere; remaining pages intact and ordered.
        for doc in docs:
            assert [p.page_number for p in doc.pages] == [1, 3]

    async def test_unsupported_and_missing_paths_skipped(self, tmp_path) -> None:
        processor = make_processor(FakeExtractor())
        paths = [str(tmp_path / "good.stub"), str(tmp_path / "bad.zip"), "/no/file.pdf"]
        docs = await processor.process_batch(paths)
        assert len(docs) == 1
        assert docs[0].source_path == str(tmp_path / "good.stub")

    async def test_callbacks_fire(self, stub_files) -> None:
        page_calls = []
        doc_calls = []
        processor = make_processor(FakeExtractor())
        await processor.process_batch(
            stub_files,
            on_page_complete=lambda doc_id, n, page: page_calls.append((doc_id, n)),
            on_doc_complete=lambda doc_id, doc: doc_calls.append(doc_id),
        )
        assert len(page_calls) == 9
        assert len(doc_calls) == 3

    async def test_async_callbacks_fire(self, stub_files) -> None:
        events = []

        async def on_page(doc_id, n, page):
            events.append(("page", n))

        async def on_doc(doc_id, doc):
            events.append(("doc", doc.page_count))

        processor = make_processor(FakeExtractor())
        await processor.process_batch(
            stub_files[:1], on_page_complete=on_page, on_doc_complete=on_doc
        )
        assert ("page", 1) in events and ("doc", 3) in events

    async def test_empty_input_returns_empty(self) -> None:
        processor = make_processor(FakeExtractor())
        assert await processor.process_batch([]) == []

    async def test_pages_selection_passed_to_converters(self, stub_files) -> None:
        converter = StubConverter(pages_per_doc=5)
        processor = BatchProcessor(FakeExtractor(), max_workers=2, converters=[converter])
        docs = await processor.process_batch(stub_files[:1], pages=[1, 3])
        assert converter.last_pages_seen == [1, 3]
        # Stub honors the selection; only selected pages come back.
        assert docs[0].page_count == 2
        assert [p.page_number for p in docs[0].pages] == [1, 3]

    async def test_true_page_numbers_preserved(self, stub_files) -> None:
        """Selected page 4 must stay labeled 4 (not renumbered to 1)."""
        converter = StubConverter(pages_per_doc=6)
        processor = BatchProcessor(FakeExtractor(), max_workers=2, converters=[converter])
        docs = await processor.process_batch(stub_files[:1], pages=[4, 6])
        assert [p.page_number for p in docs[0].pages] == [4, 6]

    async def test_page_cache_hits_skip_extraction(self, stub_files, tmp_path) -> None:
        """Second run must serve every page from cache: zero extractor calls."""
        from structmd.cache.manager import CacheManager

        for f in stub_files:  # cache keys require a real stat-able file
            Path(f).write_bytes(b"stub")
        cache = CacheManager(str(tmp_path / "cache"))
        first = BatchProcessor(
            FakeExtractor(), max_workers=2, converters=[StubConverter()], cache=cache
        )
        docs1 = await first.process_batch(stub_files)
        assert sum(d.page_count for d in docs1) == 9

        extractor2 = FakeExtractor()
        second = BatchProcessor(
            extractor2, max_workers=2, converters=[StubConverter()], cache=cache
        )
        docs2 = await second.process_batch(stub_files)
        assert extractor2.calls == []  # nothing re-extracted
        assert [p.page_number for d in docs2 for p in d.pages] == [
            p.page_number for d in docs1 for p in d.pages
        ]

    async def test_force_bypasses_cache_reads(self, stub_files, tmp_path) -> None:
        from structmd.cache.manager import CacheManager

        for f in stub_files:
            Path(f).write_bytes(b"stub")
        cache = CacheManager(str(tmp_path / "cache"))
        warm = BatchProcessor(
            FakeExtractor(), max_workers=2, converters=[StubConverter()], cache=cache
        )
        await warm.process_batch(stub_files[:1])

        extractor2 = FakeExtractor()
        forced = BatchProcessor(
            extractor2,
            max_workers=2,
            converters=[StubConverter()],
            cache=cache,
            force=True,
        )
        await forced.process_batch(stub_files[:1])
        assert sorted(extractor2.calls) == [1, 2, 3]  # re-extracted despite cache

    async def test_metadata_records_model(self, stub_files) -> None:
        class WithConfig(FakeExtractor):
            class Config:
                ollama_model = "fake-vlm:1b"

            config = Config()

        docs = await make_processor(WithConfig()).process_batch(stub_files[:1])
        assert docs[0].metadata["model"] == "fake-vlm:1b"
