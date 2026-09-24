"""Tests for the file-hash cache."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from structmd.cache.manager import CacheManager
from structmd.core import ExtractedDocument, ExtractedPage


@pytest.fixture()
def source_file(tmp_path: Path) -> Path:
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"%PDF-1.4 fake content")
    return f


@pytest.fixture()
def sample_doc() -> ExtractedDocument:
    return ExtractedDocument(
        document_id="cached-doc",
        page_count=1,
        pages=[ExtractedPage(page_number=1, width=100, height=200)],
        metadata={"model": "test"},
    )


class TestKeying:
    def test_key_changes_when_file_modified(self, source_file: Path) -> None:
        cache = CacheManager(cache_dir=str(source_file.parent / "cache"))
        key1 = cache.compute_key(str(source_file))
        os.utime(source_file, (0, 0))  # change mtime
        key2 = cache.compute_key(str(source_file))
        assert key1 != key2

    def test_key_stable_for_untouched_file(self, source_file: Path) -> None:
        cache = CacheManager(cache_dir=str(source_file.parent / "cache"))
        assert cache.compute_key(str(source_file)) == cache.compute_key(str(source_file))

    def test_missing_file_returns_none(self, tmp_path) -> None:
        cache = CacheManager(cache_dir=str(tmp_path / "cache"))
        assert cache.compute_key("/no/such/file.pdf") is None


class TestDocumentCache:
    def test_save_and_load_roundtrip(
        self, source_file: Path, sample_doc: ExtractedDocument
    ) -> None:
        cache = CacheManager(cache_dir=str(source_file.parent / "cache"))
        entry = cache.save(str(source_file), sample_doc)
        assert entry is not None and entry.is_file()

        loaded = cache.load(str(source_file))
        assert loaded is not None
        assert loaded.document_id == "cached-doc"
        assert loaded.pages[0].height == 200

    def test_get_cache_path_fresh(self, source_file: Path, sample_doc: ExtractedDocument) -> None:
        cache = CacheManager(cache_dir=str(source_file.parent / "cache"))
        cache.save(str(source_file), sample_doc)
        path = cache.get_cache_path(str(source_file))
        assert path is not None and path.is_file()
        # Sharded layout: {dir}/{key[:2]}/{key}.json
        assert path.parent.name == path.stem[:2]

    def test_stale_after_modification(
        self, source_file: Path, sample_doc: ExtractedDocument
    ) -> None:
        cache = CacheManager(cache_dir=str(source_file.parent / "cache"))
        cache.save(str(source_file), sample_doc)
        assert cache.get_cache_path(str(source_file)) is not None

        source_file.write_bytes(b"modified content!")
        assert cache.get_cache_path(str(source_file)) is None
        assert cache.load(str(source_file)) is None

    def test_invalidate_removes_entries(
        self, source_file: Path, sample_doc: ExtractedDocument
    ) -> None:
        cache = CacheManager(cache_dir=str(source_file.parent / "cache"))
        cache.save(str(source_file), sample_doc)
        cache.save_page(str(source_file), sample_doc.pages[0])
        cache.invalidate(str(source_file))
        assert cache.get_cache_path(str(source_file)) is None
        assert cache.load_page(str(source_file), 1) is None

    def test_corrupt_entry_treated_as_miss(
        self, source_file: Path, sample_doc: ExtractedDocument
    ) -> None:
        cache = CacheManager(cache_dir=str(source_file.parent / "cache"))
        entry = cache.save(str(source_file), sample_doc)
        assert entry is not None
        entry.write_text("{corrupt json", encoding="utf-8")
        assert cache.get_cache_path(str(source_file)) is None


class TestPageCache:
    def test_page_roundtrip(self, source_file: Path, sample_doc: ExtractedDocument) -> None:
        cache = CacheManager(cache_dir=str(source_file.parent / "cache"))
        cache.save_page(str(source_file), sample_doc.pages[0])
        page = cache.load_page(str(source_file), 1)
        assert page is not None
        assert page.page_number == 1

    def test_page_filename_convention(
        self, source_file: Path, sample_doc: ExtractedDocument
    ) -> None:
        """Spec convention: {hash}_page{N}.json."""
        cache = CacheManager(cache_dir=str(source_file.parent / "cache"))
        cache.save_page(str(source_file), sample_doc.pages[0])
        key = cache.compute_key(str(source_file))
        expected = cache.cache_dir / key[:2] / f"{key}_page1.json"
        assert expected.is_file()
