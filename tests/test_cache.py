"""Tests for the file-hash cache."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

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
        key = cache.compute_key(str(source_file))
        assert key is not None
        cache.save_asset(key, 1, "figure", 144, b"crop")
        cache.save_asset(key, 2, "other", 200, b"other crop")
        other_key = key[:2] + "0" * 62
        if other_key == key:
            other_key = key[:2] + "1" * 62
        cache.save_asset(other_key, 1, "figure", 144, b"unrelated")
        cache.invalidate(str(source_file))
        assert cache.get_cache_path(str(source_file)) is None
        assert cache.load_page(str(source_file), 1) is None
        assert cache.load_asset(key, 1, "figure", 144) is None
        assert cache.load_asset(key, 2, "other", 200) is None
        assert cache.load_asset(other_key, 1, "figure", 144) == b"unrelated"

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
        assert key is not None
        expected = cache.cache_dir / key[:2] / f"{key}_page1.json"
        assert expected.is_file()


@pytest.mark.parametrize("level", ["document", "page"])
@pytest.mark.parametrize(
    "payload",
    [
        [],
        None,
        {"elements": [{"bbox": {"x1": 0, "y1": 0, "x2": 1}}]},
        {"elements": [None]},
        {"width": "bad"},
    ],
)
def test_corrupt_payload_is_cache_miss(source_file, sample_doc, level, payload) -> None:
    cache = CacheManager(str(source_file.parent / "cache"))
    key = cache.compute_key(str(source_file))
    assert key is not None
    if level == "document":
        path = cache.save(str(source_file), sample_doc)
    else:
        cache.save_page(str(source_file), sample_doc.pages[0])
        path = cache._page_entry_path(key, 1)
    assert path is not None
    entry = json.loads(path.read_text())
    entry[level] = {"pages": [payload]} if level == "document" else payload
    path.write_text(json.dumps(entry))
    loaded = (
        cache.load(str(source_file))
        if level == "document"
        else cache.load_page(str(source_file), 1)
    )
    assert loaded is None


@pytest.mark.parametrize("raw", ["[]", "null", '"wrong"', "{broken", b"\xff"])
def test_corrupt_envelope_is_cache_miss(source_file, raw) -> None:
    cache = CacheManager(str(source_file.parent / "cache"))
    key = cache.compute_key(str(source_file))
    assert key is not None
    data = raw if isinstance(raw, bytes) else raw.encode()
    cache._entry_path(key).parent.mkdir(parents=True, exist_ok=True)
    cache._entry_path(key).write_bytes(data)
    cache._page_entry_path(key, 1).write_bytes(data)
    assert cache.load(str(source_file)) is None
    assert cache.load_page(str(source_file), 1) is None


def test_concurrent_asset_writes_publish_complete_files(source_file, monkeypatch) -> None:
    cache = CacheManager(str(source_file.parent / "cache"))
    key = cache.compute_key(str(source_file))
    assert key is not None
    initial = b"initial crop"
    cache.save_asset(key, 1, "figure", 144, initial)
    payloads = [bytes([n]) * 100000 for n in range(8)]
    barrier = Barrier(len(payloads))
    replaced = []
    replace = os.replace

    def synchronized_replace(src, dst):
        src, dst = Path(src), Path(dst)
        assert src.parent == dst.parent
        assert src.read_bytes() in payloads
        assert dst.read_bytes() == initial
        barrier.wait(timeout=10)
        replace(src, dst)
        replaced.append(src)

    monkeypatch.setattr("structmd.cache.manager.os.replace", synchronized_replace)
    with ThreadPoolExecutor(max_workers=len(payloads)) as pool:
        list(pool.map(lambda data: cache.save_asset(key, 1, "figure", 144, data), payloads))
    assert len(set(replaced)) == len(payloads)
    assert cache.load_asset(key, 1, "figure", 144) in payloads
    assert not list(cache.cache_dir.rglob("*.tmp"))


def test_asset_ids_are_collision_resistant_and_bounded(source_file) -> None:
    cache = CacheManager(str(source_file.parent / "cache"))
    key = cache.compute_key(str(source_file))
    assert key is not None
    ids = ["a/b", "a?b", "a_b", "", "x" * 10000]
    for id_ in ids:
        cache.save_asset(key, 1, id_, 144, id_.encode() or b"empty ID")
    assert len(list(cache.cache_dir.rglob("*.png"))) == len(ids)
    for id_ in ids:
        assert cache.load_asset(key, 1, id_, 144) == (id_.encode() or b"empty ID")


@pytest.mark.parametrize("failure", ["mkdir", "create", "write", "replace"])
def test_cache_write_errors_are_nonfatal_and_cleaned_up(
    source_file, sample_doc, monkeypatch, failure
) -> None:
    import tempfile

    cache = CacheManager(str(source_file.parent / "cache"))
    key = cache.compute_key(str(source_file))
    assert key is not None

    def fail(*args, **kwargs):
        raise OSError("cache unavailable")

    if failure == "mkdir":
        monkeypatch.setattr(Path, "mkdir", fail)
    elif failure == "create":
        monkeypatch.setattr("structmd.cache.manager.tempfile.NamedTemporaryFile", fail)
    elif failure == "write":
        create = tempfile.NamedTemporaryFile

        def failing_writer(*args, **kwargs):
            tmp = create(*args, **kwargs)
            monkeypatch.setattr(tmp, "write", fail)
            return tmp

        monkeypatch.setattr("structmd.cache.manager.tempfile.NamedTemporaryFile", failing_writer)
    else:
        monkeypatch.setattr("structmd.cache.manager.os.replace", fail)
    assert cache.save(str(source_file), sample_doc) is None
    cache.save_page(str(source_file), sample_doc.pages[0])
    cache.save_asset(key, 1, "figure", 144, b"crop")
    assert not list(cache.cache_dir.rglob("*.tmp"))
    assert not list(cache.cache_dir.rglob("*.png"))


def test_cache_read_errors_are_misses(source_file, monkeypatch) -> None:
    cache = CacheManager(str(source_file.parent / "cache"))
    key = cache.compute_key(str(source_file))
    assert key is not None

    def fail(*args, **kwargs):
        raise PermissionError("cache unavailable")

    monkeypatch.setattr(Path, "read_text", fail)
    monkeypatch.setattr(Path, "read_bytes", fail)
    assert cache.load(str(source_file)) is None
    assert cache.load_page(str(source_file), 1) is None
    assert cache.load_asset(key, 1, "figure", 144) is None


def test_cache_directory_failure_is_nonfatal(tmp_path, monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise PermissionError("cache unavailable")

    monkeypatch.setattr(Path, "mkdir", fail)
    cache = CacheManager(str(tmp_path / "cache"))
    assert cache.load("missing.pdf") is None
