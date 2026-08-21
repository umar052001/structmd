"""File-hash based JSON cache for extraction results.

Cache key = ``sha256(abs_path + mtime + size)`` so moving a file invalidates
nothing (path-independent), while any modification does.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from structmd.core import ExtractedDocument, ExtractedPage

logger = logging.getLogger(__name__)


class CacheManager:
    """Stores extraction JSON keyed by file content identity."""

    def __init__(self, cache_dir: str = "~/.cache/structmd") -> None:
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Key computation
    # ------------------------------------------------------------------

    def compute_key(self, file_path: str) -> Optional[str]:
        """sha256 of absolute path + mtime + size; None if file missing."""
        path = Path(file_path).expanduser().resolve()
        try:
            stat = path.stat()
        except OSError:
            return None
        identity = f"{path}:{stat.st_mtime_ns}:{stat.st_size}"
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def _shard_dir(self, key: str) -> Path:
        return self.cache_dir / key[:2]

    def _entry_path(self, key: str) -> Path:
        return self._shard_dir(key) / f"{key}.json"

    def _page_entry_path(self, key: str, page_number: int) -> Path:
        return self._shard_dir(key) / f"{key}_page{page_number}.json"

    @staticmethod
    def _is_fresh(entry: Dict[str, Any], file_path: str) -> bool:
        """Compare stored stat snapshot against the current file."""
        try:
            stat = Path(file_path).expanduser().resolve().stat()
        except OSError:
            return False
        return entry.get("mtime") == stat.st_mtime_ns and entry.get("size") == stat.st_size

    # ------------------------------------------------------------------
    # Document-level API
    # ------------------------------------------------------------------

    def get_cache_path(self, file_path: str) -> Optional[Path]:
        """Return the cached JSON path if it exists and is fresh, else None."""
        key = self.compute_key(file_path)
        if key is None:
            return None
        entry_path = self._entry_path(key)
        if not entry_path.is_file():
            return None
        try:
            entry = json.loads(entry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Discarding unreadable cache entry %s: %s", entry_path, exc)
            return None
        if not self._is_fresh(entry, file_path):
            logger.debug("Cache stale for %s", file_path)
            return None
        return entry_path

    def load(self, file_path: str) -> Optional[ExtractedDocument]:
        """Convenience wrapper: fresh cached document or None."""
        entry_path = self.get_cache_path(file_path)
        if entry_path is None:
            return None
        try:
            entry = json.loads(entry_path.read_text(encoding="utf-8"))
            return ExtractedDocument.from_dict(entry["document"])
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            logger.warning("Could not load cache entry %s: %s", entry_path, exc)
            return None

    def save(self, file_path: str, extracted: ExtractedDocument) -> Optional[Path]:
        """Persist an extraction result; returns the entry path."""
        key = self.compute_key(file_path)
        if key is None:
            logger.debug("Cannot cache %s: source missing", file_path)
            return None
        try:
            stat = Path(file_path).expanduser().resolve().stat()
        except OSError:
            return None
        entry = {
            "file_path": str(Path(file_path).resolve()),
            "mtime": stat.st_mtime_ns,
            "size": stat.st_size,
            "document": extracted.to_dict(),
        }
        entry_path = self._entry_path(key)
        entry_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = entry_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_path, entry_path)  # atomic swap
        logger.debug("Cached extraction of %s at %s", file_path, entry_path)
        return entry_path

    def invalidate(self, file_path: str) -> None:
        """Remove document and page entries for ``file_path``."""
        key = self.compute_key(file_path)
        if key is None:
            return
        shard = self._shard_dir(key)
        removed = 0
        if shard.is_dir():
            for candidate in shard.glob(f"{key}*.json"):
                try:
                    candidate.unlink()
                    removed += 1
                except OSError as exc:
                    logger.warning("Could not remove cache file %s: %s", candidate, exc)
        if removed:
            logger.debug("Invalidated %d cache file(s) for %s", removed, file_path)

    # ------------------------------------------------------------------
    # Page-level API (partial reuse across runs)
    # ------------------------------------------------------------------

    def save_page(self, file_path: str, page: ExtractedPage) -> None:
        """Cache a single page's extraction."""
        key = self.compute_key(file_path)
        if key is None:
            return
        try:
            stat = Path(file_path).expanduser().resolve().stat()
        except OSError:
            return
        entry = {
            "mtime": stat.st_mtime_ns,
            "size": stat.st_size,
            "page": page.to_dict(),
        }
        entry_path = self._page_entry_path(key, page.page_number)
        entry_path.parent.mkdir(parents=True, exist_ok=True)
        entry_path.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")

    def load_page(self, file_path: str, page_number: int) -> Optional[ExtractedPage]:
        """Return a fresh cached page or None."""
        key = self.compute_key(file_path)
        if key is None:
            return None
        entry_path = self._page_entry_path(key, page_number)
        if not entry_path.is_file():
            return None
        try:
            entry = json.loads(entry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not self._is_fresh(entry, file_path):
            return None
        return ExtractedPage.from_dict(entry["page"])
