"""File-hash based JSON cache for extraction results.

Cache key = ``sha256(abs_path + mtime + size)``. Any modification to the
source file therefore invalidates its entries; an untouched file keeps its
entries until something changes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from structmd.core import ExtractedDocument, ExtractedPage

logger = logging.getLogger(__name__)


class CacheManager:
    """Stores extraction JSON keyed by file content identity."""

    def __init__(self, cache_dir: str = "~/.cache/structmd") -> None:
        self.cache_dir = Path(cache_dir).expanduser()
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("Could not create cache directory %s: %s", self.cache_dir, exc)

    # ------------------------------------------------------------------
    # Key computation
    # ------------------------------------------------------------------

    def compute_key(self, file_path: str) -> Optional[str]:
        """sha256 of absolute path + mtime + size; None if file missing."""
        try:
            path = Path(file_path).expanduser().resolve()
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
    def _write(path: Path, data: bytes) -> bool:
        """Atomically publish a complete cache entry; cache I/O is best-effort."""
        tmp_path = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as tmp:
                tmp_path = Path(tmp.name)
                tmp.write(data)
            os.replace(tmp_path, path)
            return True
        except OSError as exc:
            logger.warning("Could not write cache entry %s: %s", path, exc)
            return False
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError as exc:
                    logger.warning("Could not remove cache tempfile %s: %s", tmp_path, exc)

    @staticmethod
    def _is_fresh(entry: Dict[str, Any], file_path: str) -> bool:
        """Compare stored stat snapshot against the current file."""
        try:
            stat = Path(file_path).expanduser().resolve().stat()
        except OSError:
            return False
        return (
            isinstance(entry, dict)
            and entry.get("mtime") == stat.st_mtime_ns
            and entry.get("size") == stat.st_size
        )

    # ------------------------------------------------------------------
    # Document-level API
    # ------------------------------------------------------------------

    def get_cache_path(self, file_path: str) -> Optional[Path]:
        """Return the cached JSON path if it exists and is fresh, else None."""
        key = self.compute_key(file_path)
        if key is None:
            return None
        entry_path = self._entry_path(key)
        try:
            entry = json.loads(entry_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
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
        except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError) as exc:
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
        if not self._write(entry_path, json.dumps(entry, ensure_ascii=False).encode("utf-8")):
            return None
        logger.debug("Cached extraction of %s at %s", file_path, entry_path)
        return entry_path

    def invalidate(self, file_path: str) -> None:
        """Remove document, page and asset entries for ``file_path``."""
        key = self.compute_key(file_path)
        if key is None:
            return
        shard = self._shard_dir(key)
        removed = 0
        try:
            for candidate in shard.glob(f"{key}*"):
                if candidate.suffix not in {".json", ".png"}:
                    continue
                try:
                    candidate.unlink()
                    removed += 1
                except OSError as exc:
                    logger.warning("Could not remove cache file %s: %s", candidate, exc)
        except OSError as exc:
            logger.warning("Could not list cache directory %s: %s", shard, exc)
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
        self._write(entry_path, json.dumps(entry, ensure_ascii=False).encode("utf-8"))

    def load_page(self, file_path: str, page_number: int) -> Optional[ExtractedPage]:
        """Return a fresh cached page or None."""
        key = self.compute_key(file_path)
        if key is None:
            return None
        entry_path = self._page_entry_path(key, page_number)
        try:
            entry = json.loads(entry_path.read_text(encoding="utf-8"))
            if not self._is_fresh(entry, file_path):
                return None
            return ExtractedPage.from_dict(entry["page"])
        except FileNotFoundError:
            return None
        except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError) as exc:
            logger.warning("Could not load page cache entry %s: %s", entry_path, exc)
            return None

    # ------------------------------------------------------------------
    # Asset crops (binary, keyed by document identity + element)
    # ------------------------------------------------------------------
    #
    # Rendered figure crops live under the same document key as their
    # extraction JSON, so a source edit invalidates crops with the pages.
    # AssetExtractor includes the effective rectangle in the crop identity.

    def _asset_path(self, key: str, page_number: int, element_id: str, dpi: int) -> Path:
        safe_id = hashlib.sha256(element_id.encode("utf-8")).hexdigest()
        return self._shard_dir(key) / f"{key}_p{page_number}_{safe_id}_{dpi}px.png"

    def load_asset(self, key: str, page_number: int, element_id: str, dpi: int) -> Optional[bytes]:
        """Return cached crop bytes for one image element, or None."""
        asset_path = self._asset_path(key, page_number, element_id, dpi)
        try:
            return asset_path.read_bytes() or None
        except FileNotFoundError:
            return None
        except OSError as exc:
            logger.warning("Could not read asset cache %s: %s", asset_path, exc)
            return None

    def save_asset(
        self, key: str, page_number: int, element_id: str, dpi: int, data: bytes
    ) -> None:
        """Persist rendered crop bytes (atomic swap)."""
        asset_path = self._asset_path(key, page_number, element_id, dpi)
        if self._write(asset_path, data):
            logger.debug("Cached asset crop at %s", asset_path)
