"""Batch processing: parallel page extraction across many documents.

Documents are converted to page images up front, then all pages from all
documents flow through a single async worker pool. Results are reassembled
per document in page order.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from tqdm.asyncio import tqdm as atqdm

from structmd.converters.base import BaseConverter, detect_converter
from structmd.converters.image import ImageConverter
from structmd.converters.office import OfficeConverter
from structmd.converters.pdf import PDFConverter
from structmd.core import ExtractedDocument
from structmd.extractors.base import BaseExtractor

logger = logging.getLogger(__name__)

# (doc_id, source_path, page_number, image)
WorkItem = tuple


def collect_input_files(paths: Iterable[str]) -> Tuple[List[str], List[str]]:
    """Expand a mix of files and directories into a flat list of convertible files.

    Directories are walked recursively; anything whose extension is handled by a
    built-in converter is included. Unknown extensions inside directories are
    skipped (debug-logged), while explicitly named paths that do not exist are
    reported as missing so callers can fail fast.

    Args:
        paths: File and/or directory paths, in user-supplied order.

    Returns:
        ``(files, missing)`` — de-duplicated supported files preserving input
        order, and the paths that could not be found at all.
    """
    supported = {
        ext.lower()
        for converter in (PDFConverter, OfficeConverter, ImageConverter)
        for ext in getattr(converter, "_supported_extensions", ())
    }
    files: List[str] = []
    seen: set = set()
    missing: List[str] = []

    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_dir():
            found = [
                str(child)
                for child in sorted(path.rglob("*"))
                if child.is_file() and child.suffix.lower() in supported
            ]
            skipped = sum(1 for child in path.rglob("*") if child.is_file()) - len(found)
            if skipped:
                logger.debug("Skipped %d unsupported file(s) under %s", skipped, path)
            for item in found:
                if item not in seen:
                    seen.add(item)
                    files.append(item)
        elif path.is_file():
            item = str(path)
            if item not in seen:
                seen.add(item)
                files.append(item)
        else:
            missing.append(raw)

    return files, missing


class BatchProcessor:
    """Processes many documents through a shared async extraction pool."""

    def __init__(
        self,
        extractor: BaseExtractor,
        max_workers: int = 4,
        dpi: int = 150,
        converters: Optional[List[BaseConverter]] = None,
    ) -> None:
        self.extractor = extractor
        self.max_workers = max(1, max_workers)
        self.converters = converters or [
            PDFConverter(dpi=dpi),
            OfficeConverter(dpi=dpi),
            ImageConverter(),
        ]

    async def process_batch(
        self,
        paths: List[str],
        on_page_complete: Optional[Callable[..., Any]] = None,
        on_doc_complete: Optional[Callable[..., Any]] = None,
        pages: Optional[List[int]] = None,
    ) -> List[ExtractedDocument]:
        """Extract every page of every document concurrently.

        Args:
            paths: Document paths; unsupported/missing files are logged and
                skipped so one bad input cannot sink the batch.
            on_page_complete: ``callback(doc_id, page_number, page)`` after
                each page; may be sync or async.
            on_doc_complete: ``callback(doc_id, document)`` per document;
                may be sync or async.
            pages: Optional 1-indexed page selection applied to every
                document (e.g. ``[1, 3]``); None means all pages.

        Returns:
            Extracted documents, one per successfully converted input path,
            in the order the paths were given.
        """
        # ---- Phase 1: convert inputs to page images --------------------
        doc_ids: Dict[str, str] = {}  # path -> doc_id (input order preserved)
        jobs: Dict[str, Dict[str, Any]] = {}  # doc_id -> {source_path, pages{}}
        queue: asyncio.Queue = asyncio.Queue()
        total_pages = 0

        for path in paths:
            try:
                converter = detect_converter(path, self.converters)
                images = await asyncio.to_thread(converter.convert, path, pages)
            except Exception as exc:
                logger.error("Skipping %s: %s", path, exc)
                continue
            doc_id = uuid.uuid4().hex
            doc_ids[path] = doc_id
            jobs[doc_id] = {"source_path": path, "pages": {}}
            for page_idx, image in enumerate(images, start=1):
                queue.put_nowait((doc_id, path, page_idx, image))
                total_pages += 1

        # ---- Phase 2: worker pool ---------------------------------------
        completed = {"count": 0}

        async def worker() -> None:
            while True:
                item = await queue.get()
                try:
                    if item is None:
                        return
                    doc_id, source_path, page_number, image = item
                    try:
                        page = await self.extractor.extract_page_async(image, page_number)
                        jobs[doc_id]["pages"][page_number] = page
                        if on_page_complete is not None:
                            await _maybe_await(on_page_complete(doc_id, page_number, page))
                    except Exception as exc:
                        logger.error("Page %d of %s failed: %s", page_number, source_path, exc)
                finally:
                    completed["count"] += 1
                    queue.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(self.max_workers)]
        for _ in range(self.max_workers):
            queue.put_nowait(None)

        with atqdm(
            total=total_pages, desc="structmd", unit="page", disable=not total_pages
        ) as progress:
            while completed["count"] < total_pages:
                await asyncio.sleep(0.05)
                progress.update(completed["count"] - progress.n)
            progress.update(total_pages - progress.n)  # final flush
        await asyncio.gather(*workers)
        # ---- Phase 3: assemble documents in input order ------------------
        documents: List[ExtractedDocument] = []
        for path, doc_id in doc_ids.items():
            job = jobs[doc_id]
            extracted_pages = [job["pages"][n] for n in sorted(job["pages"])]
            document = ExtractedDocument(
                source_path=path,
                page_count=len(extracted_pages),
                pages=extracted_pages,
                metadata={
                    "extractor": type(self.extractor).__name__,
                    "model": _extractor_model(self.extractor),
                },
            )
            documents.append(document)
            if on_doc_complete is not None:
                await _maybe_await(on_doc_complete(doc_id, document))
        return documents


def _extractor_model(extractor: BaseExtractor) -> Optional[str]:
    config = getattr(extractor, "config", None)
    return getattr(config, "ollama_model", None)


async def _maybe_await(result: Any) -> None:
    """Await callbacks that return coroutines; pass through sync returns."""
    if inspect.isawaitable(result):
        await result
