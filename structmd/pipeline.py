"""Main orchestrator: convert -> extract -> (cache) -> build."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

from structmd.assets import attach_assets
from structmd.batch.processor import BatchProcessor
from structmd.builders.markdown import BuilderConfig, MarkdownBuilder
from structmd.cache.manager import CacheManager
from structmd.config import StructMDConfig, load_config
from structmd.converters.base import detect_converter
from structmd.converters.image import ImageConverter
from structmd.converters.office import OfficeConverter
from structmd.converters.pdf import PDFConverter
from structmd.core import ExtractedDocument, ExtractedPage, MarkdownDocument
from structmd.extractors.ollama import OllamaExtractor

logger = logging.getLogger(__name__)


class StructMDPipeline:
    """End-to-end facade over extraction, caching, and Markdown building."""

    def __init__(self, config: Optional[StructMDConfig] = None) -> None:
        self.config = config or load_config()
        if self.config.verbose:
            logging.getLogger("structmd").setLevel(logging.DEBUG)
        self.extractor = OllamaExtractor(self.config)
        self.builder = MarkdownBuilder(self._builder_config())
        self.cache = CacheManager(self.config.cache_dir)
        self.converters = [
            PDFConverter(dpi=self.config.dpi),
            OfficeConverter(dpi=self.config.dpi),
            ImageConverter(),
        ]

    def _builder_config(self) -> BuilderConfig:
        cfg = self.config
        return BuilderConfig(
            include_page_numbers=cfg.include_page_numbers,
            page_number_format=cfg.page_number_format,
            merge_continued_paragraphs=cfg.merge_continued_paragraphs,
            detect_columns=cfg.detect_columns,
            normalize_headings=cfg.normalize_headings,
            table_caption_position=cfg.table_caption_position,
        )

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def process(
        self,
        input_path: str,
        output_json: Optional[str] = None,
        output_md: Optional[str] = None,
        pages: Optional[List[int]] = None,
        force: bool = False,
    ) -> MarkdownDocument:
        """Convert one document to Markdown end-to-end.

        Args:
            input_path: PDF / Office / image file.
            output_json: Where to persist the Stage 1 extraction JSON.
            output_md: Where to write the final Markdown.
            pages: 1-indexed page subset (document cache bypassed when given;
                page-level cache still applies).
            force: Skip cache reads; re-extract and refresh cached entries.
        """
        extracted: Optional[ExtractedDocument] = None

        # 1) Fresh cache hit short-circuits conversion + extraction.
        if pages is None and self.config.cache_enabled and not force:
            extracted = self.cache.load(input_path)
            if extracted is not None:
                logger.info("Cache hit for %s", input_path)

        # 2-4) Convert and extract (page-level cache applies inside).
        if extracted is None:
            extracted = self.extract_only(input_path, pages=pages, force=force)
            # 5) Persist to cache (full-document runs only).
            if pages is None and self.config.cache_enabled:
                self.cache.save(input_path, extracted)

        # 6) Optional intermediate JSON.
        if output_json:
            written = extracted.save_json(output_json)
            logger.info("Extraction JSON written to %s", written)

        # 7) Optional figure assets (annotates elements before building).
        if output_md:
            self._extract_assets_for(extracted, output_md)

        # 8) Deterministic build.
        markdown = self.builder.build(extracted)

        # 9) Optional Markdown output.
        if output_md:
            written = markdown.save(output_md)
            logger.info("Markdown written to %s", written)

        return markdown

    # ------------------------------------------------------------------
    # Figure assets
    # ------------------------------------------------------------------

    def _extract_assets_for(self, document: ExtractedDocument, md_target: str) -> None:
        """Crop figure regions beside ``md_target`` when assets are enabled.

        No-op unless ``config.save_assets`` is set or the source is missing.
        """
        if not self.config.save_assets or not document.source_path:
            return
        try:
            attach_assets(
                document.source_path,
                document,
                md_target,
                dpi=self.config.assets_dpi,
                dirname=self.config.assets_dirname,
                cache=self.cache if self.config.cache_enabled else None,
            )
        except Exception as exc:  # noqa: BLE001 - assets must never kill a run
            logger.warning("Asset extraction failed for %s: %s", document.source_path, exc)

    # ------------------------------------------------------------------
    # Stage 1 only
    # ------------------------------------------------------------------

    def extract_only(
        self,
        input_path: str,
        output_json: Optional[str] = None,
        pages: Optional[List[int]] = None,
        force: bool = False,
    ) -> ExtractedDocument:
        """Run conversion + VLM extraction only; returns the JSON document.

        Page-level results are cached individually: a re-run only pays for
        pages that changed or were never extracted. Pass ``force=True`` to
        bypass cache reads (fresh writes are still recorded).
        """
        converter = detect_converter(input_path, self.converters)
        pairs = converter.convert(input_path, pages=pages)
        if not pairs:
            raise ValueError(f"No pages rendered from {input_path!r}")

        use_cache = self.config.cache_enabled and not force
        by_number: Dict[int, ExtractedPage] = {}
        misses: List[Tuple[int, Image.Image]] = []
        if use_cache:
            for number, image in pairs:
                cached = self.cache.load_page(input_path, number)
                if cached is not None:
                    by_number[number] = cached
                else:
                    misses.append((number, image))
        else:
            misses = list(pairs)

        if misses:
            suffix = f" ({len(by_number)} from cache)" if by_number else ""
            logger.info(
                "Extracting %d page(s) from %s with %s%s",
                len(misses),
                input_path,
                self.config.ollama_model,
                suffix,
            )
            for number, image in misses:
                page = self.extractor.extract_page(image, number)
                by_number[number] = page
                if self.config.cache_enabled:
                    self.cache.save_page(input_path, page)

        extracted = ExtractedDocument(
            source_path=input_path,
            page_count=len(by_number),
            pages=[by_number[n] for n in sorted(by_number)],
            metadata={"extractor": "ollama", "model": self.config.ollama_model},
        )
        extracted.metadata["dpi"] = self.config.dpi

        if output_json:
            written = extracted.save_json(output_json)
            logger.info("Extraction JSON written to %s", written)
        return extracted

    # ------------------------------------------------------------------
    # Stage 2 only
    # ------------------------------------------------------------------

    def build_from_json(self, json_path: str, output_md: Optional[str] = None) -> MarkdownDocument:
        """Build Markdown from an existing extraction JSON (no VLM needed)."""
        raw = Path(json_path).expanduser().read_text(encoding="utf-8")
        try:
            extracted = ExtractedDocument.from_json(raw)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{json_path!r} is not a valid structmd extraction file: {exc}"
            ) from exc

        markdown = self.builder.build(extracted)
        if output_md:
            written = markdown.save(output_md)
            logger.info("Markdown written to %s", written)
        return markdown

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------

    async def process_batch_async(
        self,
        paths: List[str],
        pages: Optional[List[int]] = None,
        force: bool = False,
        output_dir: Optional[str] = None,
    ) -> List[MarkdownDocument]:
        """Async batch variant of :meth:`process_batch`."""
        processor = BatchProcessor(
            self.extractor,
            max_workers=self.config.ollama_max_workers,
            dpi=self.config.dpi,
            converters=self.converters,
            cache=self.cache if self.config.cache_enabled else None,
            force=force,
        )
        documents = await processor.process_batch(paths, pages=pages)
        results = []
        for document in documents:
            if document.source_path and self.config.cache_enabled:
                self.cache.save(document.source_path, document)
            if output_dir and document.source_path:
                stem = Path(document.source_path).stem
                self._extract_assets_for(document, str(Path(output_dir) / f"{stem}.md"))
            results.append(self.builder.build(document))
        return results

    def process_batch(
        self,
        paths: List[str],
        pages: Optional[List[int]] = None,
        force: bool = False,
        output_dir: Optional[str] = None,
    ) -> List[MarkdownDocument]:
        """Convert many documents; pages flow through a shared worker pool.

        Args:
            paths: Document paths.
            pages: Optional 1-indexed page selection applied to every document.
            force: Skip cache reads; re-extract and refresh cached entries.
            output_dir: When set together with ``config.save_assets``, figure
                PNGs are written to ``<output_dir>/<assets_dirname>/``.
        """
        import asyncio

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.process_batch_async(paths, pages, force, output_dir))
        else:
            # Already inside an event loop (e.g. Jupyter): run in a helper
            # thread with its own loop so we can block for the result.
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(
                    asyncio.run, self._batch_coro(paths, pages, force, output_dir)
                ).result()

    async def _batch_coro(
        self,
        paths: List[str],
        pages: Optional[List[int]] = None,
        force: bool = False,
        output_dir: Optional[str] = None,
    ) -> List[MarkdownDocument]:
        return await self.process_batch_async(paths, pages, force, output_dir)

    def close(self) -> None:
        """Release HTTP resources held by the extractor."""
        self.extractor.close()

    def __enter__(self) -> StructMDPipeline:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
