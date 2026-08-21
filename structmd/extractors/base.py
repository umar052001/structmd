"""Abstract extractor interface.

Stage 1 of the pipeline: turn page images into :class:`ExtractedDocument`
JSON. Implementations only need the two sync methods; the async variants have
thread-based default implementations so any sync extractor is automatically
batch-compatible.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import List, Optional

from PIL import Image

from structmd.core import ExtractedDocument, ExtractedPage


class BaseExtractor(ABC):
    """Contract for layout extraction backends (Ollama today, others later)."""

    @abstractmethod
    def extract_page(self, image: Image.Image, page_number: int) -> ExtractedPage:
        """Extract structured elements from a single page image."""

    @abstractmethod
    def extract_document(
        self, images: List[Image.Image], source_path: Optional[str] = None
    ) -> ExtractedDocument:
        """Extract an entire document from ordered page images."""

    async def extract_page_async(self, image: Image.Image, page_number: int) -> ExtractedPage:
        """Async variant; runs the sync implementation in a worker thread."""
        return await asyncio.to_thread(self.extract_page, image, page_number)

    async def extract_document_async(
        self, images: List[Image.Image], source_path: Optional[str] = None
    ) -> ExtractedDocument:
        """Async variant; runs the sync implementation in a worker thread."""
        return await asyncio.to_thread(self.extract_document, images, source_path)
