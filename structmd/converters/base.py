"""Abstract converter interface: input file -> page images."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from PIL import Image

from structmd.core import ConversionError

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
PDF_EXTENSIONS = {".pdf"}
OFFICE_EXTENSIONS = {
    ".docx",
    ".pptx",
    ".xlsx",
    ".odt",
    ".ods",
    ".odp",
    ".doc",
    ".ppt",
    ".xls",
}


class BaseConverter(ABC):
    """Contract for turning a source document into PIL page images."""

    @abstractmethod
    def convert(self, path: str, pages: Optional[List[int]] = None) -> List[Image.Image]:
        """Convert ``path`` to page images.

        Args:
            path: Source document path.
            pages: 1-indexed page numbers to render; None means all pages.

        Returns:
            Ordered list of RGB PIL images.
        """

    @abstractmethod
    def supports(self, path: str) -> bool:
        """Return True when this converter handles the given file type."""

    @staticmethod
    def _extension(path: str) -> str:
        return Path(path).suffix.lower()

    @staticmethod
    def _require_file(path: str) -> None:
        if not os.path.isfile(path):
            raise ConversionError(f"Input file not found: {path}")


def detect_converter(path: str, converters: List[BaseConverter]) -> BaseConverter:
    """Pick the first converter that supports ``path``."""
    for converter in converters:
        if converter.supports(path):
            return converter
    supported = sorted(
        ext for conv in converters for ext in getattr(conv, "_supported_extensions", ())
    )
    raise ConversionError(
        f"No converter supports {path!r}. Supported types: {', '.join(supported) or '(unknown)'}"
    )
