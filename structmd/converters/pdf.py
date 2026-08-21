"""PDF converter: PyMuPDF -> PIL page images."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from PIL import Image

from structmd.converters.base import PDF_EXTENSIONS, BaseConverter
from structmd.core import ConversionError

logger = logging.getLogger(__name__)

try:  # PyMuPDF >= 1.24 exposes the modern name.
    import pymupdf as fitz_module
except ImportError:  # pragma: no cover - legacy import path
    import fitz as fitz_module  # type: ignore[no-redef, import-untyped]


class PDFConverter(BaseConverter):
    """Renders PDF pages to RGB images at a configurable DPI."""

    _supported_extensions = sorted(PDF_EXTENSIONS)

    def __init__(self, dpi: int = 150) -> None:
        self.dpi = dpi

    def supports(self, path: str) -> bool:
        return self._extension(path) in PDF_EXTENSIONS

    def get_page_count(self, path: str) -> int:
        """Return the number of pages in the PDF."""
        self._require_file(path)
        try:
            with fitz_module.open(path) as doc:
                return int(doc.page_count)
        except Exception as exc:
            raise ConversionError(f"Could not open PDF {path!r}: {exc}") from exc

    def convert(
        self, path: str, pages: Optional[List[int]] = None
    ) -> List[Tuple[int, Image.Image]]:
        """Render pages (1-indexed; None = all) to PIL RGB images."""
        self._require_file(path)
        try:
            doc = fitz_module.open(path)
        except Exception as exc:
            raise ConversionError(f"Could not open PDF {path!r}: {exc}") from exc

        images: List[Tuple[int, Image.Image]] = []
        try:
            total = doc.page_count
            if pages is not None:
                invalid = [p for p in pages if p < 1 or p > total]
                if invalid:
                    raise ConversionError(
                        f"Page(s) {invalid} out of range for {path!r} ({total} pages)"
                    )
                indices = sorted({p - 1 for p in pages})
            else:
                indices = list(range(total))

            matrix = fitz_module.Matrix(self.dpi / 72.0, self.dpi / 72.0)
            for idx in indices:
                page = doc.load_page(idx)
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                images.append((idx + 1, image))
                pix = None  # allow eager frees of large pixmaps
        except ConversionError:
            raise
        except Exception as exc:
            raise ConversionError(f"Failed rendering {path!r}: {exc}") from exc
        finally:
            doc.close()

        logger.debug("Rendered %d page(s) from %s at %d DPI", len(images), path, self.dpi)
        return images
