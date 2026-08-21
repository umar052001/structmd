"""Figure asset extraction: crop ``image`` regions out of source PDFs.

Sits between Stage 1 (VLM extraction, which yields per-element bounding
boxes) and Stage 2 (Markdown build): every element of type ``image`` with a
usable bbox is clip-rendered from the source PDF at high DPI and saved as a
PNG next to the Markdown output. The builder then links the real file
instead of an ``image_placeholder``.

Clip-rendering (rather than extracting embedded raster objects) is
deliberate: figures in academic PDFs are usually *vector* graphics plus text
labels, which only exist as pixels once rendered.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional

from structmd.converters.pdf import fitz_module
from structmd.core import BoundingBox, ElementType, ExtractedDocument

logger = logging.getLogger(__name__)

MIN_SIDE_PT = 12.0  # ignore degenerate boxes smaller than this (in points)


class AssetExtractor:
    """Renders ``image`` elements' bounding boxes from a source PDF to PNGs."""

    def __init__(self, dpi: int = 200) -> None:
        if dpi < 72:
            raise ValueError("Asset DPI must be at least 72")
        self.dpi = dpi

    def extract_assets(
        self,
        source_path: str,
        document: ExtractedDocument,
        output_dir: str,
        dirname: str = "figures",
    ) -> List[str]:
        """Crop every IMAGE element with a usable bbox; returns written paths.

        Elements are annotated in place with ``metadata["asset_path"]`` (a
        relative POSIX path suitable for Markdown links) and
        ``metadata["asset_dpi"]``. Only PDF sources are supported; anything
        else is skipped with a debug log so callers can stay unconditional.
        """
        src = Path(source_path)
        if src.suffix.lower() not in {".pdf"}:
            logger.debug("Asset extraction supports PDF sources only; skipping %s", src.name)
            return []

        assets_root = Path(output_dir) / dirname
        assets_root.mkdir(parents=True, exist_ok=True)

        written: List[str] = []
        with fitz_module.open(str(src)) as pdf:
            for page in document.pages:
                if not 1 <= page.page_number <= pdf.page_count:
                    logger.warning(
                        "Page %s out of range for %s; skipping its assets",
                        page.page_number,
                        src.name,
                    )
                    continue
                pdf_page = pdf[page.page_number - 1]
                scale = self._points_per_pixel(pdf_page.rect, page.width)
                for idx, element in enumerate(page.elements):
                    if element.type is not ElementType.IMAGE or element.bbox is None:
                        continue
                    rect = self._to_pdf_rect(element.bbox, scale, pdf_page.rect)
                    if rect is None:
                        logger.debug("Skipping degenerate image bbox on page %s", page.page_number)
                        continue
                    filename = f"{src.stem}_p{page.page_number:02d}_{idx}.png"
                    pixmap = pdf_page.get_pixmap(clip=rect, dpi=self.dpi)
                    if self._is_blank(pixmap):
                        logger.debug(
                            "Skipping blank crop (VLM bbox on empty region) page %s",
                            page.page_number,
                        )
                        continue
                    pixmap.save(str(assets_root / filename))
                    relative = f"{dirname}/{filename}"
                    element.metadata["asset_path"] = relative
                    element.metadata["asset_dpi"] = self.dpi
                    written.append(relative)

        if written:
            logger.info("Extracted %d figure asset(s) from %s", len(written), src.name)
        return written

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _points_per_pixel(page_rect: Any, page_width_px: float) -> float:
        """PDF points per extracted pixel, from real rendered dimensions."""
        if page_width_px and page_width_px > 0:
            return float(page_rect.width) / float(page_width_px)
        return 1.0

    @staticmethod
    def _to_pdf_rect(bbox: BoundingBox, scale: float, page_rect: Any) -> Optional[Any]:
        """Convert a pixel bbox to a clamped PDF-space rect; None if degenerate."""
        x1 = min(bbox.x1, bbox.x2) * scale
        y1 = min(bbox.y1, bbox.y2) * scale
        x2 = max(bbox.x1, bbox.x2) * scale
        y2 = max(bbox.y1, bbox.y2) * scale
        rect = fitz_module.Rect(x1, y1, x2, y2)
        rect &= page_rect  # clamp into the page (VLM boxes can overshoot)
        if rect.is_empty or rect.width < MIN_SIDE_PT or rect.height < MIN_SIDE_PT:
            return None
        return rect

    @staticmethod
    def _is_blank(pixmap: Any, subsample_step: int = 97) -> bool:
        """True when the rendered crop is essentially one flat color.

        VLM bboxes occasionally land on empty page regions; such crops would
        become broken-looking figure links. We subsample the raw pixel bytes
        and require some tonal variety before accepting the crop.
        """
        samples = pixmap.samples
        if not samples:
            return True
        distinct = {samples[i] for i in range(0, len(samples), subsample_step)}
        return len(distinct) < 3


def attach_assets(
    source_path: str,
    document: ExtractedDocument,
    markdown_output_path: str,
    dpi: int = 200,
    dirname: str = "figures",
) -> List[str]:
    """Convenience wrapper: extract assets beside a Markdown output file.

    Assets land in ``<markdown_dir>/<dirname>/`` and element metadata is
    updated so a subsequent :meth:`MarkdownBuilder.build` emits real links.
    """
    output_dir = Path(markdown_output_path).expanduser().parent
    return AssetExtractor(dpi=dpi).extract_assets(
        source_path, document, str(output_dir), dirname=dirname
    )


__all__ = ["AssetExtractor", "attach_assets"]
