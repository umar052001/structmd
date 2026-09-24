"""Asset extraction tests: figure crops, builder links, pipeline wiring.

All offline — PDFs are generated on the fly and no VLM is involved.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from structmd.assets import AssetExtractor, attach_assets
from structmd.builders.markdown import MarkdownBuilder
from structmd.cache.manager import CacheManager
from structmd.config import StructMDConfig
from structmd.core import (
    BoundingBox,
    DocumentElement,
    ElementType,
    ExtractedDocument,
    ExtractedPage,
)
from structmd.pipeline import StructMDPipeline


@pytest.fixture()
def figure_pdf(tmp_path: Path) -> Path:
    """A 2-page PDF whose first page has a drawn rectangle (vector figure)."""
    import pymupdf

    path = tmp_path / "report.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 100), "Report", fontsize=24)
    page.draw_rect(pymupdf.Rect(100, 200, 400, 500), color=(0, 0, 1), fill=(0.9, 0.9, 1))
    page2 = doc.new_page(width=612, height=792)
    page2.insert_text((72, 100), "Page two", fontsize=18)
    doc.save(str(path))
    doc.close()
    return path


def make_document(page_width_px: float = 612.0) -> ExtractedDocument:
    """One page, one IMAGE element with a bbox over the drawn rectangle."""
    element = DocumentElement(
        type=ElementType.IMAGE,
        bbox=BoundingBox(x1=100, y1=200, x2=400, y2=500),
        text="A blue rectangle figure",
        page_number=1,
    )
    page = ExtractedPage(page_number=1, width=page_width_px, height=792.0, elements=[element])
    return ExtractedDocument(source_path=None, page_count=1, pages=[page])


class TestAssetExtractor:
    def test_crops_image_elements_to_pngs(self, figure_pdf: Path, tmp_path: Path) -> None:
        document = make_document()
        document.source_path = str(figure_pdf)

        extractor = AssetExtractor(dpi=144)
        written = extractor.extract_assets(str(figure_pdf), document, str(tmp_path))

        assert len(written) == 1
        asset_file = tmp_path / written[0]
        assert asset_file.is_file()
        # bbox was 300x300 px at 72dpi-equivalent scale -> 600x600 at 144dpi
        with Image.open(asset_file) as img:
            assert img.size == (600, 600)
        # element annotated for the builder
        el = document.pages[0].elements[0]
        assert el.metadata["asset_path"] == written[0]
        assert el.metadata["asset_dpi"] == 144

    def test_non_pdf_source_is_skipped(self, tmp_path: Path) -> None:
        document = make_document()
        written = AssetExtractor().extract_assets("x.docx", document, str(tmp_path))
        assert written == []
        assert document.pages[0].elements[0].metadata.get("asset_path") is None

    def test_degenerate_bbox_skipped(self, figure_pdf: Path, tmp_path: Path) -> None:
        document = make_document()
        document.source_path = str(figure_pdf)
        document.pages[0].elements.append(
            DocumentElement(
                type=ElementType.IMAGE,
                bbox=BoundingBox(x1=10, y1=10, x2=14, y2=12),  # tiny -> degenerate
                text="sliver",
                page_number=1,
            )
        )
        written = AssetExtractor().extract_assets(str(figure_pdf), document, str(tmp_path))
        assert len(written) == 1  # only the real figure
        assert document.pages[0].elements[1].metadata.get("asset_path") is None

    def test_out_of_range_page_skipped(self, figure_pdf: Path, tmp_path: Path) -> None:
        document = make_document()
        document.source_path = str(figure_pdf)
        document.pages[0].page_number = 99
        written = AssetExtractor().extract_assets(str(figure_pdf), document, str(tmp_path))
        assert written == []

    def test_blank_crop_skipped(self, figure_pdf: Path, tmp_path: Path) -> None:
        """A bbox over an empty page region must not produce a blank PNG."""
        document = make_document()
        document.source_path = str(figure_pdf)
        # Page 2 has text only near the top; this box is empty white space.
        document.pages[0].elements.append(
            DocumentElement(
                type=ElementType.IMAGE,
                bbox=BoundingBox(x1=300, y1=600, x2=560, y2=760),
                text="empty region",
                page_number=1,
            )
        )
        written = AssetExtractor(dpi=144).extract_assets(str(figure_pdf), document, str(tmp_path))
        assert len(written) == 1  # only the real figure survives
        assert document.pages[0].elements[1].metadata.get("asset_path") is None

    def test_invalid_dpi_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least 72"):
            AssetExtractor(dpi=10)

    def test_attach_assets_helper(self, figure_pdf: Path, tmp_path: Path) -> None:
        md_target = tmp_path / "out" / "report.md"
        document = make_document()
        document.source_path = str(figure_pdf)
        written = attach_assets(str(figure_pdf), document, str(md_target), dpi=144)
        assert (tmp_path / "out" / written[0]).is_file()

    def test_filename_derived_from_element_id_not_position(
        self, figure_pdf: Path, tmp_path: Path
    ) -> None:
        """Crop names must not depend on element ordering in the page."""
        first = make_document()
        first.source_path = str(figure_pdf)
        written_first = AssetExtractor(dpi=144).extract_assets(
            str(figure_pdf), first, str(tmp_path / "run1")
        )

        # Same element (same id), different position in the element list.
        element = first.pages[0].elements[0]
        reordered = make_document()
        reordered.source_path = str(figure_pdf)
        reordered.pages[0].elements = [
            DocumentElement(type=ElementType.PARAGRAPH, text="preamble", page_number=1),
            element,
        ]
        written_again = AssetExtractor(dpi=144).extract_assets(
            str(figure_pdf), reordered, str(tmp_path / "run2")
        )

        assert written_first == written_again

    def test_crops_served_from_cache_without_rerendering(
        self, figure_pdf: Path, tmp_path: Path
    ) -> None:
        """A warmed run must reuse rendered crops, not re-render them."""
        cache = CacheManager(cache_dir=str(tmp_path / "cache"))
        document = make_document()
        document.source_path = str(figure_pdf)

        extractor = AssetExtractor(dpi=144, cache=cache)
        renders: list = []
        original_render = extractor._render
        extractor._render = lambda page, rect: renders.append(rect) or original_render(page, rect)

        first = extractor.extract_assets(str(figure_pdf), document, str(tmp_path / "out1"))
        assert len(renders) == 1

        renders.clear()
        second = extractor.extract_assets(str(figure_pdf), document, str(tmp_path / "out2"))
        assert len(renders) == 0, "cached run must not re-render any crop"
        assert first == second
        # both outputs exist and are byte-identical
        assert (tmp_path / "out1" / first[0]).read_bytes() == (
            tmp_path / "out2" / second[0]
        ).read_bytes()


class TestBuilderLinks:
    def test_asset_path_replaces_placeholder(self) -> None:
        element = DocumentElement(
            type=ElementType.IMAGE,
            bbox=BoundingBox(0, 0, 10, 10),
            text="Architecture diagram",
            page_number=1,
            metadata={"asset_path": "figures/report_p01_0.png"},
        )
        document = ExtractedDocument(
            source_path="report.pdf",
            page_count=1,
            pages=[ExtractedPage(page_number=1, elements=[element])],
        )
        markdown = MarkdownBuilder().build(document)
        assert "](figures/report_p01_0.png)" in markdown.content
        assert "image_placeholder" not in markdown.content

    def test_placeholder_without_assets(self) -> None:
        element = DocumentElement(
            type=ElementType.IMAGE,
            bbox=BoundingBox(0, 0, 10, 10),
            text="diagram",
            page_number=1,
        )
        document = ExtractedDocument(
            source_path="report.pdf",
            page_count=1,
            pages=[ExtractedPage(page_number=1, elements=[element])],
        )
        markdown = MarkdownBuilder().build(document)
        assert "](image_placeholder)" in markdown.content


class TestPipelineWiring:
    def test_process_writes_assets_beside_markdown(
        self, figure_pdf: Path, tmp_path: Path, monkeypatch
    ) -> None:
        class StubExtractor:
            def __init__(self, config) -> None:
                self.config = config

            def extract_page(self, image, page_number: int):
                return ExtractedPage(
                    page_number=page_number,
                    width=float(image.width),
                    height=float(image.height),
                    elements=[
                        DocumentElement(
                            type=ElementType.IMAGE,
                            bbox=BoundingBox(100, 200, 400, 500),
                            text="figure",
                            page_number=page_number,
                        )
                    ],
                )

            def close(self) -> None:
                pass

        config = StructMDConfig(
            cache_dir=str(tmp_path / "cache"),
            save_assets=True,
            assets_dirname="figs",
            assets_dpi=144,
        )
        pipe = StructMDPipeline(config=config)
        pipe.extractor = StubExtractor(config)
        monkeypatch.setattr(
            "structmd.pipeline.PDFConverter.convert",
            lambda self, path, pages=None: [
                (n, Image.new("RGB", (612, 792))) for n in (pages or [1, 2])
            ],
        )

        md_out = tmp_path / "out" / "report.md"
        try:
            markdown = pipe.process(str(figure_pdf), output_md=str(md_out))
        finally:
            pipe.close()

        assert md_out.is_file()
        figures = sorted((tmp_path / "out" / "figs").glob("*.png"))
        assert figures, "expected cropped figure PNGs beside the markdown"
        assert "](figs/" in markdown.content
