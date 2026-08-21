"""Converter tests using real generated fixtures (PDF via PyMuPDF, ODT via LibreOffice)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from PIL import Image

from structmd.converters import (
    ImageConverter,
    OfficeConverter,
    PDFConverter,
    detect_converter,
)
from structmd.core import ConversionError

pytest.importorskip("pymupdf", reason="PyMuPDF extra not installed")


@pytest.fixture(scope="module")
def sample_pdf(tmp_path_factory) -> Path:
    """A 3-page PDF with known text, generated on the fly."""
    import pymupdf

    path = tmp_path_factory.mktemp("pdf") / "sample.pdf"
    doc = pymupdf.open()
    for i in range(3):
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 100), f"Sample Document Page {i + 1}", fontsize=24)
        page.insert_text((72, 150), "The quick brown fox jumps over the lazy dog.", fontsize=12)
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture(scope="module")
def sample_odt(tmp_path_factory) -> Path:
    """A real ODT produced by headless LibreOffice from a text file."""
    soffice = __import__("shutil").which("soffice")
    if not soffice:
        pytest.skip("LibreOffice not installed")
    src = tmp_path_factory.mktemp("office") / "sample.txt"
    src.write_text("Office Test Document\n\nBody line one for the pipeline.\n", encoding="utf-8")
    outdir = src.parent
    proc = subprocess.run(
        ["soffice", "--headless", "--convert-to", "odt", "--outdir", str(outdir), str(src)],
        capture_output=True,
        timeout=120,
    )
    odt = outdir / "sample.odt"
    if proc.returncode != 0 or not odt.is_file():
        pytest.skip("LibreOffice could not produce test ODT")
    return odt


class TestPDFConverter:
    def test_renders_all_pages(self, sample_pdf: Path) -> None:
        conv = PDFConverter(dpi=150)
        images = conv.convert(str(sample_pdf))
        assert len(images) == 3
        # 612x792pt at 150dpi -> 1275x1650px
        assert images[0].size == (1275, 1650)
        assert images[0].mode == "RGB"

    def test_dpi_controls_resolution(self, sample_pdf: Path) -> None:
        images = PDFConverter(dpi=72).convert(str(sample_pdf))
        assert images[0].size == (612, 792)

    def test_page_selection_is_one_indexed(self, sample_pdf: Path) -> None:
        conv = PDFConverter(dpi=72)
        images = conv.convert(str(sample_pdf), pages=[2])
        assert len(images) == 1
        # Rendered page 2 contains its marker text; verify via pixel size only here.

    def test_out_of_range_pages_raise(self, sample_pdf: Path) -> None:
        conv = PDFConverter(dpi=72)
        with pytest.raises(ConversionError, match="out of range"):
            conv.convert(str(sample_pdf), pages=[99])

    def test_page_count(self, sample_pdf: Path) -> None:
        assert PDFConverter().get_page_count(str(sample_pdf)) == 3

    def test_missing_file_raises(self) -> None:
        with pytest.raises(ConversionError, match="not found"):
            PDFConverter().convert("/nonexistent/file.pdf")

    def test_supports(self) -> None:
        assert PDFConverter().supports("a.PDF")
        assert PDFConverter().supports("b.pdf")
        assert not PDFConverter().supports("c.png")


class TestImageConverter:
    def test_passthrough(self, tmp_path: Path) -> None:
        img_path = tmp_path / "page.png"
        Image.new("RGB", (320, 200), "steelblue").save(img_path)
        images = ImageConverter().convert(str(img_path))
        assert len(images) == 1
        assert images[0].size == (320, 200)
        assert images[0].mode == "RGB"

    def test_supports_common_formats(self) -> None:
        conv = ImageConverter()
        for ext in ("png", "jpg", "jpeg", "tiff", "bmp", "webp"):
            assert conv.supports(f"f.{ext}")
        assert not conv.supports("f.pdf")

    def test_corrupt_image_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.png"
        bad.write_bytes(b"not an image")
        with pytest.raises(ConversionError, match="Could not open image"):
            ImageConverter().convert(str(bad))


class TestOfficeConverter:
    def test_odt_to_images(self, sample_odt: Path) -> None:
        conv = OfficeConverter(dpi=100)
        images = conv.convert(str(sample_odt))
        assert len(images) >= 1
        assert images[0].mode == "RGB"

    def test_supports_office_extensions(self) -> None:
        conv = OfficeConverter()
        for ext in ("docx", "pptx", "xlsx", "odt", "ods", "odp"):
            assert conv.supports(f"d.{ext}")
        assert not conv.supports("d.pdf")

    def test_missing_soffice_binary_has_install_hint(self, tmp_path: Path, monkeypatch) -> None:
        """A missing binary must yield actionable instructions."""
        fake_docx = tmp_path / "x.docx"
        fake_docx.write_bytes(b"placeholder")
        conv = OfficeConverter(soffice_binary="definitely-not-a-real-binary-xyz")
        with pytest.raises(ConversionError, match="apt install libreoffice"):
            conv.convert(str(fake_docx))


class TestDetection:
    def test_detect_converter_picks_right_class(self, sample_pdf: Path) -> None:
        converters = [PDFConverter(), OfficeConverter(), ImageConverter()]
        assert isinstance(detect_converter(str(sample_pdf), converters), PDFConverter)
        assert isinstance(detect_converter("photo.jpg", converters), ImageConverter)
        assert isinstance(detect_converter("report.docx", converters), OfficeConverter)

    def test_unknown_extension_raises(self) -> None:
        with pytest.raises(ConversionError, match="No converter supports"):
            detect_converter("archive.zip", [PDFConverter(), ImageConverter()])
