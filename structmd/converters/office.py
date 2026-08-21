"""Office converter: LibreOffice headless -> PDF -> page images."""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image

from structmd.converters.base import OFFICE_EXTENSIONS, BaseConverter
from structmd.converters.pdf import PDFConverter
from structmd.core import ConversionError

logger = logging.getLogger(__name__)

SOFFICE_TIMEOUT_SECONDS = 120


class OfficeConverter(BaseConverter):
    """Converts Office documents via a headless LibreOffice PDF round-trip."""

    _supported_extensions = sorted(OFFICE_EXTENSIONS)

    def __init__(self, dpi: int = 150, soffice_binary: str = "soffice") -> None:
        self.dpi = dpi
        self.soffice_binary = soffice_binary
        self._pdf_converter = PDFConverter(dpi=dpi)

    def supports(self, path: str) -> bool:
        return self._extension(path) in OFFICE_EXTENSIONS

    def convert(
        self, path: str, pages: Optional[List[int]] = None
    ) -> List[Tuple[int, Image.Image]]:
        """Convert an Office document to page images via PDF intermediate."""
        self._require_file(path)
        source = Path(path)

        with tempfile.TemporaryDirectory(prefix="structmd-office-") as tmpdir:
            pdf_path = self._to_pdf(source, Path(tmpdir))
            logger.debug("Converted %s -> %s", path, pdf_path)
            return self._pdf_converter.convert(str(pdf_path), pages=pages)

    def _to_pdf(self, source: Path, outdir: Path) -> Path:
        """Run ``soffice --headless --convert-to pdf`` and return the PDF path."""
        cmd = [
            self.soffice_binary,
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            str(outdir),
            str(source),
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=SOFFICE_TIMEOUT_SECONDS,
                check=False,
            )
        except FileNotFoundError as exc:
            raise ConversionError(
                "LibreOffice executable not found. Install it to convert Office documents:\n"
                "  Debian/Ubuntu: sudo apt install libreoffice\n"
                "  Fedora:        sudo dnf install libreoffice\n"
                "  macOS (brew):  brew install --cask libreoffice\n"
                "  Arch:          sudo pacman -S libreoffice-still"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise ConversionError(
                f"LibreOffice conversion of {source.name!r} timed out after "
                f"{SOFFICE_TIMEOUT_SECONDS}s"
            ) from exc

        expected = outdir / (source.stem + ".pdf")
        if proc.returncode != 0 or not expected.is_file():
            stderr = (proc.stderr or proc.stdout or "").strip()
            raise ConversionError(
                f"LibreOffice failed converting {source.name!r} "
                f"(exit code {proc.returncode}): {stderr[:500]}"
            )
        return expected
