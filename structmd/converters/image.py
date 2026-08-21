"""Image converter: direct passthrough for raster formats."""

from __future__ import annotations

import logging
from typing import List, Optional

from PIL import Image

from structmd.converters.base import IMAGE_EXTENSIONS, BaseConverter
from structmd.core import ConversionError

logger = logging.getLogger(__name__)


class ImageConverter(BaseConverter):
    """Loads a raster image file as a single-page document."""

    _supported_extensions = sorted(IMAGE_EXTENSIONS)

    def supports(self, path: str) -> bool:
        return self._extension(path) in IMAGE_EXTENSIONS

    def convert(self, path: str, pages: Optional[List[int]] = None) -> List[Image.Image]:
        """Open the image and return it as a one-element RGB list.

        ``pages`` is accepted for interface compatibility but ignored: an
        image is always exactly one page.
        """
        self._require_file(path)
        try:
            with Image.open(path) as img:
                return [img.convert("RGB")]
        except OSError as exc:
            raise ConversionError(f"Could not open image {path!r}: {exc}") from exc
