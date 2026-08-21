"""Converter subpackage: documents and images -> page images."""

from structmd.converters.base import (
    IMAGE_EXTENSIONS,
    OFFICE_EXTENSIONS,
    PDF_EXTENSIONS,
    BaseConverter,
    detect_converter,
)
from structmd.converters.image import ImageConverter
from structmd.converters.office import OfficeConverter
from structmd.converters.pdf import PDFConverter

__all__ = [
    "BaseConverter",
    "PDFConverter",
    "OfficeConverter",
    "ImageConverter",
    "detect_converter",
    "IMAGE_EXTENSIONS",
    "OFFICE_EXTENSIONS",
    "PDF_EXTENSIONS",
]
