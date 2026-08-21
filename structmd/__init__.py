"""structmd — document-to-Markdown conversion via Ollama VLMs.

Two-stage architecture:
  Stage 1: VLM layout extraction to JSON (inspectable, cacheable).
  Stage 2: deterministic Markdown building from that JSON.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version

from structmd.assets import AssetExtractor, attach_assets
from structmd.batch.processor import BatchProcessor
from structmd.builders.markdown import BuilderConfig, MarkdownBuilder
from structmd.cache.manager import CacheManager
from structmd.config import StructMDConfig, load_config
from structmd.converters.base import BaseConverter, detect_converter
from structmd.converters.image import ImageConverter
from structmd.converters.office import OfficeConverter
from structmd.converters.pdf import PDFConverter
from structmd.core import (
    BoundingBox,
    DocumentElement,
    ElementType,
    ExtractedDocument,
    ExtractedPage,
    MarkdownDocument,
)
from structmd.extractors.base import BaseExtractor
from structmd.extractors.ollama import OllamaExtractor
from structmd.pipeline import StructMDPipeline

try:
    __version__ = _package_version("structmd")
except PackageNotFoundError:  # running from a source checkout without install
    __version__ = "0.0.0.dev0"

__all__ = [
    # Pipeline & config
    "StructMDPipeline",
    "StructMDConfig",
    "load_config",
    # Core models
    "BoundingBox",
    "DocumentElement",
    "ElementType",
    "ExtractedDocument",
    "ExtractedPage",
    "MarkdownDocument",
    # Extraction
    "BaseExtractor",
    "OllamaExtractor",
    # Building
    "MarkdownBuilder",
    "BuilderConfig",
    # Conversion
    "BaseConverter",
    "PDFConverter",
    "OfficeConverter",
    "ImageConverter",
    "detect_converter",
    # Batch & cache
    "BatchProcessor",
    "CacheManager",
    # Figure assets
    "AssetExtractor",
    "attach_assets",
    "__version__",
]
