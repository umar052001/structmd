"""Core data models for structmd.

This module defines the JSON-intermediate representation that sits between the
VLM extraction stage (Stage 1) and the deterministic Markdown builder (Stage 2).

The JSON produced by these models is the single source of truth for the rest of
the pipeline: it is inspectable, editable, cacheable, and deterministic.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class StructMDError(Exception):
    """Base class for all structmd errors."""


class OllamaConnectionError(StructMDError):
    """Raised when the Ollama server cannot be reached or keeps failing."""


class ModelNotFoundError(StructMDError):
    """Raised when the configured model is not present on the Ollama server."""


class ConversionError(StructMDError):
    """Raised when an input document cannot be converted to page images."""


class CacheError(StructMDError):
    """Raised when cache read/write fails in a non-recoverable way."""


class ElementType(str, Enum):
    """Semantic element types recognized during VLM layout extraction."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    LIST_ITEM = "list_item"
    CAPTION = "caption"
    PAGE_NUMBER = "page_number"
    FOOTNOTE = "footnote"
    HEADER = "header"
    FOOTER = "footer"
    IMAGE = "image"
    CODE_BLOCK = "code_block"
    BLOCKQUOTE = "blockquote"
    HORIZONTAL_RULE = "horizontal_rule"


@dataclass
class BoundingBox:
    """Axis-aligned bounding box in page coordinates.

    Coordinates follow image conventions: origin at the top-left corner,
    ``x`` grows rightwards and ``y`` grows downwards. ``(x1, y1)`` is the
    top-left corner and ``(x2, y2)`` is the bottom-right corner.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        """Width of the box."""
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        """Height of the box."""
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        """Area of the box."""
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        """Center point of the box as ``(cx, cy)``."""
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def to_tuple(self) -> Tuple[float, float, float, float]:
        """Return the box as a plain ``(x1, y1, x2, y2)`` tuple."""
        return (self.x1, self.y1, self.x2, self.y2)

    def to_dict(self) -> Dict[str, float]:
        """Serialize to a dict suitable for JSON output."""
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BoundingBox:
        """Build a :class:`BoundingBox` from a dict with keys ``x1..y2``.

        Also accepts a 4-item list/tuple ``[x1, y1, x2, y2]`` which is the shape
        VLMs most commonly emit.
        """
        if isinstance(data, (list, tuple)) and len(data) == 4:
            return cls(x1=float(data[0]), y1=float(data[1]), x2=float(data[2]), y2=float(data[3]))
        return cls(
            x1=float(data["x1"]),
            y1=float(data["y1"]),
            x2=float(data["x2"]),
            y2=float(data["y2"]),
        )

    def _intersection_area(self, other: BoundingBox) -> float:
        """Compute the raw intersection area with another box."""
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        return (ix2 - ix1) * (iy2 - iy1)

    def intersects(self, other: BoundingBox, threshold: float = 0.5) -> bool:
        """Return True when the overlap with ``other`` is significant.

        The overlap ratio is computed as intersection area divided by the
        smaller of the two box areas (intersection-over-min). A threshold of
        0.5 means at least half of the smaller box must be covered.
        """
        inter = self._intersection_area(other)
        if inter <= 0.0:
            return False
        denom = min(self.area, other.area)
        if denom <= 0.0:
            return False
        return (inter / denom) >= threshold

    def contains(self, other: BoundingBox, threshold: float = 0.9) -> bool:
        """Return True when ``other`` lies mostly inside this box.

        The containment ratio is intersection area divided by the area of
        ``other``. A threshold of 0.9 means 90% of ``other`` must be inside.
        """
        inter = self._intersection_area(other)
        if other.area <= 0.0:
            return False
        return (inter / other.area) >= threshold


@dataclass
class DocumentElement:
    """A single semantic element extracted from one document page."""

    id: Optional[str] = None
    type: ElementType = ElementType.PARAGRAPH
    bbox: Optional[BoundingBox] = None
    text: str = ""
    page_number: int = 0
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    heading_level: Optional[int] = None
    list_type: Optional[str] = None  # "ordered" | "unordered"
    indent_level: int = 0
    continues_on_next_page: bool = False
    table_data: Optional[List[List[str]]] = None

    def __post_init__(self) -> None:
        if self.id is None:
            self.id = f"elem_{uuid.uuid4().hex[:12]}"
        # Accept either an ElementType or a raw string on construction.
        if isinstance(self.type, str):
            try:
                self.type = ElementType(self.type)
            except ValueError:
                raise ValueError(
                    f"Unknown element type {self.type!r}. "
                    f"Valid types: {[t.value for t in ElementType]}"
                ) from None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-safe dict.

        ``type`` is serialized as its string value and ``bbox`` as a dict.
        """
        data = asdict(self)
        data["type"] = self.type.value
        data["bbox"] = self.bbox.to_dict() if self.bbox is not None else None
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DocumentElement:
        """Rebuild a :class:`DocumentElement` from :meth:`to_dict` output.

        Tolerates partial dicts (e.g. straight from a small VLM): missing
        optional fields fall back to defaults and unknown types are reported
        clearly rather than crashing deep inside the builder.
        """
        payload = dict(data)
        raw_type = payload.pop("type", ElementType.PARAGRAPH)
        if isinstance(raw_type, ElementType):
            element_type: ElementType = raw_type
        else:
            try:
                element_type = ElementType(str(raw_type))
            except ValueError:
                raise ValueError(
                    f"Unknown element type {raw_type!r}. "
                    f"Valid types: {[t.value for t in ElementType]}"
                ) from None

        raw_bbox = payload.pop("bbox", None)
        bbox = BoundingBox.from_dict(raw_bbox) if raw_bbox is not None else None

        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in payload.items() if k in known_fields}
        return cls(type=element_type, bbox=bbox, **filtered)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> DocumentElement:
        """Deserialize from a JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class ExtractedPage:
    """All elements extracted from a single page."""

    page_number: int = 0
    width: float = 0.0
    height: float = 0.0
    elements: List[DocumentElement] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "page_number": self.page_number,
            "width": self.width,
            "height": self.height,
            "elements": [el.to_dict() for el in self.elements],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExtractedPage:
        """Rebuild an :class:`ExtractedPage` from :meth:`to_dict` output."""
        return cls(
            page_number=int(data.get("page_number", 0)),
            width=float(data.get("width", 0.0)),
            height=float(data.get("height", 0.0)),
            elements=[DocumentElement.from_dict(el) for el in data.get("elements", [])],
        )


@dataclass
class ExtractedDocument:
    """The full extraction result for a document — Stage 1 output."""

    document_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    source_path: Optional[str] = None
    page_count: int = 0
    pages: List[ExtractedPage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "document_id": self.document_id,
            "source_path": self.source_path,
            "page_count": self.page_count,
            "pages": [page.to_dict() for page in self.pages],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExtractedDocument:
        """Rebuild an :class:`ExtractedDocument` from :meth:`to_dict` output."""
        return cls(
            document_id=data.get("document_id", uuid.uuid4().hex),
            source_path=data.get("source_path"),
            page_count=int(data.get("page_count", 0)),
            pages=[ExtractedPage.from_dict(p) for p in data.get("pages", [])],
            metadata=data.get("metadata", {}),
        )

    def to_json(self, indent: int = 2) -> str:
        """Serialize to a pretty-printed JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> ExtractedDocument:
        """Deserialize from a JSON string."""
        return cls.from_dict(json.loads(json_str))

    def save_json(self, path: str) -> Path:
        """Write the extraction JSON to ``path`` and return it."""
        out_path = Path(path).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(self.to_json(), encoding="utf-8")
        return out_path


@dataclass
class MarkdownDocument:
    """Final Stage 2 output: rendered Markdown plus metadata."""

    title: Optional[str] = None
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: str) -> Path:
        """Write Markdown to ``path``.

        If a title exists and the content does not already start with an H1,
        the title is prepended as ``# {title}``.
        """
        content = self.content
        if self.title and not content.lstrip().startswith("# "):
            content = f"# {self.title}\n\n{content}"
        out_path = Path(path).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")
        return out_path
