"""Tests for structmd.core data models."""

from __future__ import annotations

import json

import pytest

from structmd.core import (
    BoundingBox,
    DocumentElement,
    ElementType,
    ExtractedDocument,
    ExtractedPage,
    MarkdownDocument,
)


class TestBoundingBox:
    def test_basic_properties(self) -> None:
        box = BoundingBox(10, 20, 110, 120)
        assert box.width == 100
        assert box.height == 100
        assert box.area == 10000
        assert box.center == (60.0, 70.0)

    def test_to_tuple_and_dict(self) -> None:
        box = BoundingBox(1.0, 2.0, 3.0, 4.0)
        assert box.to_tuple() == (1.0, 2.0, 3.0, 4.0)
        assert box.to_dict() == {"x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0}

    def test_from_dict(self) -> None:
        box = BoundingBox.from_dict({"x1": 1, "y1": 2, "x2": 3, "y2": 4})
        assert box == BoundingBox(1.0, 2.0, 3.0, 4.0)

    def test_from_list(self) -> None:
        """VLMs usually emit [x1, y1, x2, y2] lists."""
        box = BoundingBox.from_dict([5, 6, 7, 8])
        assert box == BoundingBox(5.0, 6.0, 7.0, 8.0)

    def test_from_dict_rejects_missing_coordinates(self) -> None:
        with pytest.raises(ValueError, match="missing coordinate"):
            BoundingBox.from_dict({"x1": 1, "y1": 2, "x2": 3})

    def test_from_dict_rejects_non_numeric_values(self) -> None:
        with pytest.raises(ValueError, match="Non-numeric"):
            BoundingBox.from_dict({"x1": "a", "y1": 2, "x2": 3, "y2": 4})
        with pytest.raises(ValueError, match="Non-numeric"):
            BoundingBox.from_dict([1, 2, "oops", 4])

    def test_from_dict_rejects_wrong_length_sequence(self) -> None:
        with pytest.raises(ValueError, match="expected 4 coordinates"):
            BoundingBox.from_dict([1, 2, 3])
        with pytest.raises(ValueError, match="expected 4 coordinates"):
            BoundingBox.from_dict([1, 2, 3, 4, 5])

    def test_from_dict_rejects_unknown_payload_type(self) -> None:
        with pytest.raises(ValueError, match="expected a dict or 4-item sequence"):
            BoundingBox.from_dict("1,2,3,4")  # type: ignore[arg-type]

    def test_intersects_full_overlap(self) -> None:
        a = BoundingBox(0, 0, 100, 100)
        b = BoundingBox(10, 10, 90, 90)
        assert a.intersects(b, threshold=0.5)
        assert b.intersects(a, threshold=0.5)

    def test_intersects_partial_below_threshold(self) -> None:
        a = BoundingBox(0, 0, 100, 100)
        # Overlap is 10x100 = 1000; min area is 100*100=10000 -> ratio 0.1
        b = BoundingBox(95, 0, 205, 100)
        assert not a.intersects(b, threshold=0.5)
        assert a.intersects(b, threshold=0.05)

    def test_intersects_no_overlap(self) -> None:
        a = BoundingBox(0, 0, 10, 10)
        b = BoundingBox(20, 20, 30, 30)
        assert not a.intersects(b)

    def test_contains(self) -> None:
        outer = BoundingBox(0, 0, 100, 100)
        inner = BoundingBox(25, 25, 75, 75)
        assert outer.contains(inner)
        assert not inner.contains(outer)

    def test_contains_threshold(self) -> None:
        outer = BoundingBox(0, 0, 100, 100)
        # Intersection = 80x80 = 6400; other.area = 8100 -> ratio ~0.79
        poking_out = BoundingBox(20, 20, 110, 110)
        assert outer.contains(poking_out, threshold=0.7)
        assert not outer.contains(poking_out, threshold=0.9)


class TestDocumentElement:
    def make_element(self) -> DocumentElement:
        return DocumentElement(
            id="el-1",
            type=ElementType.HEADING,
            bbox=BoundingBox(0, 0, 200, 40),
            text="Introduction",
            page_number=1,
            confidence=0.97,
            heading_level=1,
        )

    def test_serialization_shape(self) -> None:
        el = self.make_element()
        d = el.to_dict()
        assert d["type"] == "heading"  # serialized as string
        assert isinstance(d["bbox"], dict)  # serialized as dict
        assert d["heading_level"] == 1

    def test_roundtrip_dict(self) -> None:
        el = self.make_element()
        el2 = DocumentElement.from_dict(el.to_dict())
        assert el2.id == "el-1"
        assert el2.type == ElementType.HEADING
        assert el2.bbox == BoundingBox(0, 0, 200, 40)
        assert el2.heading_level == 1
        assert el2.confidence == pytest.approx(0.97)

    def test_roundtrip_json(self) -> None:
        el = self.make_element()
        el2 = DocumentElement.from_json(el.to_json())
        assert el2.to_dict() == el.to_dict()

    def test_auto_generated_id_is_stable_after_roundtrip(self) -> None:
        el = DocumentElement(type=ElementType.PARAGRAPH, text="hi", page_number=1)
        el2 = DocumentElement.from_dict(el.to_dict())
        assert el2.id == el.id

    def test_accepts_string_type(self) -> None:
        el = DocumentElement(type="table", text="t", page_number=1)
        assert el.type == ElementType.TABLE

    def test_rejects_unknown_type(self) -> None:
        with pytest.raises(ValueError, match="Unknown element type"):
            DocumentElement(type="hieroglyph", text="x", page_number=1)

    def test_from_dict_ignores_unknown_keys(self) -> None:
        """Small VLMs hallucinate extra keys; parsing must tolerate them."""
        el = DocumentElement.from_dict(
            {"type": "paragraph", "text": "hello", "page_number": 2, "font_size": 12}
        )
        assert el.text == "hello"
        assert el.page_number == 2


class TestExtractedPageAndDocument:
    def build_document(self) -> ExtractedDocument:
        page = ExtractedPage(
            page_number=1,
            width=612,
            height=792,
            elements=[
                DocumentElement(
                    type=(
                        ElementType.TITLE if hasattr(ElementType, "TITLE") else ElementType.HEADING
                    ),
                    text="Report",
                    page_number=1,
                    heading_level=1,
                    bbox=BoundingBox(0, 0, 300, 30),
                ),
                DocumentElement(
                    type=ElementType.PARAGRAPH,
                    text="Body text.",
                    page_number=1,
                    bbox=BoundingBox(0, 50, 500, 90),
                ),
            ],
        )
        return ExtractedDocument(
            document_id="doc-1",
            source_path="sample-report.pdf",
            page_count=1,
            pages=[page],
            metadata={"model": "qwen2-vl:2b"},
        )

    def test_document_json_roundtrip(self) -> None:
        doc = self.build_document()
        restored = ExtractedDocument.from_json(doc.to_json())
        assert restored.document_id == "doc-1"
        assert restored.page_count == 1
        assert len(restored.pages[0].elements) == 2
        assert restored.pages[0].elements[0].text == "Report"
        assert restored.metadata["model"] == "qwen2-vl:2b"

    def test_to_json_is_valid_json(self) -> None:
        doc = self.build_document()
        parsed = json.loads(doc.to_json())
        assert parsed["pages"][0]["page_number"] == 1

    def test_deterministic_serialization(self) -> None:
        """Same document must serialize byte-for-byte identically."""
        doc = self.build_document()
        assert doc.to_json() == doc.to_json()

    def test_save_json_creates_parents(self, tmp_path) -> None:
        doc = self.build_document()
        out = tmp_path / "deep" / "nested" / "extraction.json"
        written = doc.save_json(str(out))
        assert written.exists()
        assert (
            ExtractedDocument.from_json(written.read_text(encoding="utf-8")).document_id == "doc-1"
        )


class TestMarkdownDocument:
    def test_save_prepends_title(self, tmp_path) -> None:
        md = MarkdownDocument(title="My Report", content="Some body text.")
        path = md.save(str(tmp_path / "out.md"))
        text = path.read_text(encoding="utf-8")
        assert text.startswith("# My Report\n\n")
        assert "Some body text." in text

    def test_save_keeps_existing_h1(self, tmp_path) -> None:
        md = MarkdownDocument(title="Ignored", content="# Already Titled\n\nbody")
        path = md.save(str(tmp_path / "out.md"))
        text = path.read_text(encoding="utf-8")
        assert text.count("# Already Titled") == 1
        assert "# Ignored" not in text

    def test_save_without_title(self, tmp_path) -> None:
        md = MarkdownDocument(content="plain")
        path = md.save(str(tmp_path / "sub" / "out.md"))
        assert path.read_text(encoding="utf-8") == "plain"
