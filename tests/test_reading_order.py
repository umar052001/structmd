"""Reading-order tests: synthetic bboxes verified through _resolve_columns."""

from __future__ import annotations

from typing import List

from structmd.builders.markdown import MarkdownBuilder
from structmd.core import BoundingBox, DocumentElement


def el(
    text: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    page: int = 1,
    type_: str = "paragraph",
) -> DocumentElement:
    return DocumentElement(
        type=type_,
        text=text,
        page_number=page,
        bbox=BoundingBox(x1, y1, x2, y2),
    )


def order_texts(elements: List[DocumentElement]) -> List[str]:
    """Run the real pipeline order: flatten_and_sort -> resolve_columns."""
    builder = MarkdownBuilder()
    from structmd.core import ExtractedDocument, ExtractedPage

    doc = ExtractedDocument(
        pages=[ExtractedPage(page_number=1, width=612.0, height=792.0, elements=elements)]
    )
    resolved = builder._resolve_columns(builder._flatten_and_sort(doc))
    return [e.text for e in resolved]


class TestSingleColumn:
    def test_y_then_x_order(self) -> None:
        elements = [
            el("right", 300, 0, 500, 20),
            el("left", 0, 0, 200, 20),
            el("below", 0, 40, 500, 60),
        ]
        assert order_texts(elements) == ["left", "right", "below"]

    def test_pages_grouped_in_order(self) -> None:
        elements = [
            el("p2-first", 0, 10, 100, 20, page=2),
            el("p1-last", 0, 400, 100, 420, page=1),
        ]
        assert order_texts(elements) == ["p1-last", "p2-first"]


class TestMultiColumn:
    def test_two_column_interleave(self) -> None:
        """Classic two-column academic layout on a 612pt-wide page."""
        # mid = 612/2 = 306; tolerance 50 -> left: x2 <= 356, right: x1 >= 256
        elements = [
            # Full-width title spanning both columns.
            el("TITLE", 50, 20, 560, 60, type_="heading"),
            # Left column.
            el("L1", 60, 90, 280, 150),
            el("L2", 60, 160, 280, 220),
            el("L3", 60, 230, 280, 290),
            # Right column.
            el("R1", 330, 95, 550, 155),
            el("R2", 330, 165, 550, 225),
            el("R3", 330, 235, 550, 295),
        ]
        assert order_texts(elements) == ["TITLE", "L1", "R1", "L2", "R2", "L3", "R3"]

    def test_full_width_element_breaks_columns(self) -> None:
        """A full-width table between column blocks is emitted first."""
        elements = [
            el("L1", 60, 90, 280, 150),
            el("L2", 60, 160, 280, 220),
            el("L3", 60, 230, 280, 290),
            el("WIDE", 55, 320, 555, 380),  # spans the middle
            el("R1", 330, 95, 550, 155),
            el("R2", 330, 165, 550, 225),
            el("R3", 330, 235, 550, 295),
        ]
        result = order_texts(elements)
        assert result[0] == "WIDE"
        # Columns still interleaved after the wide element.
        assert result[1:] == ["L1", "R1", "L2", "R2", "L3", "R3"]

    def test_insufficient_elements_not_split(self) -> None:
        """Few elements must never trigger the column heuristic."""
        elements = [
            el("a", 0, 0, 250, 30),
            el("b", 350, 0, 600, 30),
            el("c", 0, 40, 250, 70),
            el("d", 350, 40, 600, 70),
        ]
        assert order_texts(elements) == ["a", "b", "c", "d"]

    def test_unbalanced_columns_not_split(self) -> None:
        """One crowded column + one stray element stays single-column.

        The heuristic must refuse to split (right column too sparse), so the
        stray element simply lands at its natural y-position in the flow.
        """
        elements = [el(f"L{i}", 60, 50 + i * 40, 280, 80 + i * 40) for i in range(6)] + [
            el("stray-right", 400, 60, 560, 90)
        ]
        result = order_texts(elements)
        assert result == ["L0", "stray-right"] + [f"L{i}" for i in range(1, 6)]

    def test_column_detection_disabled(self) -> None:
        builder = MarkdownBuilder()
        builder.config.detect_columns = False
        elements = [
            el("L1", 60, 90, 280, 150),
            el("L2", 60, 160, 280, 220),
            el("L3", 60, 230, 280, 290),
            el("R1", 330, 95, 550, 155),
            el("R2", 330, 165, 550, 225),
            el("R3", 330, 235, 550, 295),
        ]
        # With detection off, pure (page, y1, x1) order applies: the right
        # column's first line shares y-band with L1 and wins on nothing —
        # strict y-sort interleaves by y1.
        from structmd.core import ExtractedDocument, ExtractedPage

        doc = ExtractedDocument(
            pages=[ExtractedPage(page_number=1, width=612.0, height=792.0, elements=elements)]
        )
        resolved = builder._resolve_columns(builder._flatten_and_sort(doc))
        texts = [e.text for e in resolved]
        assert texts.index("L1") < texts.index("L2") < texts.index("L3")
        assert texts.index("R1") < texts.index("R2") < texts.index("R3")
        assert texts != ["L1", "L2", "L3", "R1", "R2", "R3"]  # columns NOT grouped


class TestEndToEndReadingOrder:
    def test_multicolumn_markdown_reads_naturally(self) -> None:
        """Full pipeline: interleaved columns produce interleaved markdown."""
        from structmd.builders.markdown import BuilderConfig

        builder = MarkdownBuilder(BuilderConfig(include_page_numbers=False))
        doc = builder.build(
            type(
                "D",
                (),
                {
                    "document_id": "x",
                    "source_path": None,
                    "page_count": 1,
                    "metadata": {},
                    "pages": [
                        type(
                            "P",
                            (),
                            {
                                "page_number": 1,
                                "width": 612.0,
                                "height": 792.0,
                                "elements": [
                                    el("Left intro sentence.", 60, 90, 280, 150),
                                    el("Right intro sentence.", 330, 95, 550, 155),
                                    el("Left second.", 60, 160, 280, 220),
                                    el("Right second.", 330, 165, 550, 225),
                                    el("Left third.", 60, 230, 280, 290),
                                    el("Right third.", 330, 235, 550, 295),
                                ],
                            },
                        )()
                    ],
                },
            )()
        )
        lines = doc.content.splitlines()
        assert lines.index("Left intro sentence.") < lines.index("Right intro sentence.")
        assert lines.index("Right intro sentence.") < lines.index("Left second.")
