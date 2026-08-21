"""Tests for the deterministic MarkdownBuilder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

from structmd.builders.markdown import BuilderConfig, MarkdownBuilder
from structmd.core import (
    BoundingBox,
    DocumentElement,
    ExtractedDocument,
    ExtractedPage,
)

FIXTURES = Path(__file__).parent / "fixtures"


def make_element(
    type_: str,
    text: str = "",
    page: int = 1,
    bbox=None,
    **kwargs,
) -> DocumentElement:
    return DocumentElement(
        type=type_,
        text=text,
        page_number=page,
        bbox=BoundingBox(*bbox) if bbox else None,
        **kwargs,
    )


def build_doc(pages: List[List[DocumentElement]], width: float = 612.0) -> ExtractedDocument:
    return ExtractedDocument(
        document_id="test-doc",
        source_path="test.pdf",
        page_count=len(pages),
        pages=[
            ExtractedPage(page_number=i + 1, width=width, height=792.0, elements=els)
            for i, els in enumerate(pages)
        ],
    )


@pytest.fixture()
def sample_document() -> ExtractedDocument:
    data = json.loads((FIXTURES / "sample_extraction.json").read_text(encoding="utf-8"))
    return ExtractedDocument.from_dict(data)


class TestHeadings:
    def test_heading_rendering_and_title(self) -> None:
        builder = MarkdownBuilder(BuilderConfig(include_page_numbers=False))
        doc = builder.build(build_doc([[make_element("heading", "My Title", heading_level=1)]]))
        assert doc.content == "# My Title\n"
        assert doc.title == "My Title"

    def test_nested_sections(self) -> None:
        builder = MarkdownBuilder(BuilderConfig(include_page_numbers=False))
        doc = builder.build(
            build_doc(
                [
                    [
                        make_element("heading", "Top", heading_level=1),
                        make_element("paragraph", "intro"),
                        make_element("heading", "Sub", heading_level=2),
                        make_element("paragraph", "detail"),
                    ]
                ]
            )
        )
        assert "# Top\n\nintro\n\n## Sub\n\ndetail" in doc.content


class TestParagraphs:
    def test_paragraph_order(self) -> None:
        builder = MarkdownBuilder(BuilderConfig(include_page_numbers=False))
        doc = builder.build(
            build_doc(
                [
                    [
                        make_element("paragraph", "first", bbox=(0, 0, 100, 20)),
                        make_element("paragraph", "second", bbox=(0, 30, 100, 50)),
                    ]
                ]
            )
        )
        assert doc.content == "first\n\nsecond\n"

    def test_continued_paragraph_merged(self) -> None:
        builder = MarkdownBuilder(BuilderConfig(include_page_numbers=False))
        doc = builder.build(
            build_doc(
                [
                    [make_element("paragraph", "The quick brown fox", continues_on_next_page=True)],
                    [make_element("paragraph", "jumps over the lazy dog.", page=2)],
                ]
            )
        )
        assert "The quick brown fox jumps over the lazy dog." in doc.content

    def test_dehyphenation_on_merge(self) -> None:
        builder = MarkdownBuilder(BuilderConfig(include_page_numbers=False))
        doc = builder.build(
            build_doc(
                [
                    [make_element("paragraph", "a docu-", continues_on_next_page=True)],
                    [make_element("paragraph", "ment continues.", page=2)],
                ]
            )
        )
        assert "a document continues." in doc.content

    def test_merge_disabled(self) -> None:
        cfg = BuilderConfig(include_page_numbers=False, merge_continued_paragraphs=False)
        builder = MarkdownBuilder(cfg)
        doc = builder.build(
            build_doc(
                [
                    [make_element("paragraph", "part one", continues_on_next_page=True)],
                    [make_element("paragraph", "part two", page=2)],
                ]
            )
        )
        assert "part one\n\npart two" in doc.content

    def test_standalone_asterisk_escaped(self) -> None:
        builder = MarkdownBuilder(BuilderConfig(include_page_numbers=False))
        doc = builder.build(build_doc([[make_element("paragraph", "* important note")]]))
        assert "\\* important note" in doc.content
        # Paired emphasis must survive.
        doc2 = builder.build(build_doc([[make_element("paragraph", "this is *italic* text")]]))
        assert "*italic*" in doc2.content


class TestLists:
    def test_unordered_list(self) -> None:
        builder = MarkdownBuilder(BuilderConfig(include_page_numbers=False))
        doc = builder.build(
            build_doc(
                [
                    [
                        make_element("list_item", "alpha", list_type="unordered"),
                        make_element("list_item", "beta", list_type="unordered"),
                    ]
                ]
            )
        )
        assert "- alpha\n- beta" in doc.content

    def test_ordered_list_numbering(self) -> None:
        builder = MarkdownBuilder(BuilderConfig(include_page_numbers=False))
        doc = builder.build(
            build_doc(
                [
                    [
                        make_element("list_item", "one", list_type="ordered"),
                        make_element("list_item", "two", list_type="ordered"),
                    ]
                ]
            )
        )
        assert "1. one\n2. two" in doc.content

    def test_new_list_on_type_change(self) -> None:
        builder = MarkdownBuilder(BuilderConfig(include_page_numbers=False))
        doc = builder.build(
            build_doc(
                [
                    [
                        make_element("list_item", "bullet", list_type="unordered"),
                        make_element("list_item", "numbered", list_type="ordered"),
                    ]
                ]
            )
        )
        assert "- bullet\n\n1. numbered" in doc.content

    def test_indent_levels(self) -> None:
        builder = MarkdownBuilder(BuilderConfig(include_page_numbers=False))
        doc = builder.build(
            build_doc(
                [
                    [
                        make_element("list_item", "parent", list_type="unordered", indent_level=0),
                        make_element("list_item", "child", list_type="unordered", indent_level=1),
                    ]
                ]
            )
        )
        assert "- parent\n  - child" in doc.content


class TestTables:
    def test_table_rendering_with_padding(self) -> None:
        builder = MarkdownBuilder(BuilderConfig(include_page_numbers=False))
        doc = builder.build(
            build_doc(
                [
                    [
                        make_element(
                            "table",
                            table_data=[["A", "B"], ["x"], ["y", "z"]],
                        )
                    ]
                ]
            )
        )
        expected = "| A | B |\n| --- | --- |\n| x |  |\n| y | z |"
        assert expected in doc.content

    def test_caption_after_table(self) -> None:
        cfg = BuilderConfig(include_page_numbers=False, table_caption_position="after")
        builder = MarkdownBuilder(cfg)
        doc = builder.build(
            build_doc(
                [
                    [
                        make_element("table", table_data=[["H"], ["v"]]),
                        make_element("caption", "Table 1: stuff"),
                    ]
                ]
            )
        )
        assert "| H |\n| --- |\n| v |\n*Table 1: stuff*" in doc.content

    def test_caption_before_table_is_default(self) -> None:
        builder = MarkdownBuilder(BuilderConfig(include_page_numbers=False))
        doc = builder.build(
            build_doc(
                [
                    [
                        make_element("caption", "Table 2: before"),
                        make_element("table", table_data=[["H"], ["v"]]),
                    ]
                ]
            )
        )
        assert "*Table 2: before*\n| H |\n| --- |\n| v |" in doc.content

    def test_caption_position_after_config(self) -> None:
        cfg = BuilderConfig(include_page_numbers=False, table_caption_position="after")
        builder = MarkdownBuilder(cfg)
        doc = builder.build(
            build_doc(
                [
                    [
                        make_element("caption", "Cap"),
                        make_element("table", table_data=[["H"], ["v"]]),
                    ]
                ]
            )
        )
        assert "| H |\n| --- |\n| v |\n*Cap*" in doc.content


class TestImagesAndCodeAndQuotes:
    def test_image_with_caption(self) -> None:
        builder = MarkdownBuilder(BuilderConfig(include_page_numbers=False))
        doc = builder.build(
            build_doc(
                [
                    [
                        make_element("image", "A chart of sales"),
                        make_element("caption", "Figure 1: Sales"),
                    ]
                ]
            )
        )
        assert "![A chart of sales](image_placeholder)\n*Figure 1: Sales*" in doc.content

    def test_code_block_with_language(self) -> None:
        builder = MarkdownBuilder(BuilderConfig(include_page_numbers=False))
        el = make_element("code_block", "print('hi')", metadata={"language": "python"})
        doc = builder.build(build_doc([[el]]))
        assert "```python\nprint('hi')\n```" in doc.content

    def test_blockquote(self) -> None:
        builder = MarkdownBuilder(BuilderConfig(include_page_numbers=False))
        doc = builder.build(build_doc([[make_element("blockquote", "wise words")]]))
        assert "> wise words" in doc.content

    def test_horizontal_rule(self) -> None:
        builder = MarkdownBuilder(BuilderConfig(include_page_numbers=False))
        doc = builder.build(build_doc([[make_element("horizontal_rule")]]))
        assert "---" in doc.content


class TestChromeAndPageNumbers:
    def test_header_footer_pagenumber_skipped_but_recorded(self) -> None:
        builder = MarkdownBuilder(BuilderConfig(include_page_numbers=True))
        doc = builder.build(
            build_doc(
                [
                    [
                        make_element("header", "CONFIDENTIAL"),
                        make_element("paragraph", "body"),
                        make_element("footer", "legal line"),
                        make_element("page_number", "7"),
                    ]
                ]
            )
        )
        assert "CONFIDENTIAL" not in doc.content
        assert "legal line" not in doc.content
        assert "body" in doc.content
        assert doc.metadata["page_numbers"] == [7]

    def test_page_break_markers_inserted(self) -> None:
        builder = MarkdownBuilder(BuilderConfig(include_page_numbers=True))
        doc = builder.build(
            build_doc(
                [
                    [make_element("paragraph", "page one content")],
                    [make_element("paragraph", "page two content", page=2)],
                ]
            )
        )
        assert "<!-- Page 2 -->" in doc.content

    def test_no_page_breaks_when_disabled(self) -> None:
        builder = MarkdownBuilder(BuilderConfig(include_page_numbers=False))
        doc = builder.build(
            build_doc(
                [
                    [make_element("paragraph", "a")],
                    [make_element("paragraph", "b", page=2)],
                ]
            )
        )
        assert "Page" not in doc.content


class TestHeadingNormalization:
    def test_levels_remapped_sequentially(self) -> None:
        builder = MarkdownBuilder(
            BuilderConfig(normalize_headings=True, include_page_numbers=False)
        )
        doc = builder.build(
            build_doc(
                [
                    [
                        make_element("heading", "A", heading_level=1),
                        make_element("heading", "B", heading_level=3),
                        make_element("heading", "C", heading_level=3),
                    ]
                ]
            )
        )
        assert "# A" in doc.content
        assert "## B" in doc.content
        lines = doc.content.splitlines()
        assert lines[0] == "# A"
        assert "## B" in lines and "## C" in lines
        assert not any(line.startswith("### ") for line in lines)

    def test_normalization_disabled_preserves_levels(self) -> None:
        builder = MarkdownBuilder(
            BuilderConfig(normalize_headings=False, include_page_numbers=False)
        )
        doc = builder.build(
            build_doc(
                [
                    [
                        make_element("heading", "A", heading_level=2),
                        make_element("heading", "B", heading_level=5),
                    ]
                ]
            )
        )
        assert "## A" in doc.content
        assert "##### B" in doc.content


class TestDeterminism:
    def test_identical_json_identical_output(self, sample_document: ExtractedDocument) -> None:
        b1 = MarkdownBuilder().build(sample_document)
        b2 = MarkdownBuilder().build(sample_document)
        assert b1.content == b2.content
        assert b1.title == b2.title

    def test_sample_fixture_end_to_end(self, sample_document: ExtractedDocument) -> None:
        doc = MarkdownBuilder().build(sample_document)
        assert doc.title == "Quarterly Business Report"
        assert "| Metric | Q1 | Q2 |" in doc.content
        assert "*Table 1: Consolidated financials*" in doc.content
        # Continued paragraph across pages 2->3 merged.
        assert "Singapore and Tokyo come online ahead of schedule." in doc.content
        # Image consumed its caption.
        assert "*Figure 1: Updated organizational structure effective July 1*" in doc.content
        # Chrome dropped.
        assert "Confidential" not in doc.content
