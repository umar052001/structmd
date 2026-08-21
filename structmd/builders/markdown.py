"""MarkdownBuilder — Stage 2, "the cutter".

Deterministic conversion of :class:`~structmd.core.ExtractedDocument` JSON
into Markdown. No ML, no images, no randomness: identical JSON always yields
byte-for-byte identical Markdown.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from structmd.core import (
    DocumentElement,
    ElementType,
    ExtractedDocument,
    MarkdownDocument,
)

logger = logging.getLogger(__name__)

COLUMN_TOLERANCE_PX = 50.0


@dataclass
class BuilderConfig:
    """Knobs for the deterministic Markdown builder."""

    include_page_numbers: bool = True
    page_number_format: str = "\n<!-- Page {page} -->\n"
    merge_continued_paragraphs: bool = True
    paragraph_separator: str = "\n\n"
    detect_columns: bool = True
    column_threshold: float = 0.6
    normalize_headings: bool = True
    table_caption_position: str = "before"


# ---------------------------------------------------------------------------
# Internal tree nodes
# ---------------------------------------------------------------------------


@dataclass
class SectionNode:
    """A heading plus everything nested under it."""

    level: int
    title: str
    children: List[Node] = field(default_factory=list)


@dataclass
class ParagraphNode:
    text: str


@dataclass
class ListNode:
    list_type: str  # "ordered" | "unordered"
    items: List[Dict[str, Any]] = field(default_factory=list)  # {text, indent_level}


@dataclass
class TableNode:
    rows: List[List[str]]
    caption: Optional[str] = None


@dataclass
class ImageNode:
    description: str = ""
    caption: Optional[str] = None
    src: Optional[str] = None  # relative path when asset extraction ran


@dataclass
class CodeBlockNode:
    text: str
    language: Optional[str] = None


@dataclass
class BlockquoteNode:
    text: str


@dataclass
class HorizontalRuleNode:
    pass


@dataclass
class PageBreakNode:
    page: int


Node = Union[
    SectionNode,
    ParagraphNode,
    ListNode,
    TableNode,
    ImageNode,
    CodeBlockNode,
    BlockquoteNode,
    HorizontalRuleNode,
    PageBreakNode,
]


def _escape_inline_markdown(text: str) -> str:
    """Escape *standalone* asterisks and leading hashes in paragraph text.

    Paired emphasis (``*i*``, ``**b**``) is preserved; a lone ``*`` used as a
    bullet/dingbat is neutralized so it cannot restructure the document.
    """
    escaped = re.sub(r"(?:(?<=\s)|^)\*(?=\s|$)", r"\\*", text)
    if escaped.lstrip().startswith("#"):
        escaped = escaped.replace("#", "\\#", 1)
    return escaped


def _escape_table_cell(text: str) -> str:
    """Escape pipes and newlines inside table cells."""
    return text.replace("|", "\\|").replace("\n", " ")


class MarkdownBuilder:
    """Builds a :class:`MarkdownDocument` from extracted JSON deterministically."""

    def __init__(self, config: Optional[BuilderConfig] = None) -> None:
        self.config = config or BuilderConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, extracted: ExtractedDocument) -> MarkdownDocument:
        """Run the full deterministic pipeline and return the document."""
        elements = self._flatten_and_sort(extracted)
        elements = self._resolve_columns(elements) if self.config.detect_columns else elements
        roots, page_numbers_seen = self._build_hierarchy(elements)
        if self.config.normalize_headings:
            self._normalize_headings(roots)
        content, title = self._render_markdown(roots)

        return MarkdownDocument(
            title=title,
            content=content,
            metadata={
                "source_path": extracted.source_path,
                "document_id": extracted.document_id,
                "page_count": extracted.page_count or len(extracted.pages),
                "element_count": len(elements),
                "page_numbers": page_numbers_seen,
            },
        )

    # ------------------------------------------------------------------
    # Step 1: flatten + global sort
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_and_sort(extracted: ExtractedDocument) -> List[DocumentElement]:
        """Flatten elements across pages; sort by (page, y1, x1)."""
        flat: List[DocumentElement] = []
        for page in extracted.pages:
            flat.extend(page.elements)

        def sort_key(el: DocumentElement) -> tuple:
            bbox = el.bbox
            y1 = bbox.y1 if bbox else 0.0
            x1 = bbox.x1 if bbox else 0.0
            return (el.page_number, y1, x1)

        return sorted(flat, key=sort_key)

    # ------------------------------------------------------------------
    # Step 2: multi-column resolution
    # ------------------------------------------------------------------

    def _resolve_columns(self, elements: List[DocumentElement]) -> List[DocumentElement]:
        """Reorder elements for natural reading order in multi-column pages.

        Heuristic per page: split elements into left/right columns around the
        horizontal midpoint (with tolerance). When both columns hold enough
        material, emit full-width elements first, then interleave columns by
        y-position. Single-column pages pass through in sorted order.
        """
        by_page: Dict[int, List[DocumentElement]] = {}
        for el in elements:
            by_page.setdefault(el.page_number, []).append(el)

        ordered: List[DocumentElement] = []
        for page_number in sorted(by_page):
            page_elements = by_page[page_number]
            if self._is_multi_column(page_elements):
                logger.debug("Page %d detected as multi-column", page_number)
                ordered.extend(self._interleave_columns(page_elements))
            else:
                ordered.extend(page_elements)
        return ordered

    @staticmethod
    def _page_max_x(page_elements: List[DocumentElement]) -> float:
        boxes = [el.bbox for el in page_elements if el.bbox]
        return max((b.x2 for b in boxes), default=0.0)

    def _is_multi_column(self, page_elements: List[DocumentElement]) -> bool:
        """True when left/right column populations justify interleaving."""
        boxes = [el.bbox for el in page_elements if el.bbox]
        if len(boxes) < 6:  # need enough signal to trust the heuristic
            return False
        mid = self._page_max_x(page_elements) / 2.0
        tol = COLUMN_TOLERANCE_PX
        left = [b for b in boxes if b.x2 <= mid + tol]
        right = [b for b in boxes if b.x1 >= mid - tol]
        total = len(page_elements)
        return (
            len(left) > 0.3 * total
            and len(right) > 0.3 * total
            and len(left) > 2
            and len(right) > 2
        )

    def _interleave_columns(self, page_elements: List[DocumentElement]) -> List[DocumentElement]:
        """Full-width elements first, then left/right merged by y-position."""
        mid = self._page_max_x(page_elements) / 2.0
        tol = COLUMN_TOLERANCE_PX

        def y_key(el: DocumentElement) -> tuple:
            return (el.bbox.y1 if el.bbox else 0.0, el.bbox.x1 if el.bbox else 0.0)

        full_width: List[DocumentElement] = []
        left_col: List[DocumentElement] = []
        right_col: List[DocumentElement] = []
        for el in page_elements:
            bbox = el.bbox
            if bbox is None:
                full_width.append(el)
            elif bbox.x2 <= mid + tol:
                left_col.append(el)
            elif bbox.x1 >= mid - tol:
                right_col.append(el)
            else:
                full_width.append(el)
        left_col.sort(key=y_key)
        right_col.sort(key=y_key)
        full_width.sort(key=y_key)

        merged: List[DocumentElement] = []
        i = j = 0
        while i < len(left_col) and j < len(right_col):
            if y_key(left_col[i]) <= y_key(right_col[j]):
                merged.append(left_col[i])
                i += 1
            else:
                merged.append(right_col[j])
                j += 1
        merged.extend(left_col[i:])
        merged.extend(right_col[j:])
        return full_width + merged

    # ------------------------------------------------------------------
    # Step 3: hierarchy construction
    # ------------------------------------------------------------------

    def _build_hierarchy(self, elements: List[DocumentElement]) -> tuple[List[Node], List[int]]:
        """Convert a flat element stream into a tree of nodes.

        Returns the root-level nodes and the list of page numbers observed on
        PAGE_NUMBER elements (recorded into document metadata).
        """
        roots: List[Node] = []
        section_stack: List[SectionNode] = []
        page_numbers_seen: List[int] = []
        last_element: Optional[DocumentElement] = None
        current_page: Optional[int] = None

        idx = 0
        while idx < len(elements):
            el = elements[idx]
            idx += 1

            # --- chrome elements: record page numbers, otherwise drop ---
            if el.type == ElementType.PAGE_NUMBER:
                match = re.search(r"\d+", el.text)
                if match:
                    page_num = int(match.group())
                    if page_num not in page_numbers_seen:
                        page_numbers_seen.append(page_num)
                continue
            if el.type in (ElementType.HEADER, ElementType.FOOTER):
                continue

            # --- page-break markers between pages ---
            container = self._current_container(roots, section_stack)
            will_merge_across_break = (
                self.config.merge_continued_paragraphs
                and last_element is not None
                and last_element.continues_on_next_page
                and el.type in (ElementType.PARAGRAPH, ElementType.FOOTNOTE)
                and bool(container)
                and isinstance(container[-1], ParagraphNode)
            )
            if (
                self.config.include_page_numbers
                and current_page is not None
                and el.page_number != current_page
                and not will_merge_across_break
            ):
                container.append(PageBreakNode(page=el.page_number))
            current_page = el.page_number

            if el.type == ElementType.HEADING:
                level = el.heading_level or 1
                section = SectionNode(level=level, title=el.text.strip())
                while section_stack and section_stack[-1].level >= level:
                    section_stack.pop()
                self._current_container(roots, section_stack).append(section)
                section_stack.append(section)

            elif el.type == ElementType.PARAGRAPH:
                self._append_paragraph(container, el, last_element)
            elif el.type == ElementType.FOOTNOTE:
                # Rendered as plain paragraphs (v1 keeps footnote text inline).
                self._append_paragraph(container, el, last_element)

            elif el.type == ElementType.LIST_ITEM:
                list_type = el.list_type or "unordered"
                new_list = (
                    not container
                    or not isinstance(container[-1], ListNode)
                    or container[-1].list_type != list_type
                    or (
                        container[-1].items
                        and container[-1].items[-1]["indent_level"] != el.indent_level
                    )
                )
                if new_list:
                    container.append(ListNode(list_type=list_type))
                container[-1].items.append(  # type: ignore[union-attr]
                    {"text": el.text.strip(), "indent_level": el.indent_level}
                )

            elif el.type == ElementType.TABLE:
                caption = None
                # Caption immediately after the table (reading order)?
                if idx < len(elements) and elements[idx].type == ElementType.CAPTION:
                    caption = elements[idx].text.strip()
                    idx += 1
                else:
                    # Or a caption paragraph placed before the table.
                    prev = self._pop_preceding_caption(container)
                    if prev is not None:
                        caption = prev
                container.append(TableNode(rows=el.table_data or [], caption=caption))

            elif el.type == ElementType.IMAGE:
                description = el.text.strip()
                caption = None
                if idx < len(elements) and elements[idx].type == ElementType.CAPTION:
                    caption = elements[idx].text.strip()
                    idx += 1
                container.append(
                    ImageNode(
                        description=description,
                        caption=caption,
                        src=el.metadata.get("asset_path"),
                    )
                )

            elif el.type == ElementType.CODE_BLOCK:
                language = el.metadata.get("language")
                container.append(CodeBlockNode(text=el.text, language=language))

            elif el.type == ElementType.BLOCKQUOTE:
                container.append(BlockquoteNode(text=el.text.strip()))

            elif el.type == ElementType.HORIZONTAL_RULE:
                container.append(HorizontalRuleNode())

            elif el.type == ElementType.CAPTION:
                # Standalone captions were consumed by their parents; anything
                # left over (e.g. orphaned) is kept as emphasized paragraph.
                container.append(ParagraphNode(text=f"*{el.text.strip()}*"))

            last_element = el

        return roots, page_numbers_seen

    @staticmethod
    def _current_container(roots: List[Node], stack: List[SectionNode]) -> List[Node]:
        return stack[-1].children if stack else roots

    def _append_paragraph(
        self,
        container: List[Node],
        el: DocumentElement,
        last_element: Optional[DocumentElement],
    ) -> None:
        """Append a paragraph, merging into the previous one when it continues."""
        prev = container[-1] if container else None
        should_merge = (
            self.config.merge_continued_paragraphs
            and last_element is not None
            and last_element.continues_on_next_page
            and last_element.type in (ElementType.PARAGRAPH, ElementType.FOOTNOTE)
            and isinstance(prev, ParagraphNode)
        )
        if should_merge and isinstance(prev, ParagraphNode):
            prev_text = prev.text.rstrip()
            cont_text = el.text.strip()
            if prev_text.endswith("-"):
                # De-hyphenate words split across the page boundary.
                prev.text = prev_text[:-1] + cont_text
            else:
                prev.text = prev_text + " " + cont_text
            return
        container.append(ParagraphNode(text=el.text.strip()))

    @staticmethod
    def _pop_preceding_caption(container: List[Node]) -> Optional[str]:
        """If the previous node is an emphasized caption paragraph, pop it."""
        if container and isinstance(container[-1], ParagraphNode):
            text = container[-1].text
            if text.startswith("*") and text.endswith("*"):
                container.pop()
                return text.strip("*").strip()
        return None

    # ------------------------------------------------------------------
    # Step 4: heading normalization
    # ------------------------------------------------------------------

    def _normalize_headings(self, roots: List[Node]) -> None:
        """Remap heading levels to be sequential by first appearance.

        A document using levels [1, 3, 3] becomes [1, 2, 2]; [2, 2, 5]
        becomes [1, 1, 2].
        """
        sections = self._collect_sections(roots)
        if not sections:
            return
        level_map: Dict[int, int] = {}

        def normalized(level: int) -> int:
            if level not in level_map:
                level_map[level] = len(level_map) + 1
            return min(6, level_map[level])

        for section in sections:
            section.level = normalized(section.level)

    @staticmethod
    def _collect_sections(nodes: List[Node]) -> List[SectionNode]:
        found: List[SectionNode] = []
        for node in nodes:
            if isinstance(node, SectionNode):
                found.append(node)
                found.extend(MarkdownBuilder._collect_sections(node.children))
        return found

    # ------------------------------------------------------------------
    # Step 5: rendering
    # ------------------------------------------------------------------

    def _render_markdown(self, roots: List[Node]) -> tuple[str, Optional[str]]:
        """Render the node tree to Markdown; also return the extracted title.

        Blocks are joined with the configured paragraph separator, except that
        adjacent list blocks are joined tightly (single newline) so nested or
        split lists render as one continuous structure.
        """
        blocks: List[tuple[str, str]] = []  # (text, kind) where kind is "block"|"list"
        title: Optional[str] = None

        def emit(text: str, kind: str = "block") -> None:
            if text:
                blocks.append((text, kind))

        def render_nodes(nodes: List[Node]) -> None:
            nonlocal title
            for node in nodes:
                if isinstance(node, SectionNode):
                    if title is None:
                        title = node.title
                    emit(f"{'#' * min(6, node.level)} {node.title}")
                    render_nodes(node.children)
                elif isinstance(node, ParagraphNode):
                    emit(_escape_inline_markdown(node.text))
                elif isinstance(node, ListNode):
                    emit(self._render_list(node), kind=f"list:{node.list_type}")
                elif isinstance(node, TableNode):
                    emit(self._render_table(node))
                elif isinstance(node, ImageNode):
                    emit(self._render_image(node))
                elif isinstance(node, CodeBlockNode):
                    lang = node.language or ""
                    emit(f"```{lang}\n{node.text}\n```")
                elif isinstance(node, BlockquoteNode):
                    quoted = "\n".join(f"> {line}" for line in node.text.splitlines())
                    emit(quoted)
                elif isinstance(node, HorizontalRuleNode):
                    emit("---")
                elif isinstance(node, PageBreakNode):
                    emit(self.config.page_number_format.format(page=node.page).strip("\n"))
                else:  # pragma: no cover - exhaustive by construction
                    raise TypeError(f"Unhandled node type: {type(node)!r}")

        render_nodes(roots)

        pieces: List[str] = []
        prev_kind: Optional[str] = None
        for text, kind in blocks:
            if not text:
                continue
            tight = prev_kind is not None and prev_kind == kind and kind.startswith("list:")
            if prev_kind is not None:
                pieces.append("\n" if tight else self.config.paragraph_separator)
            pieces.append(text)
            prev_kind = kind

        content = "".join(pieces)
        return content + ("\n" if content else ""), title

    def _render_list(self, node: ListNode) -> str:
        lines: List[str] = []
        counters: Dict[int, int] = {}
        for item in node.items:
            indent = "  " * item["indent_level"]
            if node.list_type == "ordered":
                level = item["indent_level"]
                counters[level] = counters.get(level, 0) + 1
                marker = f"{counters[level]}. "
            else:
                marker = "- "
            lines.append(f"{indent}{marker}{_escape_inline_markdown(item['text'])}")
        return "\n".join(lines)

    def _render_table(self, node: TableNode) -> str:
        rows = [[_escape_table_cell(cell) for cell in row] for row in node.rows]
        if not rows:
            return ""
        width = max(len(r) for r in rows)
        rows = [r + [""] * (width - len(r)) for r in rows]

        lines = [
            "| " + " | ".join(rows[0]) + " |",
            "| " + " | ".join(["---"] * width) + " |",
        ]
        lines.extend("| " + " | ".join(r) + " |" for r in rows[1:])
        table = "\n".join(lines)

        caption = f"*{node.caption}*" if node.caption else None
        if caption:
            if self.config.table_caption_position == "after":
                return f"{table}\n{caption}"
            return f"{caption}\n{table}"
        return table

    def _render_image(self, node: ImageNode) -> str:
        alt = node.description or node.caption or "image"
        src = node.src or "image_placeholder"
        md = f"![{_escape_table_cell(alt)}]({src})"
        if node.caption:
            md += f"\n*{node.caption}*"
        return md
