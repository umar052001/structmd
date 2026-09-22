"""Ground truth extraction from the PDF text layer.

Born-digital PDFs embed their textual content in a text layer, which gives
us an objective reference for content preservation metrics. This module
reads that layer with PyMuPDF in document order; scanned documents have no
usable layer and are out of scope for the harness.
"""

from __future__ import annotations

from typing import List, Optional

from structmd.converters.pdf import fitz_module

PROSE_TYPES = {"paragraph", "heading", "list_item", "caption"}


def extract_ground_truth(pdf_path: str, pages: Optional[List[int]] = None) -> str:
    """Return the text layer of ``pdf_path``, pages in order.

    ``pages`` is 1-indexed like the rest of structmd.
    """
    chunks: List[str] = []
    with fitz_module.open(pdf_path) as doc:
        numbers = pages or list(range(1, doc.page_count + 1))
        for number in numbers:
            if not 1 <= number <= doc.page_count:
                continue
            chunks.append(doc[number - 1].get_text("text"))
    return "\n".join(chunks)


def extracted_text_pages(document) -> List[str]:
    """Join each extracted page's prose into one text blob, in order."""
    blobs: List[str] = []
    for page in document.pages:
        parts = []
        for element in page.elements:
            if element.type.value in PROSE_TYPES and element.text:
                parts.append(element.text)
            elif element.type.value == "table" and element.table_data:
                td = element.table_data
                if isinstance(td, dict):
                    headers = td.get("headers") or []
                    rows = td.get("rows") or []
                    parts.append(" ".join(headers))
                    parts.append(" ".join(" ".join(cells) for cells in rows))
                else:
                    for row in td:
                        parts.append(" ".join(str(c) for c in row))
        blobs.append("\n".join(parts))
    return blobs
