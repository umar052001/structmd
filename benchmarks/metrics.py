"""Extraction-quality metrics for the benchmark harness.

Every metric is deterministic and dependency-free (stdlib only) so the
harness can run anywhere, including CI, against collected ground truth.

These are lightweight regression proxies based on what reference text is
available without manual annotation, not an implementation of a named benchmark:

- content F1:      word-level precision/recall/F1 of extracted text vs the
                   PDF's native text layer (born-digital documents only)
- reading order:   SequenceMatcher ratio between extracted and reference word
                   streams, with autojunk disabled
- heading grounding: share of distinct headings absent from the normalized
                     source text (not a heading-detection accuracy score)
- table health:    share of tables whose data rows match the header width
- figure helper:   broken asset link detection (not figure detection accuracy)

Ground truth comes from the PDFs' embedded text layer, so scoring scans
(e.g. photographed pages) is out of scope for now.
"""

from __future__ import annotations

import difflib
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase and collapse whitespace; the basis of all text comparisons."""
    return _WS.sub(" ", text.lower()).strip()


def token_counts(text: str) -> Counter:
    return Counter(normalize(text).split())


def content_f1(extracted_text: str, ground_truth: str) -> Tuple[float, float, float]:
    """Word-level precision / recall / F1 of extracted vs ground truth."""
    truth = token_counts(ground_truth)
    found = token_counts(extracted_text)
    overlap = sum((truth & found).values())
    total_found = sum(found.values())
    total_truth = sum(truth.values())
    precision = overlap / total_found if total_found else 1.0
    recall = overlap / total_truth if total_truth else 1.0
    f1 = 0.0 if precision + recall == 0.0 else 2 * precision * recall / (precision + recall)
    return round(precision, 4), round(recall, 4), round(f1, 4)


def reading_order_similarity(extracted_text: str, ground_truth: str) -> float:
    """0..1 similarity of the word streams, order-sensitive."""
    left = normalize(extracted_text).split()
    right = normalize(ground_truth).split()
    return round(difflib.SequenceMatcher(None, left, right, autojunk=False).ratio(), 4)


def heading_text_absence_rate(headings: Sequence[str], ground_truth: str) -> float:
    """Share of distinct extracted headings absent from the source text layer.

    This is a text-grounding proxy, not proof of hallucination or correct
    heading classification. Whitespace and PDF line wrapping are ignored.
    """
    truth = f" {normalize(ground_truth)} "
    distinct = {normalize(h) for h in headings if normalize(h)}
    if not distinct:
        return 0.0
    missing = sum(1 for heading in distinct if f" {heading} " not in truth)
    return round(missing / len(distinct), 4)


def table_width_consistency(
    headers: Optional[Sequence[str]], rows: Optional[Sequence[Sequence[str]]]
) -> Optional[bool]:
    """True when every row has as many cells as the header (or longest row).

    Returns None when the table carries no usable data.
    """
    if not rows:
        return None
    width = len(headers) if headers else max(len(r) for r in rows)
    if width == 0:
        return None
    return all(len(row) == width for row in rows)


def aggregate(values: Sequence[float]) -> float:
    """Mean of a metric series."""
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def merge_f1(f1s: Sequence[Tuple[float, float, float]]) -> Dict[str, float]:
    """Average (precision, recall, f1) tuples into a single dict."""
    if not f1s:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    precision = aggregate([p for p, _r, _f in f1s])
    recall = aggregate([r for _p, r, _f in f1s])
    f1 = aggregate([f for _p, _r, f in f1s])
    return {"precision": precision, "recall": recall, "f1": f1}


def broken_figure_links(asset_paths: Iterable[str], markdown_root: str) -> List[str]:
    """Assets whose files do not exist under ``markdown_root``."""
    from pathlib import Path

    broken: List[str] = []
    for asset in asset_paths:
        if not (Path(markdown_root) / asset).is_file():
            broken.append(asset)
    return broken
