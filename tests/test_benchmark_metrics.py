"""Unit tests for the benchmark metrics module (offline, no model needed)."""

from __future__ import annotations

from benchmarks import metrics


def test_normalize_collapses_whitespace_and_case() -> None:
    assert metrics.normalize("  Deep   Residual\tNETWORKS\n") == "deep residual networks"


def test_content_f1_perfect_match() -> None:
    text = "the quick brown fox jumps over the lazy dog"
    assert metrics.content_f1(text, text) == (1.0, 1.0, 1.0)


def test_content_f1_penalizes_missing_words() -> None:
    truth = "alpha beta gamma delta"
    extracted = "alpha beta"
    precision, recall, f1 = metrics.content_f1(extracted, truth)
    assert precision == 1.0
    assert recall == 0.5
    assert 0.66 < f1 < 0.67


def test_content_f1_empty_truth_does_not_crash() -> None:
    assert metrics.content_f1("", "") == (1.0, 1.0, 1.0)
    assert metrics.content_f1("something", "")[1] == 1.0  # nothing missing


def test_reading_order_similarity_penalizes_reordering() -> None:
    ordered = "one two three four five"
    same_string = "one two three four five"
    ish = "four five three one two"
    assert metrics.reading_order_similarity(ordered, same_string) == 1.0
    assert metrics.reading_order_similarity(ordered, ish) < 1.0


def test_heading_hallucination_flags_invented_heads() -> None:
    truth = "Introduction\nDeep Residual Networks\nConclusion\nsome body text"
    headings = ["Introduction", "Deep Residual Networks", "Made Up Section"]
    rate = metrics.heading_hallucination_rate(headings, truth)
    assert rate > 0.33 and rate < 0.34


def test_heading_hallucination_ignores_case_and_whitespace() -> None:
    truth = "3.1 Identity Mappings in Deep Residual Networks"
    headings = ["  3.1 identity mappings in deep residual networks "]
    assert metrics.heading_hallucination_rate(headings, truth) == 0.0


def test_table_width_consistency() -> None:
    assert metrics.table_width_consistency(["a", "b"], [["1", "2"], ["3", "4"]]) is True
    assert metrics.table_width_consistency(["a", "b"], [["1", "2"], ["3"]]) is False
    assert metrics.table_width_consistency(None, []) is None


def test_aggregate_and_merge_f1() -> None:
    assert metrics.aggregate([1.0, 0.0, 0.5]) == 0.5
    merged = metrics.merge_f1([(1.0, 1.0, 1.0), (0.5, 0.5, 0.5)])
    assert merged["f1"] == 0.75


def test_broken_figure_links(tmp_path) -> None:
    (tmp_path / "figures").mkdir()
    (tmp_path / "figures" / "ok.png").write_bytes(b"png")
    broken = metrics.broken_figure_links(["figures/ok.png", "figures/missing.png"], str(tmp_path))
    assert broken == ["figures/missing.png"]
