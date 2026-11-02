"""Exercise offline benchmark wiring without a model or network."""

import json

import pytest

from benchmarks import run
from structmd.cache.manager import CacheManager
from structmd.config import StructMDConfig
from structmd.converters.pdf import fitz_module
from structmd.core import DocumentElement, ElementType, ExtractedDocument, ExtractedPage


def test_no_live_never_constructs_pipeline(tmp_path, monkeypatch):
    source = tmp_path / "paper.pdf"
    with fitz_module.open() as pdf:
        pdf.new_page().insert_text((50, 50), "Hello world")
        pdf.save(source)
    config = StructMDConfig(cache_dir=str(tmp_path / "cache"))
    monkeypatch.setattr(run, "load_config", lambda: config)

    def forbid_pipeline(*args, **kwargs):
        raise AssertionError("offline scoring must never initialize a model client")

    monkeypatch.setattr(run, "StructMDPipeline", forbid_pipeline)
    output = tmp_path / "report.json"
    args = ["--no-live", "--corpus", str(tmp_path), "-o", str(output)]
    assert run.main(args) == 2
    assert not output.exists()
    CacheManager(config.cache_dir).save_page(
        str(source),
        ExtractedPage(
            page_number=1,
            elements=[DocumentElement(type=ElementType.PARAGRAPH, text="Hello world")],
        ),
    )
    assert run.main(args) == 0
    report = json.loads(output.read_text())
    assert report["summary"]["content_f1"]["f1"] == 1.0
    assert report["meta"]["live"] is False


def test_score_document_handles_row_tables(monkeypatch):
    monkeypatch.setattr(run.gt, "extract_ground_truth", lambda *args: "name value alpha beta")
    doc = ExtractedDocument(
        page_count=1,
        pages=[
            ExtractedPage(
                page_number=1,
                elements=[
                    DocumentElement(
                        type=ElementType.TABLE, table_data=[["name", "value"], ["alpha", "beta"]]
                    )
                ],
            )
        ],
    )
    report = run.score_document("test", doc, "unused.pdf")
    assert report.content["f1"] == 1.0
    assert report.tables == {"count": 1, "rows": 2, "width_consistent": 1.0}


def test_scans_without_text_are_not_given_a_score(monkeypatch):
    monkeypatch.setattr(run.gt, "extract_ground_truth", lambda *args: "  ")
    with pytest.raises(ValueError, match="PDF text layer"):
        run.score_document(
            "scan", ExtractedDocument(pages=[ExtractedPage(page_number=1)]), "scan.pdf"
        )
