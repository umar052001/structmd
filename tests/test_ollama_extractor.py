"""Tests for the Ollama extractor: parsing, prompts, retries, model checks.

All HTTP traffic is mocked with respx — no real Ollama server needed.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import httpx
import pytest
import respx
from PIL import Image

from structmd.config import StructMDConfig
from structmd.core import ElementType, ModelNotFoundError, OllamaConnectionError
from structmd.extractors.ollama import (
    OllamaExtractor,
    detect_model_family,
    extract_json_payload,
)

OLLAMA_BASE = "http://localhost:11434"


def make_config(**overrides: Any) -> StructMDConfig:
    defaults: Dict[str, Any] = {"ollama_url": OLLAMA_BASE, "ollama_model": "qwen2-vl:2b"}
    defaults.update(overrides)
    return StructMDConfig(**defaults)


def chat_response(content: str) -> Dict[str, Any]:
    return {
        "model": "qwen2-vl:2b",
        "message": {"role": "assistant", "content": content},
        "done": True,
    }


def tags_response(models: List[str]) -> Dict[str, Any]:
    return {"models": [{"name": m} for m in models]}


VALID_JSON = json.dumps(
    {
        "elements": [
            {
                "type": "heading",
                "bbox": [10, 10, 300, 40],
                "text": "Introduction",
                "level": 1,
                "confidence": 0.98,
            },
            {
                "type": "paragraph",
                "bbox": [10, 60, 580, 120],
                "text": "Body paragraph.",
                "confidence": 0.95,
            },
            {
                "type": "table",
                "bbox": [10, 140, 580, 300],
                "text": "Name | Qty",
                "table_data": [["Name", "Qty"], ["Widget", "3"]],
                "confidence": 0.9,
            },
        ]
    }
)


class TestModelFamilyDetection:
    @pytest.mark.parametrize(
        "model,expected",
        [
            ("qwen2-vl:2b", "qwen2-vl"),
            ("qwen2.5-vl:7b", "qwen2-vl"),
            ("smolvlm:2b", "smolvlm"),
            ("paligemma:3b", "paligemma"),
            ("llama3.2-vision:11b", "llama3.2-vision"),
            ("minicpm-v:8b", "default"),
        ],
    )
    def test_family(self, model: str, expected: str) -> None:
        assert detect_model_family(model) == expected


class TestJSONParsing:
    def test_direct_json(self) -> None:
        payload = extract_json_payload(VALID_JSON)
        assert payload is not None
        assert len(payload["elements"]) == 3

    def test_markdown_fenced(self) -> None:
        wrapped = f"Here is the extraction:\n```json\n{VALID_JSON}\n```\nDone!"
        payload = extract_json_payload(wrapped)
        assert payload is not None
        assert payload["elements"][0]["text"] == "Introduction"

    def test_prose_around_bare_json(self) -> None:
        payload = extract_json_payload(f"Sure! {VALID_JSON} hope that helps")
        assert payload is not None

    def test_truncated_json_repair(self) -> None:
        truncated = VALID_JSON[: len(VALID_JSON) - 60]  # cut off the tail
        payload = extract_json_payload(truncated)
        assert payload is not None
        assert isinstance(payload.get("elements"), list)
        assert len(payload["elements"]) >= 1

    def test_garbage_returns_none(self) -> None:
        assert extract_json_payload("no json here at all") is None
        assert extract_json_payload("") is None


class TestParseElement:
    def setup_method(self) -> None:
        self.extractor = OllamaExtractor(make_config())

    def test_valid_element(self) -> None:
        el = self.extractor.parse_element(
            {"type": "heading", "bbox": [1, 2, 3, 4], "text": "Hi", "level": 2, "confidence": 0.7},
            page_number=1,
        )
        assert el is not None
        assert el.type == ElementType.HEADING
        assert el.heading_level == 2
        assert el.confidence == pytest.approx(0.7)
        assert el.bbox is not None and el.bbox.to_tuple() == (1.0, 2.0, 3.0, 4.0)

    def test_alias_title_maps_to_heading(self) -> None:
        el = self.extractor.parse_element({"type": "title", "text": "T"}, 1)
        assert el.type == ElementType.HEADING

    def test_unknown_type_falls_back_to_paragraph(self) -> None:
        el = self.extractor.parse_element({"type": "marginalia", "text": "note"}, 1)
        assert el.type == ElementType.PARAGRAPH

    def test_empty_text_dropped(self) -> None:
        assert self.extractor.parse_element({"type": "paragraph", "text": "   "}, 1) is None

    def test_list_item_fields(self) -> None:
        el = self.extractor.parse_element(
            {"type": "list_item", "text": "item", "list_type": "ordered", "indent_level": 2}, 1
        )
        assert el.list_type == "ordered"
        assert el.indent_level == 2

    def test_confidence_clamped(self) -> None:
        el = self.extractor.parse_element({"type": "paragraph", "text": "x", "confidence": 5}, 1)
        assert el.confidence == 1.0

    def test_table_without_table_data_parses_pipes(self) -> None:
        el = self.extractor.parse_element({"type": "table", "text": "A | B\n1 | 2"}, 1)
        assert el.table_data == [["A", "B"], ["1", "2"]]

    def test_continues_flag(self) -> None:
        el = self.extractor.parse_element({"type": "paragraph", "text": "x", "continues": True}, 1)
        assert el.continues_on_next_page is True


class TestParseResponse:
    def test_full_response(self) -> None:
        extractor = OllamaExtractor(make_config())
        elements = extractor.parse_response(VALID_JSON, page_number=3)
        assert [e.page_number for e in elements] == [3, 3, 3]
        assert elements[0].type == ElementType.HEADING
        assert elements[2].table_data == [["Name", "Qty"], ["Widget", "3"]]

    def test_garbage_yields_empty_with_warning(self, caplog) -> None:
        extractor = OllamaExtractor(make_config())
        with caplog.at_level("WARNING"):
            elements = extractor.parse_response("complete nonsense", page_number=1)
        assert elements == []
        assert any("could not parse" in r.message.lower() for r in caplog.records)


@respx.mock
class TestOllamaHTTP:
    def test_extract_page_happy_path(self) -> None:
        respx.get(f"{OLLAMA_BASE}/api/tags").mock(
            return_value=httpx.Response(200, json=tags_response(["qwen2-vl:2b"]))
        )
        route = respx.post(f"{OLLAMA_BASE}/api/chat").mock(
            return_value=httpx.Response(200, json=chat_response(VALID_JSON))
        )
        extractor = OllamaExtractor(make_config())
        image = Image.new("RGB", (612, 792), "white")
        page = extractor.extract_page(image, 1)

        assert page.page_number == 1
        assert page.width == 612
        assert len(page.elements) == 3
        # Request must carry base64 image and stream:false.
        sent = route.calls.last.request.read()
        body = json.loads(sent)
        assert body["stream"] is False
        assert body["model"] == "qwen2-vl:2b"
        assert body["messages"][0]["images"]

    def test_missing_model_raises_pull_hint(self) -> None:
        respx.get(f"{OLLAMA_BASE}/api/tags").mock(
            return_value=httpx.Response(200, json=tags_response(["gemma4:latest"]))
        )
        extractor = OllamaExtractor(make_config())
        with pytest.raises(ModelNotFoundError, match="ollama pull qwen2-vl:2b"):
            extractor.ensure_model_available()

    def test_auto_appends_latest_tag(self) -> None:
        respx.get(f"{OLLAMA_BASE}/api/tags").mock(
            return_value=httpx.Response(200, json=tags_response(["qwen2-vl:latest"]))
        )
        cfg = make_config(ollama_model="qwen2-vl")
        extractor = OllamaExtractor(cfg)
        extractor.ensure_model_available()
        assert cfg.ollama_model == "qwen2-vl:latest"

    def test_unreachable_server_raises_connection_error(self) -> None:
        respx.get(f"{OLLAMA_BASE}/api/tags").mock(side_effect=httpx.ConnectError("refused"))
        extractor = OllamaExtractor(make_config())
        with pytest.raises(OllamaConnectionError, match="Could not reach Ollama"):
            extractor.list_models()

    def test_retry_on_500_then_success(self, monkeypatch) -> None:
        monkeypatch.setattr("structmd.extractors.ollama.time.sleep", lambda s: None)
        respx.get(f"{OLLAMA_BASE}/api/tags").mock(
            return_value=httpx.Response(200, json=tags_response(["qwen2-vl:2b"]))
        )
        respx.post(f"{OLLAMA_BASE}/api/chat").mock(
            side_effect=[
                httpx.Response(500, text="boom"),
                httpx.Response(500, text="boom"),
                httpx.Response(200, json=chat_response(VALID_JSON)),
            ]
        )
        extractor = OllamaExtractor(make_config())
        page = extractor.extract_page(Image.new("RGB", (100, 100)), 1)
        assert len(page.elements) == 3

    def test_exhausted_retries_raise(self, monkeypatch) -> None:
        monkeypatch.setattr("structmd.extractors.ollama.time.sleep", lambda s: None)
        respx.get(f"{OLLAMA_BASE}/api/tags").mock(
            return_value=httpx.Response(200, json=tags_response(["qwen2-vl:2b"]))
        )
        respx.post(f"{OLLAMA_BASE}/api/chat").mock(return_value=httpx.Response(500, text="boom"))
        extractor = OllamaExtractor(make_config())
        with pytest.raises(OllamaConnectionError, match="after 3 attempts"):
            extractor.extract_page(Image.new("RGB", (100, 100)), 1)


@pytest.mark.asyncio
@respx.mock
class TestOllamaAsync:
    async def test_extract_page_async(self) -> None:
        respx.get(f"{OLLAMA_BASE}/api/tags").mock(
            return_value=httpx.Response(200, json=tags_response(["qwen2-vl:2b"]))
        )
        respx.post(f"{OLLAMA_BASE}/api/chat").mock(
            return_value=httpx.Response(200, json=chat_response(VALID_JSON))
        )
        extractor = OllamaExtractor(make_config())
        page = await extractor.extract_page_async(Image.new("RGB", (300, 400)), 2)
        assert page.page_number == 2
        assert len(page.elements) == 3

    async def test_extract_document_async(self) -> None:
        respx.get(f"{OLLAMA_BASE}/api/tags").mock(
            return_value=httpx.Response(200, json=tags_response(["qwen2-vl:2b"]))
        )
        respx.post(f"{OLLAMA_BASE}/api/chat").mock(
            return_value=httpx.Response(200, json=chat_response(VALID_JSON))
        )
        extractor = OllamaExtractor(make_config())
        doc = await extractor.extract_document_async(
            [Image.new("RGB", (50, 50)) for _ in range(2)], source_path="x.pdf"
        )
        assert doc.page_count == 2
        assert doc.metadata["extractor"] == "ollama"
