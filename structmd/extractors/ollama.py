"""Ollama-backed VLM layout extractor.

Talks to the Ollama HTTP API (``/api/chat``) and converts page images into
:class:`~structmd.core.ExtractedPage` JSON. Designed for small (2B-3B)
vision-language models: tolerant JSON parsing, retries with exponential
backoff, and per-model-family prompt templates.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

import httpx
from PIL import Image

from structmd.config import StructMDConfig
from structmd.core import (
    BoundingBox,
    DocumentElement,
    ElementType,
    ExtractedDocument,
    ExtractedPage,
    ModelNotFoundError,
    OllamaConnectionError,
)
from structmd.extractors.base import BaseExtractor

logger = logging.getLogger(__name__)

MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# Prompt templates, one per supported model family.
# Each must instruct the model to return ONLY a JSON object.
# ---------------------------------------------------------------------------

_JSON_CONTRACT = (
    "{\n"
    '  "elements": [\n'
    "    {\n"
    '      "type": "heading|paragraph|table|list_item|page_number|caption|footnote|header|'  # noqa: E501
    'footer|image|code_block|blockquote|horizontal_rule",\n'
    '      "bbox": [x1, y1, x2, y2],\n'
    '      "text": "extracted text",\n'
    '      "level": 1,\n'
    '      "list_type": "ordered|unordered",\n'
    '      "indent_level": 0,\n'
    '      "continues": false,\n'
    '      "confidence": 0.95,\n'
    '      "table_data": [["cell", "..."]]\n'
    "    }\n"
    "  ]\n"
    "}"
)

PROMPTS: Dict[str, str] = {
    "qwen2-vl": (
        "You are a precise document layout analysis engine. Analyze the document page image "
        "and extract every content block in reading order.\n\n"
        f"Return ONLY a valid JSON object, no prose, no markdown fences, exactly matching:\n"
        f"{_JSON_CONTRACT}\n\n"
        "Rules:\n"
        "- bbox is [x1, y1, x2, y2] in pixels of this image, origin at top-left.\n"
        "- 'level' (1-6) only for headings; 'list_type' ('ordered'|'unordered') and "
        "'indent_level' only for list_item.\n"
        "- For tables put the full text in 'text' AND rows of cell strings in 'table_data'.\n"
        "- Set 'continues': true when a paragraph visibly continues on the next page.\n"
        "- 'confidence' is your certainty from 0 to 1.\n"
        "- Output raw JSON only."
    ),
    "smolvlm": (
        "Extract all text blocks from this document page as JSON.\n"
        f"JSON shape:\n{_JSON_CONTRACT}\n"
        "Types: heading, paragraph, table, list_item, page_number, caption, footnote, header, "
        "footer, image, code_block, blockquote, horizontal_rule.\n"
        "bbox = [x1,y1,x2,y2] pixels. Reply with JSON only."
    ),
    "paligemma": (
        "Document page to structured JSON.\n"
        f"Schema:\n{_JSON_CONTRACT}\n"
        "Coordinates in pixels, top-left origin. JSON output only."
    ),
    "llama3.2-vision": (
        "You are a document digitization assistant. Look at this scanned page and transcribe "
        "its structure as JSON.\n\n"
        f"Respond with ONLY this JSON structure:\n{_JSON_CONTRACT}\n\n"
        "Guidelines:\n"
        "- Use reading order (top-to-bottom, left-to-right).\n"
        "- bbox coordinates are pixel values [x1, y1, x2, y2].\n"
        "- Headings need 'level' 1-6. Lists need 'list_type' and 'indent_level'.\n"
        "- Tables need both 'text' and 'table_data' (rows of cells).\n"
        '- Mark paragraphs that continue on the next page with "continues": true.\n'
        "- Do not add commentary. Output JSON only."
    ),
}

PROMPTS["default"] = (
    "Analyze this document page image and extract its layout structure.\n\n"
    f"Return ONLY a valid JSON object with this exact structure:\n{_JSON_CONTRACT}\n\n"
    "Field rules:\n"
    "- type: one of heading, paragraph, table, list_item, page_number, caption, footnote, "
    "header, footer, image, code_block, blockquote, horizontal_rule\n"
    "- bbox: [x1, y1, x2, y2] pixel coordinates, top-left origin\n"
    "- level: heading depth 1-6 (headings only)\n"
    "- list_type: 'ordered' or 'unordered', indent_level: nesting depth (list items only)\n"
    "- table_data: rows of cell strings (tables only); also summarize in 'text'\n"
    "- continues: true if a paragraph continues on the next page\n"
    "- confidence: 0.0-1.0 certainty estimate\n"
    "Output raw JSON only — no explanations, no markdown code fences."
)

# Synonyms small VLMs commonly emit instead of canonical type names.
_TYPE_ALIASES: Dict[str, ElementType] = {
    "title": ElementType.HEADING,
    "section_header": ElementType.HEADING,
    "h1": ElementType.HEADING,
    "h2": ElementType.HEADING,
    "h3": ElementType.HEADING,
    "p": ElementType.PARAGRAPH,
    "text": ElementType.PARAGRAPH,
    "body": ElementType.PARAGRAPH,
    "bullet": ElementType.LIST_ITEM,
    "bullets": ElementType.LIST_ITEM,
    "list": ElementType.LIST_ITEM,
    "ordered_list": ElementType.LIST_ITEM,
    "unordered_list": ElementType.LIST_ITEM,
    "quote": ElementType.BLOCKQUOTE,
    "code": ElementType.CODE_BLOCK,
    "preformatted": ElementType.CODE_BLOCK,
    "hr": ElementType.HORIZONTAL_RULE,
    "rule": ElementType.HORIZONTAL_RULE,
    "divider": ElementType.HORIZONTAL_RULE,
    "img": ElementType.IMAGE,
    "picture": ElementType.IMAGE,
    "figure": ElementType.IMAGE,
    "page_num": ElementType.PAGE_NUMBER,
    "folio": ElementType.PAGE_NUMBER,
}


def detect_model_family(model: str) -> str:
    """Map an Ollama model name onto a prompt family key."""
    name = model.lower()
    if "smolvlm" in name or "smol" in name:
        return "smolvlm"
    if "paligemma" in name:
        return "paligemma"
    if "llama3.2" in name and "vision" in name:
        return "llama3.2-vision"
    if "qwen" in name and ("vl" in name or "vision" in name):
        return "qwen2-vl"
    return "default"


def extract_json_payload(content: str) -> Optional[Dict[str, Any]]:
    """Best-effort extraction of a JSON object from raw VLM output.

    Strategy (in order):
      1. Direct ``json.loads``.
      2. Markdown fenced block (```json ... ```).
      3. Balanced-brace scan + repair of truncated JSON.

    Returns None when nothing parseable can be recovered.
    """
    if not content or not content.strip():
        return None

    # 1) Direct parse.
    try:
        parsed = json.loads(content.strip())
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # 2) Fenced ```json ... ``` block.
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if fence_match:
        try:
            parsed = json.loads(fence_match.group(1))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # 3) Balanced-brace scan from the first '{', then truncation repair.
    start = content.find("{")
    if start == -1:
        return None
    blob = _scan_balanced(content, start)
    if not blob:
        return None
    try:
        parsed = json.loads(blob)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    repaired = _repair_truncated(blob)
    if repaired is not None:
        try:
            parsed = json.loads(repaired)
            if isinstance(parsed, dict):
                logger.warning("Recovered truncated JSON via repair")
                return parsed
        except json.JSONDecodeError:
            pass
    return None


def _scan_balanced(text: str, start: int) -> str:
    """Scan from ``start`` collecting a balanced {...} region.

    Falls back to the rest of the string when braces never balance (truncated
    output), which :func:`_repair_truncated` then tries to fix.
    """
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


def _repair_truncated(blob: str) -> Optional[str]:
    """Repair truncated JSON by trimming back to a completable prefix.

    Small VLMs frequently run out of tokens mid-object. We scan the delimiter
    state of progressively shorter prefixes and accept the longest one that,
    once its open strings/brackets/braces are closed, parses as valid JSON.
    """
    max_backtrack = min(len(blob), 400)
    for cut in range(len(blob), len(blob) - max_backtrack - 1, -1):
        closed = _close_open_delimiters(blob[:cut])
        if closed is None:
            continue
        try:
            json.loads(closed)
        except json.JSONDecodeError:
            continue
        return closed
    return None


def _close_open_delimiters(prefix: str) -> Optional[str]:
    """Close all open strings/brackets/braces in ``prefix``; None if impossible."""
    stack: List[str] = []
    in_string = False
    escape = False
    for ch in prefix:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                return None  # unbalanced beyond repair
            stack.pop()
    if in_string:
        return None  # cannot safely terminate an unterminated string here
    closers = "".join("}" if c == "{" else "]" for c in reversed(stack))
    repaired = prefix + closers
    # Drop trailing commas left dangling before generated closers.
    return re.sub(r",\s*([}\]])", r"\1", repaired)


class OllamaExtractor(BaseExtractor):
    """Extracts structured page JSON using a VLM served by Ollama."""

    def __init__(self, config: StructMDConfig) -> None:
        self.config = config
        self._sync_client: Optional[httpx.Client] = None
        self._async_client: Optional[httpx.AsyncClient] = None
        self._model_checked = False

    # ------------------------------------------------------------------
    # HTTP client lifecycle
    # ------------------------------------------------------------------

    def _get_sync_client(self) -> httpx.Client:
        if self._sync_client is None:
            self._sync_client = httpx.Client(timeout=self.config.ollama_timeout)
        return self._sync_client

    def _get_async_client(self) -> httpx.AsyncClient:
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(timeout=self.config.ollama_timeout)
        return self._async_client

    def close(self) -> None:
        """Release underlying HTTP clients."""
        if self._sync_client is not None:
            self._sync_client.close()
            self._sync_client = None
        if self._async_client is not None:
            try:
                loop = asyncio.get_event_loop_policy().get_event_loop()
                if loop.is_running():
                    loop.create_task(self._async_client.aclose())
                else:
                    loop.run_until_complete(self._async_client.aclose())
            except RuntimeError:
                pass  # No usable loop; client will be GC'd.
            self._async_client = None

    # ------------------------------------------------------------------
    # Model availability
    # ------------------------------------------------------------------

    def list_models(self) -> List[str]:
        """Return names of models available on the Ollama server."""
        try:
            resp = self._get_sync_client().get(f"{self.config.ollama_url}/api/tags")
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaConnectionError(
                f"Could not reach Ollama at {self.config.ollama_url}: {exc}. "
                "Is the server running? Start it with `ollama serve`."
            ) from exc
        data = resp.json()
        return [m.get("name", "") for m in data.get("models", [])]

    def resolve_model(self, available: List[str]) -> str:
        """Normalize the configured model against available tags.

        Appends ``:latest`` when the tag is missing and raises
        :class:`ModelNotFoundError` with a pull hint when absent entirely.
        """
        requested = self.config.ollama_model.strip()
        base_names = {m.split(":")[0]: m for m in available}
        if ":" not in requested:
            if requested in base_names:
                resolved = base_names[requested]
                logger.debug("Resolved model %r -> %r", requested, resolved)
                return resolved
            requested = f"{requested}:latest"
        if requested not in available:
            raise ModelNotFoundError(
                f"Model {requested!r} is not available on {self.config.ollama_url}. "
                f"Available: {available or '(none)'}. "
                f"Install it with: ollama pull {requested}"
            )
        return requested

    def ensure_model_available(self) -> None:
        """Check once per extractor lifetime that the model exists."""
        if self._model_checked:
            return
        available = self.list_models()
        resolved = self.resolve_model(available)
        if resolved != self.config.ollama_model:
            self.config.ollama_model = resolved
        self._model_checked = True

    async def ensure_model_available_async(self) -> None:
        if self._model_checked:
            return
        try:
            resp = await self._get_async_client().get(f"{self.config.ollama_url}/api/tags")
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaConnectionError(
                f"Could not reach Ollama at {self.config.ollama_url}: {exc}. "
                "Is the server running? Start it with `ollama serve`."
            ) from exc
        available = [m.get("name", "") for m in resp.json().get("models", [])]
        resolved = self.resolve_model(available)
        if resolved != self.config.ollama_model:
            self.config.ollama_model = resolved
        self._model_checked = True

    # ------------------------------------------------------------------
    # Prompting
    # ------------------------------------------------------------------

    def build_prompt(self) -> str:
        """Return the prompt for the configured model's family."""
        family = detect_model_family(self.config.ollama_model)
        prompt = PROMPTS.get(family, PROMPTS["default"])
        logger.debug("Using %r family prompt for model %r", family, self.config.ollama_model)
        return prompt

    @staticmethod
    def encode_image(image: Image.Image) -> str:
        """Base64-encode a PIL image as PNG for the Ollama ``images`` field."""
        buffer = io.BytesIO()
        rgb = image if image.mode == "RGB" else image.convert("RGB")
        rgb.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def parse_element(raw: Dict[str, Any], page_number: int) -> Optional[DocumentElement]:
        """Convert one raw VLM element dict into a :class:`DocumentElement`.

        Returns None for elements that cannot be salvaged at all.
        """
        raw_type = (
            str(raw.get("type", "paragraph")).strip().lower().replace("-", "_").replace(" ", "_")
        )
        element_type = _TYPE_ALIASES.get(raw_type)
        if element_type is None:
            try:
                element_type = ElementType(raw_type)
            except ValueError:
                logger.debug(
                    "Unknown element type %r on page %d; treating as paragraph",
                    raw_type,
                    page_number,
                )
                element_type = ElementType.PARAGRAPH

        text = str(raw.get("text", "") or "")
        if not text.strip() and element_type != ElementType.IMAGE:
            return None

        bbox: Optional[BoundingBox] = None
        raw_bbox = raw.get("bbox")
        if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
            try:
                coords = [float(c) for c in raw_bbox]
                if all(c == c for c in coords):  # reject NaN
                    bbox = BoundingBox(*coords)
            except (TypeError, ValueError):
                bbox = None
        elif isinstance(raw_bbox, dict):
            try:
                bbox = BoundingBox.from_dict(raw_bbox)
            except (KeyError, TypeError, ValueError):
                bbox = None

        confidence = raw.get("confidence", 1.0)
        try:
            confidence = min(1.0, max(0.0, float(confidence)))
        except (TypeError, ValueError):
            confidence = 1.0

        heading_level = None
        if element_type == ElementType.HEADING:
            try:
                heading_level = int(raw.get("level", 1))
                heading_level = min(6, max(1, heading_level))
            except (TypeError, ValueError):
                heading_level = 1

        list_type = None
        indent_level = 0
        if element_type == ElementType.LIST_ITEM:
            lt = str(raw.get("list_type", "unordered")).lower()
            list_type = "ordered" if "ord" in lt else "unordered"
            try:
                indent_level = max(0, int(raw.get("indent_level", 0)))
            except (TypeError, ValueError):
                indent_level = 0

        continues = bool(raw.get("continues", False))

        table_data = None
        if element_type == ElementType.TABLE:
            maybe_rows = raw.get("table_data")
            if isinstance(maybe_rows, list):
                table_data = [
                    [str(cell) for cell in row] for row in maybe_rows if isinstance(row, list)
                ]
            if not table_data:
                table_data = _parse_table_from_text(text)

        return DocumentElement(
            type=element_type,
            bbox=bbox,
            text=text,
            page_number=page_number,
            confidence=confidence,
            heading_level=heading_level,
            list_type=list_type,
            indent_level=indent_level,
            continues_on_next_page=continues,
            table_data=table_data,
        )

    def parse_response(self, content: str, page_number: int) -> List[DocumentElement]:
        """Parse raw chat content into a list of elements.

        Emits a warning and returns [] when parsing fails entirely.
        """
        payload = extract_json_payload(content)
        if payload is None:
            logger.warning(
                "Page %d: could not parse any JSON from model response; returning empty page",
                page_number,
            )
            logger.debug("Raw response was: %.500r", content)
            return []

        elements: List[DocumentElement] = []
        for raw in payload.get("elements", []) or []:
            if not isinstance(raw, dict):
                continue
            element = self.parse_element(raw, page_number)
            if element is not None:
                elements.append(element)
        return elements

    # ------------------------------------------------------------------
    # Chat completion
    # ------------------------------------------------------------------

    def _chat_payload(self, image_b64: str) -> Dict[str, Any]:
        return {
            "model": self.config.ollama_model,
            "messages": [
                {
                    "role": "user",
                    "content": self.build_prompt(),
                    "images": [image_b64],
                }
            ],
            "stream": False,
        }

    def _call_chat_sync(self, image_b64: str) -> str:
        """POST /api/chat with retries and exponential backoff."""
        payload = self._chat_payload(image_b64)
        url = f"{self.config.ollama_url}/api/chat"
        last_error: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self._get_sync_client().post(url, json=payload)
                if resp.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"Ollama server error {resp.status_code}",
                        request=resp.request,
                        response=resp,
                    )
                resp.raise_for_status()
                data = resp.json()
                content = data.get("message", {}).get("content", "")
                logger.debug("Raw VLM response (page): %.500r", content)
                return content
            except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as exc:
                last_error = exc
                if attempt < MAX_RETRIES:
                    delay = 2 ** (attempt - 1)
                    logger.warning(
                        "Ollama call failed (attempt %d/%d): %s — retrying in %ds",
                        attempt,
                        MAX_RETRIES,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
        raise OllamaConnectionError(
            f"Ollama request failed after {MAX_RETRIES} attempts: {last_error}"
        ) from last_error

    async def _call_chat_async(self, image_b64: str) -> str:
        payload = self._chat_payload(image_b64)
        url = f"{self.config.ollama_url}/api/chat"
        last_error: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = await self._get_async_client().post(url, json=payload)
                if resp.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"Ollama server error {resp.status_code}",
                        request=resp.request,
                        response=resp,
                    )
                resp.raise_for_status()
                data = resp.json()
                content = data.get("message", {}).get("content", "")
                logger.debug("Raw VLM response (page): %.500r", content)
                return content
            except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as exc:
                last_error = exc
                if attempt < MAX_RETRIES:
                    delay = 2 ** (attempt - 1)
                    logger.warning(
                        "Ollama call failed (attempt %d/%d): %s — retrying in %ds",
                        attempt,
                        MAX_RETRIES,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)
        raise OllamaConnectionError(
            f"Ollama request failed after {MAX_RETRIES} attempts: {last_error}"
        ) from last_error

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_page(self, image: Image.Image, page_number: int) -> ExtractedPage:
        """Extract one page synchronously."""
        self.ensure_model_available()
        image_b64 = self.encode_image(image)
        content = self._call_chat_sync(image_b64)
        elements = self.parse_response(content, page_number)
        return ExtractedPage(
            page_number=page_number,
            width=float(image.width),
            height=float(image.height),
            elements=elements,
        )

    def extract_document(
        self, images: List[Image.Image], source_path: Optional[str] = None
    ) -> ExtractedDocument:
        """Extract all pages sequentially (sync)."""
        pages = [self.extract_page(img, idx + 1) for idx, img in enumerate(images)]
        return ExtractedDocument(
            source_path=source_path,
            page_count=len(pages),
            pages=pages,
            metadata={"extractor": "ollama", "model": self.config.ollama_model},
        )

    async def extract_page_async(self, image: Image.Image, page_number: int) -> ExtractedPage:
        """Extract one page asynchronously."""
        await self.ensure_model_available_async()
        image_b64 = self.encode_image(image)
        content = await self._call_chat_async(image_b64)
        elements = self.parse_response(content, page_number)
        return ExtractedPage(
            page_number=page_number,
            width=float(image.width),
            height=float(image.height),
            elements=elements,
        )

    async def extract_document_async(
        self, images: List[Image.Image], source_path: Optional[str] = None
    ) -> ExtractedDocument:
        """Extract all pages sequentially (async)."""
        pages = [await self.extract_page_async(img, idx + 1) for idx, img in enumerate(images)]
        return ExtractedDocument(
            source_path=source_path,
            page_count=len(pages),
            pages=pages,
            metadata={"extractor": "ollama", "model": self.config.ollama_model},
        )


def _parse_table_from_text(text: str) -> Optional[List[List[str]]]:
    """Fallback: derive table rows from pipe-separated text like ``a | b | c``."""
    rows: List[List[str]] = []
    for line in text.splitlines():
        line = line.strip().strip("|")
        if not line:
            continue
        cells = [c.strip() for c in line.split("|")]
        if any(cells):
            rows.append(cells)
    return rows or None
