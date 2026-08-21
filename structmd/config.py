"""Configuration system for structmd.

Configuration is merged from four layers with the following precedence
(highest wins):

1. Environment variables (``STRUCTMD_*``)
2. Project-local ``./.structmd.yaml``
3. Global ``~/.config/structmd/config.yaml``
4. Built-in defaults
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional, get_type_hints

import yaml

logger = logging.getLogger(__name__)

GLOBAL_CONFIG_PATH = "~/.config/structmd/config.yaml"
LOCAL_CONFIG_NAME = ".structmd.yaml"
ENV_PREFIX = "STRUCTMD_"


@dataclass
class StructMDConfig:
    """Top-level structmd configuration."""

    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2-vl:2b"
    ollama_timeout: int = 120
    ollama_max_workers: int = 4
    dpi: int = 150
    include_page_numbers: bool = True
    page_number_format: str = "\n<!-- Page {page} -->\n"
    merge_continued_paragraphs: bool = True
    detect_columns: bool = True
    normalize_headings: bool = True
    table_caption_position: str = "before"  # "before" or "after"
    cache_dir: str = "~/.cache/structmd"
    cache_enabled: bool = True
    verbose: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Return the config as a plain dict."""
        return asdict(self)


def _coerce(value: Any, target_type: type) -> Any:
    """Coerce a raw YAML/env value to the dataclass field's type."""
    if value is None:
        return None
    if target_type is bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    if target_type is int:
        return int(value)
    return str(value)


def _flatten_nested_yaml(data: Dict[str, Any]) -> Dict[str, Any]:
    """Map the documented nested YAML layout onto flat config keys.

    Supports::

        ollama: {url, model, timeout, max_workers}
        processing: {dpi, detect_columns, merge_continued_paragraphs,
                     normalize_headings}
        output: {include_page_numbers, page_number_format,
                 table_caption_position}
        cache: {dir}

    Flat keys are passed through untouched so both styles work.
    """
    section_map = {
        "ollama": {
            "url": "ollama_url",
            "model": "ollama_model",
            "timeout": "ollama_timeout",
            "max_workers": "ollama_max_workers",
        },
        "processing": {
            "dpi": "dpi",
            "detect_columns": "detect_columns",
            "merge_continued_paragraphs": "merge_continued_paragraphs",
            "normalize_headings": "normalize_headings",
        },
        "output": {
            "include_page_numbers": "include_page_numbers",
            "page_number_format": "page_number_format",
            "table_caption_position": "table_caption_position",
        },
        "cache": {"dir": "cache_dir"},
    }
    flat: Dict[str, Any] = {}
    for key, value in data.items():
        if key in section_map and isinstance(value, dict):
            for sub_key, sub_value in value.items():
                mapped = section_map[key].get(sub_key)
                if mapped is not None:
                    flat[mapped] = sub_value
                else:
                    logger.debug("Ignoring unknown key %r in section %r", sub_key, key)
        else:
            flat[key] = value
    return flat


def _load_yaml_layer(path: Path) -> Dict[str, Any]:
    """Load one YAML config layer, returning {} when missing/invalid."""
    expanded = path.expanduser()
    if not expanded.is_file():
        return {}
    try:
        with open(expanded, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:
        logger.warning("Could not parse config file %s: %s", expanded, exc)
        return {}
    if not isinstance(data, dict):
        logger.warning(
            "Config file %s must contain a mapping, got %s", expanded, type(data).__name__
        )
        return {}
    logger.debug("Loaded config layer: %s", expanded)
    return _flatten_nested_yaml(data)


def _field_types() -> Dict[str, Any]:
    """Resolve dataclass field annotations to actual types.

    ``f.type`` is a plain string when ``from __future__ import annotations``
    is active, so we resolve hints explicitly.
    """
    return dict(get_type_hints(StructMDConfig))


def _env_overrides() -> Dict[str, Any]:
    """Collect STRUCTMD_* environment variables matching config fields."""
    overrides: Dict[str, Any] = {}
    field_types = _field_types()
    for f in fields(StructMDConfig):
        env_name = f"{ENV_PREFIX}{f.name.upper()}"
        raw = os.environ.get(env_name)
        if raw is not None:
            overrides[f.name] = _coerce(raw, field_types[f.name])
    return overrides


def load_config(config_path: Optional[str] = None) -> StructMDConfig:
    """Build a :class:`StructMDConfig` by merging all configuration layers.

    Args:
        config_path: Optional explicit config file (e.g. from ``--config``).
            When given it replaces both the global and project-local layers;
            environment variables still take precedence over it.

    Returns:
        The fully merged configuration.
    """
    merged: Dict[str, Any] = {}

    if config_path:
        merged.update(_load_yaml_layer(Path(config_path)))
    else:
        merged.update(_load_yaml_layer(Path(GLOBAL_CONFIG_PATH)))
        merged.update(_load_yaml_layer(Path.cwd() / LOCAL_CONFIG_NAME))

    merged.update(_env_overrides())

    valid_names = {f.name for f in fields(StructMDConfig)}
    unknown = set(merged) - valid_names
    if unknown:
        logger.warning("Ignoring unknown config keys: %s", ", ".join(sorted(unknown)))

    kwargs = {k: v for k, v in merged.items() if k in valid_names}
    return StructMDConfig(**kwargs)
