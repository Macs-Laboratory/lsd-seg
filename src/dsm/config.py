from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass(slots=True)
class Config:
    raw: dict[str, Any]

    def __getitem__(self, item: str) -> Any:
        return self.raw[item]

    def get(self, item: str, default: Any = None) -> Any:
        return self.raw.get(item, default)

    def section(self, item: str) -> dict[str, Any]:
        return dict(self.raw[item])


def load_config(config_path: str | Path, defaults_path: str | Path | None = None) -> Config:
    config_path = Path(config_path)
    defaults = Path(defaults_path) if defaults_path else config_path.parent / "default.yaml"
    base = yaml.safe_load(defaults.read_text()) if defaults.exists() else {}
    override = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
    return Config(raw=_deep_update(base, override))
