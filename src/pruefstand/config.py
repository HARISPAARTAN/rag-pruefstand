"""Load pruefstand.toml and apply a named profile on top of the base settings."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pruefstand.io import canonical_json, sha256_text

DEFAULT_CONFIG = "pruefstand.toml"
LLM_MODES = ("record", "replay", "live")


def deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict: tables merge recursively, every other value in override wins."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass(frozen=True)
class Config:
    data: dict
    profile: str
    root: Path

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def path(self, rel: str | Path) -> Path:
        candidate = Path(rel)
        return candidate if candidate.is_absolute() else self.root / candidate

    @property
    def run_dir(self) -> Path:
        paths = self.data["paths"]
        return self.path(paths["runs_dir"]) / paths["run_id"]

    @property
    def run_id(self) -> str:
        return self.data["paths"]["run_id"]

    def llm_mode(self) -> str:
        mode = os.environ.get("PRUEFSTAND_LLM_MODE") or self.data["llm"]["mode"]
        if mode not in LLM_MODES:
            raise ValueError(f"PRUEFSTAND_LLM_MODE must be one of {LLM_MODES}, got {mode!r}")
        return mode

    def config_hash(self) -> str:
        return sha256_text(canonical_json(self.data))


def load_config(path: str | Path = DEFAULT_CONFIG, profile: str = "full") -> Config:
    path = Path(path).resolve()
    with open(path, "rb") as handle:
        raw = tomllib.load(handle)
    profiles = raw.pop("profile", {})
    if profile not in profiles:
        raise ValueError(f"unknown profile {profile!r}; known: {sorted(profiles)}")
    data = deep_merge(raw, profiles[profile])
    return Config(data=data, profile=profile, root=path.parent)
