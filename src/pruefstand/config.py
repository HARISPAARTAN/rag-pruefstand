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
# Settings that decide what the retrieval run produces; llm, cost, budget, generation, paths and the question
# metadata (authored, author_model) are left out on purpose, so changing them never invalidates a published run.
RETRIEVAL_SECTIONS = (
    "corpus",
    "tokenizer",
    "chunking",
    "retrieval",
    "models",
    "stats",
    "latency",
    "comparison",
    "headline",
)


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

    @property
    def source_run_id(self) -> str:
        """The run whose retrieval artefacts (hits, chunks, manifest) this profile reads; its own by default."""
        return self.data["paths"].get("source_run_id") or self.run_id

    @property
    def source_dir(self) -> Path:
        return self.path(self.data["paths"]["runs_dir"]) / self.source_run_id

    @property
    def reads_foreign_run(self) -> bool:
        return self.source_run_id != self.run_id

    def llm_mode(self) -> str:
        mode = os.environ.get("PRUEFSTAND_LLM_MODE") or self.data["llm"]["mode"]
        if mode not in LLM_MODES:
            raise ValueError(f"PRUEFSTAND_LLM_MODE must be one of {LLM_MODES}, got {mode!r}")
        return mode

    def config_hash(self) -> str:
        return sha256_text(canonical_json(self.data))

    def retrieval_config(self) -> dict:
        subset = {name: self.data[name] for name in RETRIEVAL_SECTIONS}
        subset["questions"] = {key: self.data["questions"][key] for key in ("path", "version")}
        return subset

    def retrieval_config_hash(self) -> str:
        """Hash of the settings that decide the retrieval results (see RETRIEVAL_SECTIONS)."""
        return sha256_text(canonical_json(self.retrieval_config()))


def load_config(path: str | Path = DEFAULT_CONFIG, profile: str = "full") -> Config:
    path = Path(path).resolve()
    with open(path, "rb") as handle:
        raw = tomllib.load(handle)
    profiles = raw.pop("profile", {})
    if profile not in profiles:
        raise ValueError(f"unknown profile {profile!r}; known: {sorted(profiles)}")
    data = deep_merge(raw, profiles[profile])
    return Config(data=data, profile=profile, root=path.parent)
