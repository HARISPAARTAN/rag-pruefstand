"""Versioned prompt files: the id is the file stem, the hash is SHA-256 over the LF-normalised text."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pruefstand.io import read_text_lf, sha256_text


@dataclass(frozen=True)
class Prompt:
    prompt_id: str
    text: str
    sha256: str


def load_prompt(path: str | Path) -> Prompt:
    text = read_text_lf(path).strip() + "\n"
    return Prompt(prompt_id=Path(path).stem, text=text, sha256=sha256_text(text))
