"""File helpers: JSONL (plain or gzip), canonical JSON and SHA-256 hashing."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def canonical_json(obj: Any) -> str:
    """Stable JSON text: sorted keys, no whitespace, UTF-8 characters kept."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_text_lf(path: str | Path) -> str:
    """Read UTF-8 text and normalise Windows line endings, so hashes match on every OS."""
    return Path(path).read_text(encoding="utf-8").replace("\r\n", "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    """Write rows as JSON lines; .gz files get mtime 0 so identical content gives identical bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows]
    data = "".join(lines).encode("utf-8")
    if path.suffix == ".gz":
        buffer = io.BytesIO()
        with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as handle:
            handle.write(data)
        path.write_bytes(buffer.getvalue())
    else:
        path.write_bytes(data)
    return len(lines)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    path.write_bytes(text.encode("utf-8"))
