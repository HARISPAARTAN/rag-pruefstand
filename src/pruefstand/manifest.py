"""Run manifest: everything needed to say which code, data, models and machine produced a run."""

from __future__ import annotations

import platform
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from pruefstand.config import Config
from pruefstand.io import read_json, sha256_file, write_json
from pruefstand.prompts import load_prompt

PACKAGES = (
    "pruefstand",
    "numpy",
    "bm25s",
    "PyStemmer",
    "tokenizers",
    "anthropic",
    "sentence-transformers",
    "torch",
    "transformers",
)
LLM_ROLES = ("generator", "ablation", "judge", "second_judge", "local")


def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def git_sha(root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def package_versions() -> dict[str, str]:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def write_manifest(cfg: Config, started_at: str, stages: list[str], spend_usd: float = 0.0) -> dict:
    snapshot = read_json(cfg.path(cfg["corpus"]["snapshot_path"]))
    prompt_paths = {cfg["llm"][role]["prompt"] for role in LLM_ROLES}
    prompts = {}
    for path in sorted(prompt_paths):
        if cfg.path(path).exists():
            prompt = load_prompt(cfg.path(path))
            prompts[prompt.prompt_id] = prompt.sha256
    questions_path = cfg.path(cfg["questions"]["path"])
    manifest = {
        "run_id": cfg.run_id,
        "profile": cfg.profile,
        "stages": stages,
        "git_sha": git_sha(cfg.root),
        "config_sha256": cfg.config_hash(),
        "prompts": prompts,
        "corpus": {
            "fetched_at": snapshot["fetched_at"],
            "laws": {slug: entry["sha256"] for slug, entry in snapshot["laws"].items()},
            "units_sha256": sha256_file(cfg.path(cfg["corpus"]["units_path"])),
        },
        "question_set": {"version": cfg["questions"]["version"], "sha256": sha256_file(questions_path)},
        "models": {name: {"id": m["id"], "revision": m["revision"]} for name, m in cfg["models"].items()},
        "llm": {role: cfg["llm"][role]["model"] for role in LLM_ROLES},
        "packages": package_versions(),
        "machine": {
            "os": platform.platform(),
            "cpu": platform.processor(),
            "python": platform.python_version(),
        },
        "started_at": started_at,
        "finished_at": now_iso(),
        "spend_usd": round(spend_usd, 4),
    }
    write_json(cfg.run_dir / "manifest.json", manifest)
    return manifest
