"""Run manifest: everything needed to say which code, data, models and machine produced a run."""

from __future__ import annotations

import platform
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from pruefstand.config import Config, load_config
from pruefstand.errors import PruefstandError
from pruefstand.io import canonical_json, read_json, sha256_file, sha256_text, write_json
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
RUNTIME_CONFIG = "config.toml"  # the configuration the published run ran with, committed next to its manifest
STAGE_MANIFESTS = "stage_manifests.json"
STAGE_ROLES = {"generate": "generator", "judge": "judge"}


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
    prompt_paths = {
        cfg["llm"][role][key]
        for role in LLM_ROLES
        for key in ("prompt", "faithfulness_prompt")
        if key in cfg["llm"][role]
    }
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
        "retrieval_config_sha256": cfg.retrieval_config_hash(),
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


def stage_config(cfg: Config, stage: str) -> dict:
    """The settings a paid stage depends on: its role, the generation selection, the cassette and the batch limits."""
    llm = cfg["llm"]
    return {
        "role": llm[STAGE_ROLES[stage]],
        "generation": cfg["generation"],
        "cassette": llm["cassette"],
        "batch": {key: llm[key] for key in ("batch_poll_seconds", "batch_max_wait_seconds") if key in llm},
        "questions": {key: cfg["questions"][key] for key in ("path", "version")},
    }


def _prompt_hashes(cfg: Config, role: str) -> dict[str, str]:
    hashes = {}
    for key in ("prompt", "faithfulness_prompt"):
        path = cfg["llm"][role].get(key)
        if path and cfg.path(path).exists():
            prompt = load_prompt(cfg.path(path))
            hashes[prompt.prompt_id] = prompt.sha256
    return dict(sorted(hashes.items()))


def published_retrieval_hash(cfg: Config) -> tuple[str, Config | None]:
    """The retrieval hash of the run this profile reads, from its manifest.

    Manifests written since v0.1.1 carry `retrieval_config_sha256`. The v0.1.0 manifest only has the hash of the
    whole configuration, so the run-time configuration is committed next to it (config.toml); it is accepted only
    when its hash matches the manifest. Returns the hash and that run-time configuration (None when absent).
    """
    manifest_path = cfg.source_dir / "manifest.json"
    if not manifest_path.is_file():
        raise PruefstandError(f"{manifest_path} not found: a paid stage needs the published retrieval run")
    manifest = read_json(manifest_path)
    runtime_path = cfg.source_dir / RUNTIME_CONFIG
    runtime = None
    if runtime_path.is_file():
        runtime = load_config(runtime_path, manifest["profile"])
        if runtime.config_hash() != manifest["config_sha256"]:
            raise PruefstandError(
                f"{runtime_path} (hash {runtime.config_hash()[:12]}) does not belong to {manifest_path} "
                f"(hash {manifest['config_sha256'][:12]})"
            )
    if manifest.get("retrieval_config_sha256"):
        return manifest["retrieval_config_sha256"], runtime
    if runtime is None:
        raise PruefstandError(
            f"{manifest_path} has no retrieval_config_sha256 and {runtime_path} is missing: "
            "the retrieval settings of the published run cannot be established"
        )
    return runtime.retrieval_config_hash(), runtime


def check_retrieval_provenance(cfg: Config) -> str:
    """Refuse a paid stage when the retrieval settings differ from those of the published run."""
    expected, runtime = published_retrieval_hash(cfg)
    current = cfg.retrieval_config_hash()
    if current != expected:
        changed = ""
        if runtime is not None:
            now, then = cfg.retrieval_config(), runtime.retrieval_config()
            diff = sorted(key for key in now.keys() | then.keys() if now.get(key) != then.get(key))
            changed = f" (changed sections: {', '.join(diff)})"
        raise PruefstandError(
            f"the retrieval settings hash {current[:12]} differs from the published run "
            f"{cfg.source_run_id} ({expected[:12]}){changed}; paid results would not belong to those hits. "
            "Revert the retrieval settings or publish a new retrieval run first."
        )
    return current


def write_stage_manifest(
    cfg: Config, stage: str, started_at: str, records: list[dict] | None, spend_usd: float
) -> dict:
    """Append one provenance entry for a paid stage (generate or judge) to the run's stage_manifests.json.

    The retrieval manifest is not touched. Entries are written for record and live runs only; replay changes nothing.
    """
    role = STAGE_ROLES[stage]
    cassette_path = cfg.path(cfg["llm"]["cassette"])
    entry = {
        "stage": stage,
        "mode": cfg.llm_mode(),
        "profile": cfg.profile,
        "completed": records is not None,
        "git_sha": git_sha(cfg.root),
        "config_sha256": cfg.config_hash(),
        "retrieval_config_sha256": cfg.retrieval_config_hash(),
        "stage_config_sha256": sha256_text(canonical_json(stage_config(cfg, stage))),
        "retrieval_run": cfg.source_run_id,
        "model": cfg["llm"][role]["model"],
        "prompts": _prompt_hashes(cfg, role),
        "cassette": cfg["llm"]["cassette"],
        "cassette_sha256": sha256_file(cassette_path) if cassette_path.is_file() else None,
        "question_set": {
            "version": cfg["questions"]["version"],
            "sha256": sha256_file(cfg.path(cfg["questions"]["path"])),
        },
        "anthropic_sdk": package_versions()["anthropic"],
        "n_records": None if records is None else len(records),
        "n_billed_now": None if records is None else sum(1 for r in records if r.get("billed_now")),
        "spend_usd": round(spend_usd, 6),
        "started_at": started_at,
        "finished_at": now_iso(),
    }
    path = cfg.run_dir / STAGE_MANIFESTS
    entries = read_json(path) if path.exists() else {}
    entries.setdefault(stage, []).append(entry)
    write_json(path, entries)
    return entry
