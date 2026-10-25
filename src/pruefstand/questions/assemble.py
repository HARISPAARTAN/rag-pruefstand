"""Merge authored batch files into the frozen question set with ids, provenance and filter results."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from pruefstand.corpus.gii import Unit
from pruefstand.io import read_jsonl
from pruefstand.prompts import load_prompt
from pruefstand.questions.filters import run_filters
from pruefstand.questions.schema import AUTHORED_FIELDS, check_distribution, law_of, validate_authored


def assemble(
    batch_paths: list[Path],
    units_by_id: Mapping[str, Unit],
    author_model: str,
    created_at: str,
    protocol_path: Path,
    check_targets: bool = True,
) -> tuple[list[dict], list[str]]:
    records: list[dict] = []
    origins: list[str] = []
    for path in batch_paths:
        for line_no, record in enumerate(read_jsonl(path), start=1):
            records.append(record)
            origins.append(f"{path.name}:{line_no}")
    errors: list[str] = []
    for origin, record in zip(origins, records, strict=True):
        errors.extend(f"{origin}: {problem}" for problem in validate_authored(record, units_by_id))
    if errors:
        return [], errors
    if check_targets:
        errors.extend(check_distribution(records))
    results, filter_errors = run_filters(records, units_by_id)
    errors.extend(f"{origins[index]}: {problem}" for index, problem in filter_errors)
    if errors:
        return [], errors
    protocol = load_prompt(protocol_path)
    questions = []
    for index, (record, result) in enumerate(zip(records, results, strict=True), start=1):
        question = {"qid": f"k1v1-{index:04d}", "corpus": "K1", "law": law_of(record)}
        question.update({name: record[name] for name in AUTHORED_FIELDS})
        question["answerable"] = record["type"] != "unanswerable"
        question["provenance"] = {
            "method": "agent-authored",
            "model": author_model,
            "prompt_id": protocol.prompt_id,
            "prompt_sha256": protocol.sha256,
            "source_unit_ids": record["source_unit_ids"],
            "created_at": created_at,
            "filters": result,
        }
        question["review"] = {"status": "pending", "reviewed_at": None, "note": ""}
        questions.append(question)
    return questions, []
