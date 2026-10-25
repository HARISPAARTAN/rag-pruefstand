"""LLM commands: cost estimate, generate, judge, and the synthetic CI cassette."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from pruefstand.answer import build_context
from pruefstand.bench import hits_path
from pruefstand.commands.common import DEFAULT_CONFIG_PATH, ConfigOption, ProfileOption, guarded, load
from pruefstand.corpus.gii import load_units
from pruefstand.generation import chunk_lookup, run_generation, run_judging, selected_questions
from pruefstand.io import read_jsonl
from pruefstand.llm.costs import BudgetGuard, load_prices, project_stages
from pruefstand.llm.replay import Cassette, ReplayClient
from pruefstand.llm.synthetic import SyntheticClient
from pruefstand.pipeline import stage_answer_metrics, stage_generate, stage_judge

cost_app = typer.Typer(no_args_is_help=True, help="Projected API spend (dry run, no calls).")
llm_app = typer.Typer(no_args_is_help=True, help="Synthetic CI cassette for the replay path.")
SYNTHETIC = Path("fixtures/llm/synthetic_responses.json")


@cost_app.command("estimate")
def estimate(profile: ProfileOption = "full", config: ConfigOption = DEFAULT_CONFIG_PATH) -> None:
    """Print the projected spend per stage from costs.toml and the [[cost.stage]] table."""
    cfg = load(profile, config)
    table = load_prices(cfg.path(cfg["llm"]["costs"]))
    projections = project_stages(cfg["cost"]["stage"], cfg["llm"], table)
    total = 0.0
    for p in projections:
        mode = "batch" if p.batch else "online"
        typer.echo(f"{p.name:24s} {p.model:20s} {p.calls:6d} calls {mode:6s} USD {p.usd:8.2f}")
        total += p.usd
    cap = float(cfg["budget"]["max_usd_per_run"])
    typer.echo(f"{'total':24s} USD {total:.2f} (EUR {total * table.eur_per_usd:.2f}); cap USD {cap:.2f}")
    typer.echo(f"prices as of {table.meta['as_of']} ({table.meta['source']})")
    if total > cap:
        typer.echo("projection exceeds the cap", err=True)
        raise typer.Exit(code=1)


def generate(profile: ProfileOption = "full", config: ConfigOption = DEFAULT_CONFIG_PATH) -> None:
    """Generate answers (PRUEFSTAND_LLM_MODE=record for a live run; default replay)."""
    cfg = load(profile, config)
    with guarded():
        records = stage_generate(cfg)
    typer.echo(f"{len(records)} answers, USD {sum(r['cost_usd'] for r in records):.4f} ({cfg.llm_mode()})")


def judge(profile: ProfileOption = "full", config: ConfigOption = DEFAULT_CONFIG_PATH) -> None:
    """Judge the answers (Batch API in a live run) and compute answer metrics."""
    cfg = load(profile, config)
    with guarded():
        judgements = stage_judge(cfg)
        stage_answer_metrics(cfg)
    typer.echo(
        f"{len(judgements)} judgements, USD {sum(j['cost_usd'] for j in judgements):.4f} ({cfg.llm_mode()})"
    )


@llm_app.command("synthetic-template")
def synthetic_template(profile: ProfileOption = "ci", config: ConfigOption = DEFAULT_CONFIG_PATH) -> None:
    """Write fixtures/llm/synthetic_responses.json with one empty answer and judgement per CI request."""
    cfg = load(profile, config)
    units_by_id = {unit.unit_id: unit for unit in load_units(cfg.path(cfg["corpus"]["units_path"]))}
    hits = {(row["config_id"], row["qid"]): row for row in read_jsonl(hits_path(cfg))}
    chunk_by_id = chunk_lookup(cfg)
    n_context = int(cfg["generation"]["context_chunks"])
    answers, judgements = {}, {}
    for config_id in cfg["generation"]["variants"]:
        for question in selected_questions(cfg):
            key = f"{config_id}|{question['qid']}"
            chunks = [
                chunk_by_id[h["chunk_id"]] for h in hits[(config_id, question["qid"])]["ranked"][:n_context]
            ]
            answers[key] = {
                "question_de": question["question_de"],
                "gold_answer_de": question["gold_answer_de"],
                "context_preview": build_context(chunks, units_by_id)[:600],
                "text": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "latency_ms": 0,
                "stop_reason": "end_turn",
            }
            judgements[key] = {
                "faithfulness": 0,
                "correctness": 0,
                "completeness": 0,
                "rationale": "",
                "input_tokens": 0,
                "output_tokens": 0,
            }
    payload = {
        "note": "SYNTHETIC: hand-written by the implementing agent, not model output. CI replay fixture only.",
        "answers": answers,
        "judgements": judgements,
    }
    target = cfg.path(SYNTHETIC)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    typer.echo(f"wrote {len(answers)} answer and {len(judgements)} judgement slots to {SYNTHETIC}")


@llm_app.command("record-synthetic")
def record_synthetic(profile: ProfileOption = "ci", config: ConfigOption = DEFAULT_CONFIG_PATH) -> None:
    """Record the synthetic responses into the cassette (marked synthetic) via the normal record path."""
    cfg = load(profile, config)
    responses = json.loads(cfg.path(SYNTHETIC).read_text(encoding="utf-8"))
    cassette = Cassette(cfg.path(cfg["llm"]["cassette"]))
    client = ReplayClient("record", cassette, SyntheticClient(responses), synthetic=True)
    guard = BudgetGuard(float(cfg["budget"]["max_usd_per_run"]))
    with guarded():
        run_generation(cfg, client, guard)
        run_judging(cfg, client, guard)
    typer.echo(f"cassette {cfg['llm']['cassette']} now holds {len(cassette.entries)} entries")


def register(app: typer.Typer) -> None:
    app.add_typer(cost_app, name="cost")
    app.add_typer(llm_app, name="llm")
    app.command("generate")(generate)
    app.command("judge")(judge)
