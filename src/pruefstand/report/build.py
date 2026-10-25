"""Build reports (Markdown DE/EN, JSON, SVG charts) and the static site from committed run artefacts only.

No key and no network: every number comes from runs/<run_id>/*.json. Missing generation results render an
explicit pending state instead of numbers.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

from pruefstand.bench import config_ids
from pruefstand.config import Config
from pruefstand.io import read_json, write_json
from pruefstand.report.charts import RETRIEVERS, latency_svg, recall_at_k_svg, retrieval_matrix_svg
from pruefstand.report.text import TEXT, fmt, fmt_ci

REPO_URL = "https://github.com/HARISPAARTAN/rag-pruefstand"
README_START = "<!-- results:start -->"
README_END = "<!-- results:end -->"
METRIC_COLUMNS = ("recall@1", "recall@3", "recall@5", "recall@10", "mrr@10", "ndcg@10")


def _optional(path: Path) -> dict | None:
    return read_json(path) if path.exists() else None


def _comparison(metrics: dict, a: str, b: str, metric: str = "recall@5") -> dict | None:
    for entry in metrics["comparisons"]:
        if entry["a"] == a and entry["b"] == b and entry["metric"] == metric:
            return entry
    return None


def _diff_text(template: str, entry: dict, lang: str) -> str:
    return template.format(
        a=entry["a"],
        b=entry["b"],
        diff=fmt(entry["diff"], lang, signed=True),
        lo=fmt(entry["lo"], lang, signed=True),
        hi=fmt(entry["hi"], lang, signed=True),
        verdict=TEXT[lang]["verdicts"][entry["verdict"]],
    )


def headline_lines(metrics: dict, lang: str) -> list[str]:
    """The three headline numbers (also the README block): the pre-declared configuration and two comparisons."""
    t = TEXT[lang]
    head = metrics["headline"]
    best = head["best"]
    lines = [
        t["headline_best"].format(
            config=best["config_id"],
            mean=fmt(best["mean"], lang),
            lo=fmt(best["lo"], lang),
            hi=fmt(best["hi"], lang),
            n=best["n"],
        )
    ]
    lines.append(_diff_text(t["headline_dense"], head["dense_vs_english"], lang))
    lines.append(_diff_text(t["headline_diff"], head["stack_vs_lexical"], lang))
    return lines


def finding_lines(metrics: dict, lang: str) -> list[str]:
    """Statements derived from the pre-declared comparisons; each is only made when the numbers carry it."""
    t = TEXT[lang]
    lines: list[str] = []
    hybrid = _comparison(metrics, "hyb-C2", "e5-C2")
    if hybrid is not None and hybrid["verdict"] == "no_measurable_difference":
        lines.append(_diff_text(t["finding_hybrid_none"], hybrid, lang))
    pairs = [_comparison(metrics, "e5+rr-C2", "e5-C2"), _comparison(metrics, "hyb+rr-C2", "hyb-C2")]
    if all(pair is not None and pair["verdict"] == "better" for pair in pairs):
        texts = [_diff_text(t["finding_rerank_pair"], pair, lang) for pair in pairs]
        lines.append(t["finding_rerank"].format(pairs="; ".join(texts)))
    best = metrics["headline"]["best"]
    highest = best.get("highest_point_estimate")
    if highest and highest["config_id"] != best["config_id"]:
        lines.append(
            t["finding_highest"].format(
                config=highest["config_id"],
                mean=fmt(highest["mean"], lang),
                lo=fmt(highest["lo"], lang),
                hi=fmt(highest["hi"], lang),
            )
        )
    return lines


def question_note(metrics: dict, meta: dict, lang: str) -> str:
    t = TEXT[lang]
    types = metrics["question_types"]
    answerable = sorted(
        ((kind, n) for kind, n in types.items() if kind != "unanswerable"), key=lambda i: (-i[1], i[0])
    )
    text = t["questions_note"].format(
        version=metrics["question_set"],
        answerable=metrics["n_answerable"],
        checked=metrics["n_checked"],
        spread=", ".join(f"{kind} {n}" for kind, n in answerable),
        unanswerable=types.get("unanswerable", 0),
    )
    if meta.get("authored"):
        text += t["questions_authored"].format(
            authored=meta["authored"], author=meta.get("author_model", "?")
        )
    return text


def _percent(value: float, lang: str) -> str:
    return fmt(value * 100, lang, 1)


def limit_lines(metrics: dict, meta: dict, lang: str) -> list[str]:
    """Platform and truncation facts of the run, rendered from the artefacts."""
    t = TEXT[lang]
    lines: list[str] = []
    machine = meta.get("machine")
    if machine:
        lines.append(
            t["platform_line"].format(
                os=machine["os"], cpu=machine["cpu"] or "unknown", python=machine["python"]
            )
        )
    truncation = metrics.get("truncation") or {}
    try:
        c1, c2 = truncation["C1"], truncation["C2"]
        lines.append(
            t["mini_truncation"].format(
                max_len=c1["mini"]["max_seq_length"],
                c1_rate=_percent(c1["mini"]["truncation_rate"], lang),
                c1_vis=_percent(c1["mini"]["mean_visible_share"], lang),
                c2_rate=_percent(c2["mini"]["truncation_rate"], lang),
                c2_vis=_percent(c2["mini"]["mean_visible_share"], lang),
                e5_c1=_percent(c1["e5"]["truncation_rate"], lang),
                e5_c2=_percent(c2["e5"]["truncation_rate"], lang),
            )
        )
        lines.append(
            t["reranker_truncation"].format(
                max_len=c1["reranker"]["max_length"],
                c1_rate=_percent(c1["reranker"]["truncation_rate"], lang),
                c2_rate=_percent(c2["reranker"]["truncation_rate"], lang),
            )
        )
    except (KeyError, TypeError):
        pass  # profiles without model files (ci) record no truncation rates
    return lines


def build_tables(metrics: dict, latency: dict | None, answers: dict | None, lang: str, meta: dict) -> dict:
    """Every table of the report as (headers, rows); Markdown and HTML render the same data, in config order."""
    t = TEXT[lang]
    variants = metrics["variants"]
    chunkings = meta["chunkings"]
    order = [cid for cid in meta["config_ids"] if cid in variants]
    tables = {
        "matrix": (
            [t["retriever"], *chunkings],
            [
                [r, *[fmt_ci(variants[f"{r}-{c}"]["metrics"]["recall@5"], lang) for c in chunkings]]
                for r in RETRIEVERS
            ],
        ),
        "all_metrics": (
            [t["variant"], *METRIC_COLUMNS],
            [[cid, *[fmt_ci(variants[cid]["metrics"][m], lang) for m in METRIC_COLUMNS]] for cid in order],
        ),
        "comparisons": (
            ["A", "B", t["metric"], t["diff"], t["ci"], t["verdict"]],
            [
                [
                    c["a"],
                    c["b"],
                    c["metric"],
                    fmt(c["diff"], lang, signed=True),
                    f"[{fmt(c['lo'], lang, signed=True)}, {fmt(c['hi'], lang, signed=True)}]",
                    t["verdicts"][c["verdict"]],
                ]
                for c in metrics["comparisons"]
            ],
        ),
    }
    types = sorted(metrics["question_types"])
    answerable_types = [kind for kind in types if kind != "unanswerable"]
    tables["by_type"] = (
        [t["variant"], *answerable_types],
        [
            [
                cid,
                *[
                    fmt_ci(variants[cid]["by_type"].get(kind, {}).get("recall@5", {}), lang)
                    for kind in answerable_types
                ],
            ]
            for cid in order
        ],
    )
    if latency:
        rows = []
        for cid in [c for c in order if c in latency["configs"]]:
            stages = latency["configs"][cid]
            rows.append(
                [
                    cid,
                    *[
                        f"{fmt(stages[s]['p50'], lang, 1)} / {fmt(stages[s]['p95'], lang, 1)}"
                        for s in ("embed", "search", "rerank", "total")
                    ],
                ]
            )
        tables["latency"] = ([t["variant"], t["embed"], t["search"], t["rerank"], t["total"]], rows)
    if answers:
        rows = []
        for cid in [c for c in meta["generation_variants"] if c in answers["variants"]]:
            v = answers["variants"][cid]
            rows.append(
                [
                    cid,
                    fmt_ci(v["faithfulness"], lang),
                    fmt_ci(v["correctness"], lang),
                    fmt_ci(v["citation_precision"], lang),
                    fmt_ci(v["abstention_accuracy"], lang),
                    fmt(v["usd_per_1000"], lang, 4),
                    fmt(v["eur_per_1000"], lang, 4),
                    fmt(v["latency_ms"]["p50"], lang, 0),
                ]
            )
        tables["generation"] = (t["gen_cols"], rows)
    return tables


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _method_points(metrics: dict, meta: dict, lang: str) -> list[str]:
    t = TEXT[lang]
    points = [point.format(boot=metrics["n_bootstrap"], seed=metrics["seed"]) for point in t["method_points"]]
    return points + limit_lines(metrics, meta, lang)


def render_markdown(metrics: dict, latency: dict | None, answers: dict | None, meta: dict, lang: str) -> str:
    t = TEXT[lang]
    tables = build_tables(metrics, latency, answers, lang, meta)
    corpus = metrics["corpus"]
    parts = [
        f"# {t['title']} ({t['run']} {metrics['run_id']})",
        "",
        "> " + question_note(metrics, meta, lang),
        "> "
        + t["corpus_note"].format(
            laws=len(corpus["units_per_law"]), units=corpus["n_units"], date=meta["snapshot_date"]
        ),
        "",
        f"## {t['summary']}",
        "",
        *[f"- {line}" for line in [*headline_lines(metrics, lang), *finding_lines(metrics, lang)]],
        "",
        f"## {t['matrix']}",
        "",
        md_table(*tables["matrix"]),
        "",
        f"![{t['chart_matrix_title']}](charts/retrieval_matrix.svg)",
        "",
        f"![{t['chart_k_title']}](charts/recall_at_k.svg)",
        "",
        f"## {t['all_metrics']}",
        "",
        md_table(*tables["all_metrics"]),
        "",
        f"## {t['comparisons']}",
        "",
        md_table(*tables["comparisons"]),
        "",
        f"## {t['by_type']}",
        "",
        md_table(*tables["by_type"]),
        "",
        f"## {t['latency']}",
        "",
    ]
    if latency:
        parts += [
            t["latency_note"].format(n=latency["n_questions"], cpu=latency["machine"]["cpu"] or "unknown"),
            "",
            md_table(*tables["latency"]),
            "",
            f"![{t['chart_latency_title']}](charts/latency.svg)",
        ]
    else:
        parts.append(t["latency_missing"])
    parts += ["", f"## {t['generation']}", ""]
    if answers is None:
        parts.append(t["pending_live"])
    else:
        if answers["synthetic"]:
            parts += [t["synthetic_banner"], ""]
        parts.append(md_table(*tables["generation"]))
    parts += ["", f"## {t['calibration']}", "", t["calibration_pending"], "", f"## {t['method']}", ""]
    parts += [f"- {point}" for point in _method_points(metrics, meta, lang)]
    parts += ["", f"{t['repo']}: {REPO_URL}", ""]
    return "\n".join(parts)


def html_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f'<div class="table"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _md_inline(text: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html.escape(text))


SITE_CSS = """
:root{--bg:#0b0f14;--panel:#121820;--text:#e6e8eb;--muted:#9aa4b2;--line:#222a33;--accent:#f2a93b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);
font:16px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:980px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:1.9rem;margin:0 0 8px}h2{font-size:1.25rem;margin:40px 0 12px;color:var(--accent)}
.label{font:12px/1.4 ui-monospace,SFMono-Regular,Consolas,monospace;color:var(--muted);letter-spacing:.04em;text-transform:uppercase}
.note{color:var(--muted);font-size:.95rem}.headline{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:16px 20px}
.headline li{margin:6px 0}.table{overflow-x:auto;border:1px solid var(--line);border-radius:8px}
table{border-collapse:collapse;width:100%;font:13px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace}
th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}th{color:var(--muted);font-weight:600}
figure{margin:16px 0;overflow-x:auto}figure svg{width:100%;min-width:520px;height:auto;display:block}.pending{border-left:3px solid var(--accent);padding:8px 14px;background:var(--panel)}
a{color:var(--accent)}footer{margin-top:48px;color:var(--muted);font-size:.9rem}
"""


def render_html(
    metrics: dict, latency: dict | None, answers: dict | None, meta: dict, svgs: dict[str, str], lang: str
) -> str:
    t = TEXT[lang]
    tables = build_tables(metrics, latency, answers, lang, meta)
    corpus = metrics["corpus"]
    other = "en.html" if lang == "de" else "index.html"
    summary = [*headline_lines(metrics, lang), *finding_lines(metrics, lang)]
    sections = [
        f'<p class="label">{t["run"]} {html.escape(metrics["run_id"])} · <a href="{other}">{t["other_lang"]}</a></p>',
        f"<h1>{html.escape(t['title'])}</h1>",
        f'<p class="note">{html.escape(question_note(metrics, meta, lang))}</p>',
        '<p class="note">'
        + html.escape(
            t["corpus_note"].format(
                laws=len(corpus["units_per_law"]), units=corpus["n_units"], date=meta["snapshot_date"]
            )
        )
        + "</p>",
        f"<h2>{t['summary']}</h2>",
        '<ul class="headline">' + "".join(f"<li>{html.escape(line)}</li>" for line in summary) + "</ul>",
        f"<h2>{t['matrix']}</h2>",
        f"<figure>{svgs['retrieval_matrix']}</figure>",
        html_table(*tables["matrix"]),
        f"<figure>{svgs['recall_at_k']}</figure>",
        f"<h2>{t['comparisons']}</h2>",
        html_table(*tables["comparisons"]),
        f"<h2>{t['by_type']}</h2>",
        html_table(*tables["by_type"]),
        f"<h2>{t['latency']}</h2>",
    ]
    if latency:
        sections += [
            f'<p class="note">{html.escape(t["latency_note"].format(n=latency["n_questions"], cpu=latency["machine"]["cpu"] or "unknown"))}</p>',
            f"<figure>{svgs['latency']}</figure>",
            html_table(*tables["latency"]),
        ]
    else:
        sections.append(f'<p class="note">{t["latency_missing"]}</p>')
    sections.append(f"<h2>{t['generation']}</h2>")
    if answers is None:
        sections.append(f'<p class="pending">{_md_inline(t["pending_live"])}</p>')
    else:
        if answers["synthetic"]:
            sections.append(f'<p class="pending">{_md_inline(t["synthetic_banner"])}</p>')
        sections.append(html_table(*tables["generation"]))
    sections += [
        f"<h2>{t['calibration']}</h2>",
        f'<p class="pending">{_md_inline(t["calibration_pending"])}</p>',
        f"<h2>{t['all_metrics']}</h2>",
        html_table(*tables["all_metrics"]),
        f"<h2>{t['method']}</h2>",
        "<ul>" + "".join(f"<li>{html.escape(p)}</li>" for p in _method_points(metrics, meta, lang)) + "</ul>",
        f'<footer>{t["repo"]}: <a href="{REPO_URL}">{REPO_URL}</a></footer>',
    ]
    return (
        f'<!doctype html>\n<html lang="{lang}">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<meta name="color-scheme" content="dark">\n'
        f"<title>{html.escape(t['title'])}</title>\n<style>{SITE_CSS}</style>\n</head>\n<body>\n<main>\n"
        + "\n".join(sections)
        + "\n</main>\n</body>\n</html>\n"
    )


def update_readme_block(path: Path, text: str) -> bool:
    """Replace the text between the results markers; returns False when the file or markers are missing."""
    if not path.exists():
        return False
    content = path.read_text(encoding="utf-8")
    if README_START not in content or README_END not in content:
        return False
    before, rest = content.split(README_START, 1)
    _, after = rest.split(README_END, 1)
    path.write_bytes(f"{before}{README_START}\n{text}\n{README_END}{after}".encode())
    return True


def report_meta(cfg: Config) -> dict:
    """Facts that are not in metrics.json: corpus date, chunkings, authoring, and (full profile) the machine."""
    snapshot = read_json(cfg.path(cfg["corpus"]["snapshot_path"]))
    meta = {
        "snapshot_date": snapshot["fetched_at"],
        "chunkings": list(cfg["retrieval"]["chunkings"]),
        "config_ids": config_ids(cfg),
        "generation_variants": list(cfg["generation"]["variants"]),
        "authored": cfg["questions"].get("authored"),
        "author_model": cfg["questions"].get("author_model"),
    }
    if cfg.profile == "full":
        manifest = _optional(cfg.run_dir / "manifest.json")
        if manifest:
            meta["machine"] = manifest["machine"]
    return meta


def build_report(cfg: Config) -> list[str]:
    run_dir = cfg.run_dir
    metrics = read_json(run_dir / "metrics.json")
    latency = _optional(run_dir / "latency.json")
    answers = _optional(run_dir / "answer_metrics.json")
    meta = report_meta(cfg)
    chunkings = meta["chunkings"]
    out_dir = cfg.path(cfg["paths"]["reports_dir"])
    site_dir = cfg.path(cfg["paths"]["site_dir"])
    written: list[str] = []

    def write(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))
        written.append(str(path.relative_to(cfg.root)).replace("\\", "/"))

    variants = metrics["variants"]
    svgs: dict[str, dict[str, str]] = {"light": {}, "dark": {}}
    for theme in ("light", "dark"):
        t = TEXT["de"]
        svgs[theme]["retrieval_matrix"] = retrieval_matrix_svg(
            variants, chunkings, t["chart_matrix_title"], theme
        )
        svgs[theme]["recall_at_k"] = recall_at_k_svg(variants, "C2", t["chart_k_title"], theme)
        if latency:
            svgs[theme]["latency"] = latency_svg(latency["configs"], "C2", t["chart_latency_title"], theme)
    for name, svg in svgs["light"].items():
        write(out_dir / "charts" / f"{name}.svg", svg)
    write(out_dir / "results.md", render_markdown(metrics, latency, answers, meta, "de"))
    write(out_dir / "results.en.md", render_markdown(metrics, latency, answers, meta, "en"))
    combined = {
        "retrieval": metrics,
        "latency": latency,
        "generation": answers or {"status": "pending_live_run"},
    }
    write_json(out_dir / "results.json", combined)
    written.append(str((out_dir / "results.json").relative_to(cfg.root)).replace("\\", "/"))
    write(site_dir / "index.html", render_html(metrics, latency, answers, meta, svgs["dark"], "de"))
    write(site_dir / "en.html", render_html(metrics, latency, answers, meta, svgs["dark"], "en"))
    if cfg.profile == "full":
        for name, lang in (("README.md", "de"), ("README.en.md", "en")):
            block = "\n".join(f"- {line}" for line in headline_lines(metrics, lang))
            if update_readme_block(cfg.path(name), block):
                written.append(name)
    return written
