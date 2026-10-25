"""SVG charts rendered by matplotlib, deterministic (fixed hash salt, no date metadata, text kept as text).

Palette: categorical slots from the dataviz reference palette, validated for the white (light) and #0b0f14
(dark) surfaces; light slots below 3:1 contrast are relieved by direct labels and the tables in the report.
"""

from __future__ import annotations

import io
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

THEMES = {
    "light": {
        "surface": "#ffffff",
        "text": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e4e3df",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"],
    },
    "dark": {
        "surface": "#0b0f14",
        "text": "#f0efec",
        "muted": "#c3c2b7",
        "grid": "#2a2f36",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#9085e9"],
    },
}
RETRIEVERS = ("bm25", "e5", "mini", "hyb", "e5+rr", "hyb+rr")
WEB_FONTS = "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"


def _setup() -> None:
    plt.rcParams.update(
        {
            "svg.hashsalt": "rag-pruefstand",
            "svg.fonttype": "none",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _style(ax, theme: dict) -> None:
    ax.set_facecolor(theme["surface"])
    ax.tick_params(colors=theme["muted"], length=0)
    for spine in ax.spines.values():
        spine.set_color(theme["grid"])
    ax.grid(axis="x", color=theme["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def _to_svg(fig) -> str:
    buffer = io.StringIO()
    fig.savefig(buffer, format="svg", metadata={"Date": None, "Creator": None}, bbox_inches="tight")
    plt.close(fig)
    svg = buffer.getvalue().replace("'DejaVu Sans'", WEB_FONTS)
    return re.sub(r"<\?xml[^>]*\?>\s*|<!DOCTYPE[^>]*>\s*", "", svg)


def retrieval_matrix_svg(variants: dict, chunkings: list[str], title: str, theme_name: str) -> str:
    _setup()
    theme = THEMES[theme_name]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor=theme["surface"])
    _style(ax, theme)
    height = 0.36
    for offset, chunking in enumerate(chunkings):
        color = theme["series"][offset]
        for row, retriever in enumerate(RETRIEVERS):
            entry = variants[f"{retriever}-{chunking}"]["metrics"]["recall@5"]
            y = row + (offset - 0.5) * (height + 0.04)
            ax.barh(y, entry["mean"], height=height, color=color, label=chunking if row == 0 else None)
            ax.plot([entry["lo"], entry["hi"]], [y, y], color=theme["text"], linewidth=1.2)
            ax.text(
                entry["hi"] + 0.01, y, f"{entry['mean']:.2f}", va="center", fontsize=8, color=theme["muted"]
            )
    ax.set_yticks(range(len(RETRIEVERS)), RETRIEVERS, color=theme["text"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_title(title, loc="left", color=theme["text"], fontsize=11)
    legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=len(chunkings), frameon=False)
    for text in legend.get_texts():
        text.set_color(theme["text"])
    return _to_svg(fig)


def recall_at_k_svg(variants: dict, chunking: str, title: str, theme_name: str) -> str:
    _setup()
    theme = THEMES[theme_name]
    ks = [1, 3, 5, 10]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor=theme["surface"])
    _style(ax, theme)
    ax.grid(axis="y", color=theme["grid"], linewidth=0.8)
    for slot, retriever in enumerate(RETRIEVERS):
        metrics = variants[f"{retriever}-{chunking}"]["metrics"]
        values = [metrics[f"recall@{k}"]["mean"] for k in ks]
        color = theme["series"][slot]
        ax.plot(ks, values, color=color, linewidth=2, marker="o", markersize=5, label=retriever)
    ax.set_xticks(ks, [f"k={k}" for k in ks], color=theme["text"])
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0, 1)
    ax.set_title(title, loc="left", color=theme["text"], fontsize=11)
    legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False, ncol=len(RETRIEVERS))
    for text in legend.get_texts():
        text.set_color(theme["text"])
    return _to_svg(fig)


def latency_svg(configs: dict, chunking: str, title: str, theme_name: str) -> str:
    _setup()
    theme = THEMES[theme_name]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), facecolor=theme["surface"])
    _style(ax, theme)
    color = theme["series"][0]
    for row, retriever in enumerate(RETRIEVERS):
        total = configs[f"{retriever}-{chunking}"]["total"]
        ax.barh(row, total["p50"], height=0.55, color=color)
        ax.plot([total["p95"]], [row], marker="|", markersize=14, color=theme["text"])
        ax.text(
            total["p95"] * 1.02 + 1,
            row,
            f"p50 {total['p50']:.0f} · p95 {total['p95']:.0f}",
            va="center",
            fontsize=8,
            color=theme["muted"],
        )
    ax.set_yticks(range(len(RETRIEVERS)), RETRIEVERS, color=theme["text"])
    ax.invert_yaxis()
    right = max(configs[f"{r}-{chunking}"]["total"]["p95"] for r in RETRIEVERS)
    ax.set_xlim(0, right * 1.45 + 5)
    ax.set_title(title, loc="left", color=theme["text"], fontsize=11)
    return _to_svg(fig)
