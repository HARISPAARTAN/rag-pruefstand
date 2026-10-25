"""Price table, cost from usage, cost projection and the budget guard."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from pruefstand.errors import PruefstandError
from pruefstand.llm.base import Usage

PER_MILLION = 1_000_000


class BudgetExceeded(PruefstandError):
    pass


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    cache_read: float
    cache_write: float


@dataclass(frozen=True)
class PriceTable:
    prices: dict[str, Price]
    batch_discount: float
    eur_per_usd: float
    meta: dict

    def price(self, model: str) -> Price:
        if model not in self.prices:
            raise KeyError(f"no price for model {model!r} in costs.toml")
        return self.prices[model]


def load_prices(path: str | Path) -> PriceTable:
    with open(path, "rb") as handle:
        raw = tomllib.load(handle)
    meta = raw["meta"]
    prices = {
        model: Price(entry["input"], entry["output"], entry["cache_read"], entry["cache_write"])
        for model, entry in raw["models"].items()
    }
    return PriceTable(
        prices=prices, batch_discount=meta["batch_discount"], eur_per_usd=meta["eur_per_usd"], meta=meta
    )


def cost_usd(usage: Usage, price: Price, batch: bool = False, batch_discount: float = 0.5) -> float:
    total = (
        usage.input_tokens * price.input
        + usage.output_tokens * price.output
        + usage.cache_read_input_tokens * price.cache_read
        + usage.cache_creation_input_tokens * price.cache_write
    ) / PER_MILLION
    if batch:
        total *= 1 - batch_discount
    return round(total, 8)


@dataclass(frozen=True)
class StageProjection:
    name: str
    model: str
    calls: int
    input_tokens: int
    output_tokens: int
    batch: bool
    usd: float


def project_stages(stages: list[dict], roles: dict[str, dict], table: PriceTable) -> list[StageProjection]:
    projections = []
    for stage in stages:
        model = roles[stage["role"]]["model"]
        usage = Usage(
            input_tokens=stage["calls"] * stage["input_tokens"],
            output_tokens=stage["calls"] * stage["output_tokens"],
        )
        usd = cost_usd(usage, table.price(model), batch=stage["batch"], batch_discount=table.batch_discount)
        projections.append(
            StageProjection(
                stage["name"],
                model,
                stage["calls"],
                stage["input_tokens"],
                stage["output_tokens"],
                stage["batch"],
                round(usd, 4),
            )
        )
    return projections


class BudgetGuard:
    def __init__(self, max_usd: float) -> None:
        self.max_usd = max_usd
        self.spent = 0.0

    def check_projection(self, projected_usd: float) -> None:
        if self.spent + projected_usd > self.max_usd:
            raise BudgetExceeded(
                f"projected spend USD {projected_usd:.2f} (+{self.spent:.2f} spent) exceeds the cap USD {self.max_usd:.2f}"
            )

    def add(self, usd: float) -> None:
        self.spent += usd
        if self.spent > self.max_usd:
            raise BudgetExceeded(
                f"spend USD {self.spent:.2f} passed the cap USD {self.max_usd:.2f}; run stopped"
            )
