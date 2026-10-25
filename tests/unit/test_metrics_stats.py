import math

import pytest

from pruefstand.metrics import gold_units_covered, mrr_at_k, ndcg_at_k, question_metrics, recall_at_k
from pruefstand.stats import bootstrap_ci, paired_bootstrap, percentile, weighted_kappa

LENGTHS = {"u1": 1000, "u2": 100}


def test_span_relevance_rules():
    assert gold_units_covered([{"unit_id": "u1", "start": 0, "end": 200}], ["u1"], LENGTHS) == ["u1"]
    assert gold_units_covered([{"unit_id": "u1", "start": 0, "end": 199}], ["u1"], LENGTHS) == []
    assert gold_units_covered([{"unit_id": "u2", "start": 50, "end": 100}], ["u2"], LENGTHS) == ["u2"]
    assert gold_units_covered([{"unit_id": "u2", "start": 51, "end": 100}], ["u2"], LENGTHS) == []
    assert gold_units_covered([{"unit_id": "u3", "start": 0, "end": 500}], ["u1"], LENGTHS) == []


def test_metrics_on_hand_computed_ranking():
    ranking = [[], ["u1"], ["u1"], [], ["u2"]]
    assert recall_at_k(ranking, 2, 1) == 0.0
    assert recall_at_k(ranking, 2, 3) == 0.5
    assert recall_at_k(ranking, 2, 5) == 1.0
    assert mrr_at_k(ranking) == 0.5
    dcg = 1 / math.log2(3) + 1 / math.log2(6)
    ideal = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(ranking, 2) == pytest.approx(dcg / ideal)


def test_ndcg_is_capped_and_zero_without_hits():
    assert ndcg_at_k([["u1", "u2"]], 2) == 1.0
    assert ndcg_at_k([[], []], 1) == 0.0
    values = question_metrics([["u1"]], 1)
    assert values["recall@1"] == 1.0 and values["mrr@10"] == 1.0 and values["ndcg@10"] == 1.0


def test_bootstrap_is_deterministic_with_seed():
    data = [0.0, 1.0, 1.0, 0.5, 0.25, 1.0]
    first = bootstrap_ci(data, 500, seed=7)
    assert first == bootstrap_ci(data, 500, seed=7)
    assert first["lo"] <= first["mean"] <= first["hi"]
    assert first["mean"] == round(sum(data) / len(data), 6)
    assert bootstrap_ci([], 100, 1)["mean"] is None


def test_paired_bootstrap_verdicts():
    better = paired_bootstrap([1.0] * 20, [0.0] * 20, 200, 1)
    assert better["diff"] == 1.0 and better["verdict"] == "better"
    same = paired_bootstrap([1, 0, 1, 0], [0, 1, 0, 1], 200, 1)
    assert same["verdict"] == "no_measurable_difference"
    assert paired_bootstrap([0.0] * 5, [1.0] * 5, 200, 1)["verdict"] == "worse"


def test_percentile_and_kappa():
    assert percentile([1, 2, 3, 4, 5], 50) == 3.0
    assert percentile([], 95) is None
    assert weighted_kappa([0, 1, 2, 2], [0, 1, 2, 2], [0, 1, 2]) == 1.0
    # quadratic weights (i-j)^2/4: observed disagreement 1.5/6 = 0.25, expected 3/9; kappa = 1 - 0.25 / (1/3)
    assert weighted_kappa([0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 2, 0], [0, 1, 2]) == pytest.approx(0.25)
