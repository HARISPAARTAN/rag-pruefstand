import math

import numpy as np
import pytest

from pruefstand.errors import PruefstandError
from pruefstand.metrics import (
    gold_units_covered,
    ideal_gains,
    locate_excerpts,
    mrr_at_k,
    ndcg_at_k,
    passage_spans,
    question_metrics,
    recall_at_k,
)
from pruefstand.stats import bootstrap_ci, paired_bootstrap, percentile, weighted_kappa

LENGTHS = {"u1": 1000, "u2": 100}


def test_span_relevance_rules():
    assert gold_units_covered([{"unit_id": "u1", "start": 0, "end": 200}], ["u1"], LENGTHS) == ["u1"]
    assert gold_units_covered([{"unit_id": "u1", "start": 0, "end": 199}], ["u1"], LENGTHS) == []
    assert gold_units_covered([{"unit_id": "u2", "start": 50, "end": 100}], ["u2"], LENGTHS) == ["u2"]
    assert gold_units_covered([{"unit_id": "u2", "start": 51, "end": 100}], ["u2"], LENGTHS) == []
    assert gold_units_covered([{"unit_id": "u3", "start": 0, "end": 500}], ["u1"], LENGTHS) == []


LONG = "Alpha " * 1000  # 6000 characters, a "giant" unit
LONG_LENGTHS = {"big": len(LONG), "u2": 100}
PASSAGE = "Zielwort eins  zwei drei"


def test_locate_excerpts_is_whitespace_flexible_and_handles_several_excerpts():
    text = "vorn Zielwort eins zwei drei mitte Endwort hinten"
    assert locate_excerpts(text, PASSAGE) == [(5, 28)]
    spans = locate_excerpts(text, "Zielwort eins zwei drei … Endwort")
    assert [text[a:b] for a, b in spans] == ["Zielwort eins zwei drei", "Endwort"]
    assert locate_excerpts(text, "nicht vorhanden") == []


def test_passage_spans_only_for_long_units_and_loud_when_the_passage_is_missing():
    texts = {"big": LONG + "Zielwort eins" + chr(10) + "zwei   drei", "u2": "kurz"}
    spans = passage_spans(["big", "u2"], PASSAGE, texts)
    assert list(spans) == ["big"] and len(spans["big"]) == 1
    with pytest.raises(PruefstandError, match="no excerpt"):
        passage_spans(["big"], "steht nirgends", texts, "q1")


def test_long_unit_chunk_outside_the_passage_is_not_relevant_but_a_covering_chunk_is():
    start = 3000
    spans = {"big": [(start, start + 120)]}  # a 120-character excerpt: threshold min(200, 60) = 60
    miss = [
        {"unit_id": "big", "start": 0, "end": 2000}
    ]  # 2000 characters of the same unit, far from the passage
    assert gold_units_covered(miss, ["big"], LONG_LENGTHS) == [
        "big"
    ]  # the old unit-level rule would count it
    assert gold_units_covered(miss, ["big"], LONG_LENGTHS, spans) == []
    near = [{"unit_id": "big", "start": 2500, "end": start + 59}]  # 59 characters of the excerpt: just short
    assert gold_units_covered(near, ["big"], LONG_LENGTHS, spans) == []
    hit = [{"unit_id": "big", "start": 2500, "end": start + 60}]  # 60 = half of the excerpt
    assert gold_units_covered(hit, ["big"], LONG_LENGTHS, spans) == ["big"]
    cover = [{"unit_id": "big", "start": start - 500, "end": start + 500}]
    assert gold_units_covered(cover, ["big"], LONG_LENGTHS, spans) == ["big"]


def test_long_excerpt_needs_200_characters_and_the_rule_is_per_gold_unit():
    spans = {"big": [(1000, 2000)]}  # 1000-character excerpt: threshold min(200, 500) = 200
    assert (
        gold_units_covered([{"unit_id": "big", "start": 1000, "end": 1199}], ["big"], LONG_LENGTHS, spans)
        == []
    )
    assert gold_units_covered(
        [{"unit_id": "big", "start": 1000, "end": 1200}], ["big"], LONG_LENGTHS, spans
    ) == ["big"]
    # a short gold unit next to a long one keeps the unit-level rule
    covers = [{"unit_id": "u2", "start": 50, "end": 100}, {"unit_id": "big", "start": 0, "end": 900}]
    assert gold_units_covered(covers, ["big", "u2"], LONG_LENGTHS, spans) == ["u2"]


def test_metrics_on_hand_computed_ranking():
    ranking = [[], ["u1"], ["u1"], [], ["u2"]]
    assert recall_at_k(ranking, 2, 1) == 0.0
    assert recall_at_k(ranking, 2, 3) == 0.5
    assert recall_at_k(ranking, 2, 5) == 1.0
    assert mrr_at_k(ranking) == 0.5
    dcg = 1 / math.log2(3) + 1 / math.log2(6)
    ideal = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(ranking, 2, [1, 1]) == pytest.approx(dcg / ideal)


def test_ideal_gains_is_the_greedy_cover():
    assert ideal_gains([["a", "b"], ["c"], ["a"]]) == [2, 1]
    assert ideal_gains([["a"], ["b"], ["c"]]) == [1, 1, 1]
    assert ideal_gains([["a", "b", "c"], ["a"], ["b"]]) == [3]
    assert ideal_gains([["a", "b"], ["b", "c"]]) == [2, 1]
    assert ideal_gains([[], []]) == []
    assert ideal_gains([]) == []


def test_ndcg_wide_chunk_has_no_advantage_over_exact_chunks():
    """Three gold units a, b, c. C1-like chunking: one chunk covers {a, b}, another {c}. C2-like: one per unit."""
    c1_ideal = ideal_gains([["a", "b"], ["c"]])
    c2_ideal = ideal_gains([["a"], ["b"], ["c"]])
    wide_first = ndcg_at_k([["a", "b"]], 3, c1_ideal)
    exact_two = ndcg_at_k([["a"], ["b"]], 3, c2_ideal)
    # both found two of three units; the old formula gave 0.939 against 0.765
    assert wide_first <= exact_two
    assert wide_first == pytest.approx(2 / (2 + 1 / math.log2(3)))
    assert exact_two == pytest.approx((1 + 1 / math.log2(3)) / (1 + 1 / math.log2(3) + 1 / 2))
    # the old cap case: both chunks of the cover retrieved gives exactly 1, never more
    assert ndcg_at_k([["a", "b"], ["c"]], 3, c1_ideal) == pytest.approx(1.0)
    assert ndcg_at_k([["a"], ["b"], ["c"]], 3, c2_ideal) == pytest.approx(1.0)


def test_ndcg_is_bounded_by_one_and_zero_without_hits():
    ideal = ideal_gains([["u1", "u2"]])
    assert ndcg_at_k([["u1", "u2"]], 2, ideal) == 1.0
    assert ndcg_at_k([["u1", "u2"], ["u1"], ["u2"]], 2, ideal) <= 1.0
    assert ndcg_at_k([[], []], 1, [1]) == 0.0
    assert ndcg_at_k([["u1"]], 1, []) == 0.0
    values = question_metrics([["u1"]], 1, [1])
    assert values["recall@1"] == 1.0 and values["mrr@10"] == 1.0 and values["ndcg@10"] == 1.0


def test_ndcg_never_exceeds_one_on_random_rankings():
    rng = np.random.default_rng(3)
    units = ["a", "b", "c", "d"]
    for _ in range(300):
        chunks = [[u for u in units if rng.random() < 0.4] for _ in range(int(rng.integers(len(units), 9)))]
        ideal = ideal_gains(chunks)
        order = rng.permutation(len(chunks))[:10]
        assert ndcg_at_k([chunks[i] for i in order], len(units), ideal) <= 1.0


def test_gold_validation():
    with pytest.raises(PruefstandError, match="not in the corpus"):
        gold_units_covered([{"unit_id": "u1", "start": 0, "end": 300}], ["u1", "nope"], LENGTHS)
    with pytest.raises(PruefstandError, match="duplicate"):
        gold_units_covered([{"unit_id": "u1", "start": 0, "end": 300}], ["u1", "u1"], LENGTHS)
    with pytest.raises(PruefstandError, match="at least one gold"):
        recall_at_k([["u1"]], 0, 5)
    with pytest.raises(PruefstandError, match="at least one gold"):
        ndcg_at_k([["u1"]], 0, [1])


def test_bootstrap_is_deterministic_with_seed():
    data = [0.0, 1.0, 1.0, 0.5, 0.25, 1.0]
    first = bootstrap_ci(data, 500, seed=7)
    assert first == bootstrap_ci(data, 500, seed=7)
    assert first["lo"] <= first["mean"] <= first["hi"]
    assert first["mean"] == round(sum(data) / len(data), 6)
    assert bootstrap_ci([], 100, 1)["mean"] is None


def test_bootstrap_matches_an_independent_percentile_bootstrap():
    """Full-size resampling over questions, the 2.5 / 97.5 percentiles and the configured seed."""
    data = np.array([0.0, 1.0, 1.0, 0.5, 0.25, 1.0, 0.0, 0.75, 1.0, 0.5])
    for seed in (7, 8):
        rng = np.random.default_rng(seed)
        means = data[rng.integers(0, data.size, size=(400, data.size))].mean(axis=1)
        lo, hi = np.percentile(means, [2.5, 97.5])
        result = bootstrap_ci(data.tolist(), 400, seed=seed)
        assert (result["lo"], result["hi"]) == (round(float(lo), 6), round(float(hi), 6))
    assert bootstrap_ci(data.tolist(), 400, seed=7) != bootstrap_ci(data.tolist(), 400, seed=8)


def test_bootstrap_interval_width_follows_the_standard_error():
    """For 500 zeros and 500 ones the 95% interval of the mean is about 2 * 1.96 * 0.5 / sqrt(n) wide."""
    data = [0.0, 1.0] * 500
    width = bootstrap_ci(data, 2000, seed=1)["hi"] - bootstrap_ci(data, 2000, seed=1)["lo"]
    assert 0.056 < width < 0.068
    small = [0.0, 1.0] * 50
    small_ci = bootstrap_ci(small, 2000, seed=1)
    assert 0.17 < small_ci["hi"] - small_ci["lo"] < 0.22  # about 3.2x wider for 10x fewer questions
    assert bootstrap_ci([1.0, 1.0, 1.0], 100, 1) == {"mean": 1.0, "lo": 1.0, "hi": 1.0, "n": 3}


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


def test_ndcg_ideal_is_cut_at_k():
    """Twelve gold units: a perfect top 10 (one new unit per rank) is ideal at k = 10."""
    ranking = [[f"u{i}"] for i in range(10)]
    assert ndcg_at_k(ranking, 12, [1] * 12, k=10) == pytest.approx(1.0)
    assert ndcg_at_k(ranking, 12, [1] * 12, k=5) == pytest.approx(1.0)
