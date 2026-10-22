"""Tests for evaluation metrics."""

from chemcost.evaluation.metrics import (
    ComponentMatch,
    CostPrediction,
    component_names_equivalent,
    component_precision,
    component_recall,
    cta_at_k,
    evaluate,
    tcre,
    token_efficiency,
)


def test_tcre_exact():
    assert tcre(2.0, 2.0) == 0.0


def test_tcre_relative():
    assert abs(tcre(2.5, 2.0) - 0.25) < 1e-10


def test_cta_at_k():
    preds = [
        CostPrediction("r1", predicted_cost=2.0, true_cost=2.0),  # 0% error
        CostPrediction("r2", predicted_cost=2.2, true_cost=2.0),  # 10% error
        CostPrediction("r3", predicted_cost=3.0, true_cost=2.0),  # 50% error
        CostPrediction("r4", predicted_cost=5.0, true_cost=2.0),  # 150% error
    ]

    assert cta_at_k(preds, 10) == 0.5  # r2 is exactly at the 10% boundary and counts
    assert cta_at_k(preds, 11) == 0.5  # r1 and r2 within 11%
    assert cta_at_k(preds, 50) == 0.75  # r1, r2, r3 within 50%
    assert cta_at_k(preds, 200) == 1.0  # all within 200%


def test_missing_prediction_counts_as_failure():
    preds = [
        CostPrediction("r1", predicted_cost=2.0, true_cost=2.0),
        CostPrediction("r2", predicted_cost=None, true_cost=2.0),
    ]
    # 1 correct out of 2 total
    assert cta_at_k(preds, 10) == 0.5


def test_evaluate_full():
    preds = [
        CostPrediction("r1", predicted_cost=1.0, true_cost=1.0, true_component_names=["A", "B"]),
        CostPrediction("r2", predicted_cost=1.5, true_cost=1.0, true_component_names=["C"]),
    ]
    results = evaluate(preds)
    assert results.n_total == 2
    assert results.n_predicted == 2
    assert results.mean_tcre >= 0


def test_component_alias_matching_accepts_unambiguous_aliases():
    preds = [
        CostPrediction(
            "r1",
            predicted_cost=1.0,
            true_cost=1.0,
            predicted_components=[ComponentMatch(name="Et3N", found=True)],
            true_component_names=["triethylamine"],
        ),
    ]
    assert component_names_equivalent("Et3N", "triethylamine") is True
    assert component_recall(preds) == 1.0
    assert component_precision(preds) == 1.0


def test_component_alias_matching_rejects_ambiguous_aliases():
    preds = [
        CostPrediction(
            "r1",
            predicted_cost=1.0,
            true_cost=1.0,
            predicted_components=[ComponentMatch(name="TEA", found=False)],
            true_component_names=["triethylamine"],
        ),
    ]
    assert component_names_equivalent("TEA", "triethylamine") is False
    assert component_recall(preds) == 0.0
    assert component_precision(preds) == 0.0


# --- token_efficiency tests ---


def test_token_efficiency_basic():
    results = [
        {
            "reaction_id": "r1",
            "predicted_cost_per_gram": 2.0,
            "true_cost": 2.0,
            "token_usage": {"input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500},
        },
        {
            "reaction_id": "r2",
            "predicted_cost_per_gram": 3.0,
            "true_cost": 2.0,
            "token_usage": {"input_tokens": 800, "output_tokens": 400, "total_tokens": 1200},
        },
    ]
    te = token_efficiency(results)
    assert te is not None
    assert te["total_input_tokens"] == 1800
    assert te["total_output_tokens"] == 900
    assert te["total_tokens"] == 2700
    assert te["tokens_per_reaction"] == 1350.0


def test_token_efficiency_no_token_data():
    results = [
        {"reaction_id": "r1", "predicted_cost_per_gram": 2.0, "true_cost": 2.0},
        {"reaction_id": "r2", "predicted_cost_per_gram": 3.0, "true_cost": 2.0},
    ]
    te = token_efficiency(results)
    assert te is None


def test_token_efficiency_partial_token_data():
    """Only reactions with token_usage are counted."""
    results = [
        {
            "reaction_id": "r1",
            "predicted_cost_per_gram": 2.0,
            "true_cost": 2.0,
            "token_usage": {"input_tokens": 500, "output_tokens": 200, "total_tokens": 700},
        },
        {
            "reaction_id": "r2",
            "predicted_cost_per_gram": 3.0,
            "true_cost": 2.0,
        },
    ]
    te = token_efficiency(results)
    assert te is not None
    assert te["total_tokens"] == 700
    assert te["tokens_per_reaction"] == 700.0


def test_token_efficiency_cta25_per_million():
    """CTA@25 per million tokens should be > 0 when there are correct predictions."""
    results = [
        {
            "reaction_id": "r1",
            "predicted_cost_per_gram": 2.0,
            "true_cost": 2.0,  # exact match → within 25%
            "token_usage": {"input_tokens": 500_000, "output_tokens": 500_000},
        },
    ]
    te = token_efficiency(results)
    assert te is not None
    # CTA@25 = 1.0, total = 1M tokens → cta25_per_million = 1.0
    assert te["cta25_per_million_tokens"] == 1.0


def test_token_efficiency_empty():
    te = token_efficiency([])
    assert te is None
