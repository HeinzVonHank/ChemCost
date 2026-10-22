"""Regression tests for noise propagation through evaluation input prep."""

from __future__ import annotations

import json

from chemcost.evaluation.evaluator import run_evaluation
from chemcost.noise.noise_injector import inject_noise


class RecordingAgent:
    """Minimal agent that records the reaction payload it received."""

    def __init__(self) -> None:
        self.seen: list[dict] = []

    def estimate_cost(self, reaction: dict) -> dict:
        self.seen.append(reaction)
        return {
            "predicted_cost_per_gram": 10.0,
            "predicted_components": [{"name": "substrate", "price_per_gram": 1.0}],
        }


def _write_dataset(tmp_path, record: dict) -> str:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(json.dumps(record) + "\n")
    return str(dataset_path)


def test_structured_evaluation_preserves_quantity_noise(tmp_path):
    record = {
        "reaction_id": "NOISE-001",
        "reaction_name": "Quantity Noise",
        "yield_percent": 80.0,
        "product_mw": 100.0,
        "procurement_cost_usd_per_g_product": 10.0,
        "components": [
            {"name": "Pd(OAc)2", "role": "catalyst", "equivalents": 0.05, "mw": 224.5, "amount_g": 0.022},
        ],
    }
    dataset_path = _write_dataset(tmp_path, record)
    agent = RecordingAgent()

    run_evaluation(
        agent,
        dataset_path,
        max_workers=1,
        record_transform=lambda row: inject_noise(
            row,
            noise_types=["quantity"],
            noise_level="high",
            seed=42,
        ),
    )

    assert agent.seen
    component = agent.seen[0]["components"][0]
    assert "equivalents" not in component
    assert component["amount"] == "cat."


def test_structured_evaluation_preserves_format_description(tmp_path):
    record = {
        "reaction_id": "NOISE-002",
        "reaction_name": "Format Noise",
        "yield_percent": 80.0,
        "product_mw": 100.0,
        "procurement_cost_usd_per_g_product": 10.0,
        "components": [
            {"name": "Na2CO3", "role": "base", "equivalents": 2.0, "mw": 106.0, "amount_g": 0.424},
        ],
    }
    dataset_path = _write_dataset(tmp_path, record)
    agent = RecordingAgent()

    run_evaluation(
        agent,
        dataset_path,
        max_workers=1,
        input_format="structured",
        record_transform=lambda row: inject_noise(
            row,
            noise_types=["format"],
            noise_level="medium",
            seed=42,
        ),
    )

    assert agent.seen
    seen = agent.seen[0]
    assert "description" in seen
    assert "Na2CO3" in seen["description"] and "424 mg" in seen["description"]
    assert "smiles" not in " ".join(seen).lower()


def test_evaluation_output_includes_supplementary_metrics(tmp_path):
    record = {
        "reaction_id": "NOISE-003",
        "reaction_name": "Supplementary Metrics",
        "yield_percent": 80.0,
        "product_mw": 100.0,
        "procurement_cost_usd_per_g_product": 10.0,
        "components": [
            {"name": "substrate", "role": "reactant", "equivalents": 1.0, "mw": 100.0},
        ],
    }
    dataset_path = _write_dataset(tmp_path, record)
    output_path = tmp_path / "results.json"

    class TokenAgent(RecordingAgent):
        def estimate_cost(self, reaction: dict) -> dict:
            self.seen.append(reaction)
            return {
                "predicted_cost_per_gram": 10.0,
                "predicted_components": [{"name": "substrate", "price_per_gram": 1.0}],
                "token_usage": {"input_tokens": 120, "output_tokens": 30, "total_tokens": 150},
            }

    run_evaluation(
        TokenAgent(),
        dataset_path,
        output_path=output_path,
        max_workers=1,
    )

    payload = json.loads(output_path.read_text())
    assert payload["supplementary_metrics"]["token_efficiency"]["total_tokens"] == 150
    assert "price_optimization_score" not in payload["supplementary_metrics"]
