"""Evaluation metrics for ChemCost benchmark."""

from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import dataclass, field

from ..noise.chemical_aliases import (
    ABBREVIATION_TO_FULL,
    AMBIGUOUS_ABBREVIATIONS,
    COMMON_TO_IUPAC,
    SALT_VARIATIONS,
)
from .tool_tracker import ToolUsageStats


@dataclass
class ComponentMatch:
    """Match result for a single component."""

    name: str
    found: bool
    price_pred: float | None = None
    price_true: float | None = None
    mw_pred: float | None = None
    mw_true: float | None = None


@dataclass
class CostPrediction:
    """A single cost prediction to evaluate."""

    reaction_id: str
    predicted_cost: float | None  # $/g product (procurement)
    true_cost: float | None       # $/g product (procurement ground truth)
    predicted_components: list[ComponentMatch] = field(default_factory=list)
    true_component_names: list[str] = field(default_factory=list)


def _normalize_component_name(name: str) -> str:
    """Normalize surface-form variations that are irrelevant for matching."""
    return " ".join(name.strip().lower().split())


def _build_alias_index() -> dict[str, frozenset[str]]:
    """Build equivalence classes for unambiguous component aliases."""
    groups: list[set[str]] = []

    for abbreviation, full_names in ABBREVIATION_TO_FULL.items():
        if abbreviation in AMBIGUOUS_ABBREVIATIONS:
            continue
        group = {_normalize_component_name(abbreviation)}
        group.update(_normalize_component_name(full_name) for full_name in full_names)
        if len(group) > 1:
            groups.append(group)

    for common_name, iupac_name in COMMON_TO_IUPAC.items():
        groups.append({
            _normalize_component_name(common_name),
            _normalize_component_name(iupac_name),
        })

    for named_form, formula in SALT_VARIATIONS.items():
        groups.append({
            _normalize_component_name(named_form),
            _normalize_component_name(formula),
        })

    merged_groups: list[set[str]] = []
    for group in groups:
        overlaps = [existing for existing in merged_groups if existing & group]
        if not overlaps:
            merged_groups.append(set(group))
            continue

        merged = set(group)
        for existing in overlaps:
            merged |= existing
            merged_groups.remove(existing)
        merged_groups.append(merged)

    alias_index: dict[str, frozenset[str]] = {}
    for group in merged_groups:
        frozen_group = frozenset(group)
        for alias in frozen_group:
            alias_index[alias] = frozen_group
    return alias_index


_NON_AMBIGUOUS_ALIAS_INDEX = _build_alias_index()


def component_names_equivalent(lhs: str, rhs: str) -> bool:
    """Return whether two names are equivalent without collapsing ambiguity."""
    left = _normalize_component_name(lhs)
    right = _normalize_component_name(rhs)
    if not left or not right:
        return False
    if left == right:
        return True

    left_group = _NON_AMBIGUOUS_ALIAS_INDEX.get(left)
    right_group = _NON_AMBIGUOUS_ALIAS_INDEX.get(right)
    if left_group is None or right_group is None:
        return False
    return left_group == right_group


# Tolerance for float rounding at the CTA@k boundary (e.g. 1.1 * x / x - 1 = 0.10000000000000009).
_CTA_EPS = 1e-9


def _finite_or_none(x: float) -> float | None:
    return round(x, 4) if math.isfinite(x) else None


def tcre(pred: float, true: float) -> float:
    """Total Cost Relative Error: |pred - true| / true."""
    if true == 0:
        return float("inf") if pred != 0 else 0.0
    return abs(pred - true) / true


def _is_evaluable(prediction: CostPrediction) -> bool:
    """Return whether a prediction has usable ground truth."""
    return prediction.true_cost is not None and prediction.true_cost > 0


def cta_at_k(predictions: list[CostPrediction], k: float) -> float:
    """Cost Tolerance Accuracy at k%: fraction of predictions within k% of true value.

    Missing predictions count as failures.
    """
    evaluable = [p for p in predictions if _is_evaluable(p)]
    if not evaluable:
        return 0.0
    valid = [p for p in evaluable if p.predicted_cost is not None]
    within = sum(1 for p in valid if tcre(p.predicted_cost, p.true_cost) <= k / 100 + _CTA_EPS)
    return within / len(evaluable)


def component_recall(predictions: list[CostPrediction]) -> float:
    """Fraction of true components that were identified by the agent."""
    total_true = 0
    total_found = 0
    for p in predictions:
        total_true += len(p.true_component_names)
        total_found += sum(
            1
            for true_name in p.true_component_names
            if any(
                m.found and component_names_equivalent(true_name, m.name)
                for m in p.predicted_components
            )
        )
    return total_found / total_true if total_true > 0 else 0.0


def component_precision(predictions: list[CostPrediction]) -> float:
    """Fraction of predicted components that are actually in the true set."""
    total_pred = 0
    total_correct = 0
    for p in predictions:
        for m in p.predicted_components:
            total_pred += 1
            if any(
                component_names_equivalent(m.name, true_name)
                for true_name in p.true_component_names
            ):
                total_correct += 1
    return total_correct / total_pred if total_pred > 0 else 0.0


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results."""

    n_total: int
    n_predicted: int
    mean_tcre: float
    median_tcre: float
    cta_10: float
    cta_25: float
    cta_50: float
    component_recall: float
    component_precision: float

    def to_dict(self) -> dict:
        return {
            "n_total": self.n_total,
            "n_predicted": self.n_predicted,
            "mean_tcre": _finite_or_none(self.mean_tcre),
            "median_tcre": _finite_or_none(self.median_tcre),
            "cta@10": round(self.cta_10, 4),
            "cta@25": round(self.cta_25, 4),
            "cta@50": round(self.cta_50, 4),
            "component_recall": round(self.component_recall, 4),
            "component_precision": round(self.component_precision, 4),
        }


def tool_efficiency(
    predictions: list[CostPrediction],
    total_tool_calls: int,
    tolerance_k: float = 25,
) -> float:
    """Correct predictions per tool call (CTA@tolerance_k / total_tool_calls)."""
    if total_tool_calls == 0:
        return 0.0
    valid = [p for p in predictions if p.predicted_cost is not None and _is_evaluable(p)]
    n_correct = sum(1 for p in valid if tcre(p.predicted_cost, p.true_cost) <= tolerance_k / 100 + _CTA_EPS)
    return n_correct / total_tool_calls


def tool_usage_summary(stats_list: list[ToolUsageStats]) -> dict:
    """Aggregate tool-usage stats across runs."""
    if not stats_list:
        return {
            "total_calls": 0,
            "calls_per_tool": {},
            "success_rate_per_tool": {},
            "mean_calls_per_reaction": 0.0,
            "total_redundant_calls": 0,
        }

    calls_per_tool: Counter[str] = Counter()
    successful_calls_per_tool: Counter[str] = Counter()
    total_calls = 0
    total_redundant_calls = 0
    total_avg_calls = 0.0

    for stats in stats_list:
        total_calls += stats.total_calls
        total_redundant_calls += stats.redundant_calls
        total_avg_calls += stats.avg_calls_per_reaction
        calls_per_tool.update(stats.calls_per_tool)
        for tool_name, count in stats.calls_per_tool.items():
            success_rate = stats.success_rate_per_tool.get(tool_name, 0.0)
            successful_calls_per_tool[tool_name] += success_rate * count

    success_rate_per_tool = {
        tool_name: round(successful_calls_per_tool[tool_name] / count, 4)
        for tool_name, count in calls_per_tool.items()
        if count > 0
    }

    return {
        "total_calls": total_calls,
        "calls_per_tool": dict(calls_per_tool),
        "success_rate_per_tool": success_rate_per_tool,
        "mean_calls_per_reaction": round(total_avg_calls / len(stats_list), 4),
        "total_redundant_calls": total_redundant_calls,
    }


def evaluate(predictions: list[CostPrediction]) -> BenchmarkResults:
    """Compute all benchmark metrics."""
    evaluable = [p for p in predictions if _is_evaluable(p)]
    valid = [p for p in evaluable if p.predicted_cost is not None]
    tcre_values = [tcre(p.predicted_cost, p.true_cost) for p in valid]

    return BenchmarkResults(
        n_total=len(evaluable),
        n_predicted=len(valid),
        mean_tcre=statistics.mean(tcre_values) if tcre_values else float("inf"),
        median_tcre=statistics.median(tcre_values) if tcre_values else float("inf"),
        cta_10=cta_at_k(evaluable, 10),
        cta_25=cta_at_k(evaluable, 25),
        cta_50=cta_at_k(evaluable, 50),
        component_recall=component_recall(evaluable),
        component_precision=component_precision(evaluable),
    )


def token_efficiency(results: list[dict]) -> dict | None:
    """Compute token efficiency metrics from agent run results.

    Each result dict should contain a "token_usage" key with sub-keys
    "input_tokens", "output_tokens", and "total_tokens".  If no result
    carries token data, returns None gracefully.

    Args:
        results: list of per-reaction result dicts (the raw agent outputs,
                 not CostPrediction objects).

    Returns:
        Dict with total_input_tokens, total_output_tokens, total_tokens,
        tokens_per_reaction, and cta25_per_million_tokens.
        Returns None if token data is unavailable.
    """
    token_results = [
        r for r in results
        if r.get("token_usage") is not None
    ]
    if not token_results:
        return None

    total_input = sum(r["token_usage"].get("input_tokens", 0) for r in token_results)
    total_output = sum(r["token_usage"].get("output_tokens", 0) for r in token_results)
    total = total_input + total_output

    n_reactions = len(token_results)
    tokens_per_reaction = total / n_reactions if n_reactions > 0 else 0.0

    # Compute CTA@25 for the subset with token data to get cta25_per_million_tokens
    preds_with_tokens = []
    for r in token_results:
        pred_cost = r.get("predicted_cost_per_gram")
        true_cost = r.get("true_cost")
        if true_cost is not None and true_cost > 0:
            preds_with_tokens.append(
                CostPrediction(
                    reaction_id=r.get("reaction_id", ""),
                    predicted_cost=pred_cost,
                    true_cost=true_cost,
                )
            )
    cta25 = cta_at_k(preds_with_tokens, 25) if preds_with_tokens else 0.0
    cta25_per_m = (cta25 / (total / 1_000_000)) if total > 0 else 0.0

    return {
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_tokens": total,
        "tokens_per_reaction": round(tokens_per_reaction, 1),
        "cta25_per_million_tokens": round(cta25_per_m, 4),
    }
