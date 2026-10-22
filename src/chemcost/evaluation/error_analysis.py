"""Detailed error categorization for ChemCost benchmark predictions.

Classifies prediction errors into actionable categories to help
understand *why* agents fail, not just *how much* they miss by.
"""

from __future__ import annotations

from collections import Counter
from enum import Enum

from .metrics import CostPrediction, component_names_equivalent


class ErrorCategory(Enum):
    """Categories of prediction errors."""

    CHEMICAL_NOT_FOUND = "chemical_not_found"       # Failed to identify a component
    WRONG_PRICE = "wrong_price"                      # Found wrong price (>50% off)
    WRONG_MW = "wrong_mw"                           # Wrong molecular weight (>10% off)
    WRONG_EQUIVALENTS = "wrong_equivalents"          # Misinterpreted stoichiometry
    CALCULATION_ERROR = "calculation_error"           # Arithmetic mistake
    MISSING_COMPONENT = "missing_component"          # Didn't cost a component
    EXTRA_COMPONENT = "extra_component"              # Costed something that shouldn't be
    ROUTE_SELECTION_ERROR = "wrong_route"            # Picked non-optimal route


def _visible_truth_components(ground_truth: dict) -> list[dict]:
    return [
        component
        for component in ground_truth.get("components", [])
        if component.get("role") != "product" and component.get("original_role") != "product"
    ]


def _component_name(component: dict) -> str:
    return component.get("name") or component.get("smiles") or "UNKNOWN_COMPONENT"


def _find_matching_truth_component(name: str, components: list[dict]) -> dict | None:
    for component in components:
        if component_names_equivalent(name, _component_name(component)):
            return component
    return None


def categorize_errors(
    prediction: CostPrediction, ground_truth: dict
) -> list[ErrorCategory]:
    """Categorize errors in a single prediction against ground truth.

    Args:
        prediction: The agent's cost prediction.
        ground_truth: The reaction dict with full ground-truth data
            including components with price_per_gram_usd, mw, equivalents.

    Returns:
        List of applicable error categories (may be empty if prediction
        is correct, or contain multiple categories).
    """
    errors: list[ErrorCategory] = []

    true_components = _visible_truth_components(ground_truth)

    # --- Missing components ---
    for true_name in prediction.true_component_names:
        if not any(
            component_names_equivalent(true_name, predicted.name)
            for predicted in prediction.predicted_components
        ):
            errors.append(ErrorCategory.MISSING_COMPONENT)

    # --- Extra components ---
    for m in prediction.predicted_components:
        if not any(
            component_names_equivalent(m.name, true_name)
            for true_name in prediction.true_component_names
        ):
            errors.append(ErrorCategory.EXTRA_COMPONENT)

    # --- Per-component price errors ---
    for m in prediction.predicted_components:
        if not m.found:
            errors.append(ErrorCategory.CHEMICAL_NOT_FOUND)
            continue

        true_component = _find_matching_truth_component(m.name, true_components)
        true_price = m.price_true
        if true_price is None and true_component is not None:
            true_price = true_component.get("price_per_gram_usd")

        # Check price accuracy (>50% relative error = wrong price)
        if (
            m.price_pred is not None
            and true_price is not None
            and true_price > 0
        ):
            price_error = abs(m.price_pred - true_price) / true_price
            if price_error > 0.5:
                errors.append(ErrorCategory.WRONG_PRICE)

    # --- MW errors ---
    # Check if ground truth has MW info and predicted components have MW
    for m in prediction.predicted_components:
        true_component = _find_matching_truth_component(m.name, true_components)
        true_mw = m.mw_true
        if true_mw is None and true_component is not None:
            true_mw = true_component.get("mw")
        pred_mw = m.mw_pred
        if true_mw and pred_mw and true_mw > 0:
            mw_error = abs(pred_mw - true_mw) / true_mw
            if mw_error > 0.1:
                errors.append(ErrorCategory.WRONG_MW)

    # --- Calculation error ---
    # If components and prices are roughly right but final cost is way off,
    # suspect a calculation error.
    if (
        prediction.predicted_cost is not None
        and prediction.true_cost is not None
        and prediction.true_cost > 0
    ):
        overall_error = abs(prediction.predicted_cost - prediction.true_cost) / prediction.true_cost

        # If the overall error is large (>25%) but we have no price/component
        # errors, it's likely a calculation or equivalents error.
        price_and_component_errors = {
            ErrorCategory.WRONG_PRICE,
            ErrorCategory.MISSING_COMPONENT,
            ErrorCategory.EXTRA_COMPONENT,
            ErrorCategory.CHEMICAL_NOT_FOUND,
        }
        has_data_errors = any(e in price_and_component_errors for e in errors)

        if overall_error > 0.25 and not has_data_errors:
            errors.append(ErrorCategory.CALCULATION_ERROR)

    return errors


def error_distribution(
    predictions: list[CostPrediction],
    ground_truths: list[dict],
) -> dict[str, int]:
    """Compute error category distribution across all predictions.

    Args:
        predictions: List of cost predictions.
        ground_truths: Corresponding list of ground-truth reaction dicts.

    Returns:
        Dict mapping error category value strings to counts.
    """
    counts: Counter = Counter()
    for pred, gt in zip(predictions, ground_truths):
        for error in categorize_errors(pred, gt):
            counts[error.value] += 1
    return dict(counts)


def error_summary(
    predictions: list[CostPrediction],
    ground_truths: list[dict],
) -> dict:
    """Compute a full error analysis summary.

    Returns:
        Dict with error distribution, per-reaction breakdown, and
        the fraction of reactions affected by each error type.
    """
    distribution = error_distribution(predictions, ground_truths)
    n = len(predictions)

    # Per-reaction: which errors affect each reaction
    per_reaction: list[dict] = []
    reactions_with_error: Counter = Counter()

    for pred, gt in zip(predictions, ground_truths):
        errs = categorize_errors(pred, gt)
        err_values = [e.value for e in errs]
        per_reaction.append({
            "reaction_id": pred.reaction_id,
            "errors": err_values,
            "n_errors": len(err_values),
        })
        # Count unique error types per reaction
        for e in set(err_values):
            reactions_with_error[e] += 1

    # Fraction of reactions affected
    fraction_affected = {
        cat: count / n if n > 0 else 0.0
        for cat, count in reactions_with_error.items()
    }

    return {
        "distribution": distribution,
        "fraction_reactions_affected": fraction_affected,
        "per_reaction": per_reaction,
        "total_errors": sum(distribution.values()),
    }
