"""Evaluation pipeline: run an agent on the benchmark and compute metrics."""

from __future__ import annotations

import json
import logging
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Literal, Protocol

from .metrics import (
    BenchmarkResults,
    ComponentMatch,
    CostPrediction,
    component_names_equivalent,
    evaluate,
    token_efficiency,
)
from .tool_tracker import ToolTracker

logger = logging.getLogger(__name__)


class CostEstimationAgent(Protocol):
    """Protocol for agents that estimate reaction costs."""

    def estimate_cost(self, reaction: dict) -> dict:
        """Given a reaction record, return a cost estimate.

        Input: reaction dict with keys like reaction_name, product_mw,
        and components (names only, with SMILES intentionally withheld).
        Output: dict with predicted_cost_per_gram and predicted_components
        (list of {name, price_per_gram}).
        """
        ...


def _ground_truth_cost(reaction: dict) -> float | None:
    """Return the benchmark ground-truth procurement cost ($/g product)."""
    return reaction.get("procurement_cost_usd_per_g_product")


def _coerce_cost(value) -> float | None:
    """Return a finite float cost, or None (abstention) for anything else."""
    if isinstance(value, bool):
        return None
    try:
        cost = float(value)
    except (TypeError, ValueError):
        return None
    return cost if math.isfinite(cost) else None


def _component_dicts(value) -> list[dict]:
    """Keep only dict entries from an agent's predicted_components."""
    return [c for c in value if isinstance(c, dict)] if isinstance(value, list) else []


def load_dataset(path: str | Path) -> list[dict]:
    """Load a JSONL dataset."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _visible_name(component: dict) -> str:
    """Agent-visible component name. Never falls back to SMILES, which are withheld."""
    if component.get("from_step") is not None:
        return f"product of step {component['from_step']}"
    return (component.get("name") or "").strip() or "UNKNOWN_COMPONENT"


def format_amount(component: dict) -> str | None:
    """The laboratory amount of a component as shown to an agent ("1.234 g", "56.7 mg").

    A noisy description (from quantity noise) wins; solvents and unknown amounts give None.
    Five significant figures keep every derived quantity within 0.005%.
    """
    if component.get("quantity_description"):
        return str(component["quantity_description"])
    grams = component.get("amount_g")
    if grams is None or component.get("role") == "solvent":
        return None
    return f"{grams:.5g} g" if grams >= 1 else f"{grams * 1000:.5g} mg"


def _product_label(product: dict | None) -> str | None:
    """How a product is identified to an agent: its name, or its molecular formula when no
    verified name exists."""
    product = product or {}
    if product.get("display_name"):
        return product["display_name"]
    return f"compound of formula {product['formula']}" if product.get("formula") else None


def _product_mass(product: dict | None, yield_percent) -> str | None:
    """Isolated product mass of the laboratory run (hidden when the yield is unknown)."""
    grams = (product or {}).get("mass_g")
    if grams is None or yield_percent is None:
        return None
    # five significant figures: the cost is divided by the product mass, so its rounding goes straight
    # into the answer (4 figures would allow 0.05%)
    return f"{grams:.5g} g" if grams >= 1 else f"{grams * 1000:.5g} mg"


def _agent_visible_steps(reaction: dict) -> list[dict] | None:
    """Per-step product, isolated product mass and scale reference of a multi-step route."""
    steps = reaction.get("steps")
    if not steps:
        return None
    return [
        {
            "step_number": st.get("step_number"),
            "product": _product_label(st.get("product")),
            "product_mass": _product_mass(st.get("product"), st.get("yield_percent")),
            "scale_reference": st.get("scale_reference"),
        }
        for st in steps
    ]


def _is_product_component(component: dict) -> bool:
    """Return whether a component represents the target product, not an input."""
    return component.get("role") == "product" or component.get("original_role") == "product"


def _agent_visible_components(reaction: dict) -> list[dict]:
    """Return the component rows that should be visible to an evaluated agent."""
    return [
        component
        for component in reaction.get("components", [])
        if not _is_product_component(component)
    ]


def score_prediction(reaction: dict, result: dict) -> CostPrediction:
    """Score one agent result against its record (cost and component matches).

    A listed component counts as found when it matches the visible name or the component's
    common name (agents often report the familiar name of a compound shown by another name).
    """
    bought = [c for c in _agent_visible_components(reaction)
              if c.get("from_step") is None and c.get("role") != "solvent"]
    true_component_names = [_visible_name(c) for c in bought]
    aliases = [n for c in bought for n in (_visible_name(c), c.get("name_common")) if n]
    matches, seen_names = [], set()
    for pc in _component_dicts(result.get("predicted_components")):
        pc_name = str(pc.get("name") or "")
        key = " ".join(pc_name.lower().split())
        if not key or key in seen_names:  # listing a name twice earns nothing
            continue
        seen_names.add(key)
        found = any(component_names_equivalent(pc_name, alias) for alias in aliases)
        matches.append(ComponentMatch(name=pc_name, found=found))
    return CostPrediction(
        reaction_id=reaction["reaction_id"],
        predicted_cost=_coerce_cost(result.get("predicted_cost_per_gram")),
        true_cost=_ground_truth_cost(reaction),
        predicted_components=matches,
        true_component_names=true_component_names,
    )


def prepare_agent_input(reaction: dict) -> dict:
    """The agent's view of a record: component names and laboratory amounts, the product (by name, or
    by formula when no verified name exists) and the mass of it isolated, and the compound whose
    1 mmol sets the scale. Nothing an agent can work out itself is given: no reaction class, roles,
    structures, equivalents, molecular weights, yields, prices or ids."""
    component_rows = []
    for c in _agent_visible_components(reaction):
        row = {"name": _visible_name(c), "amount": format_amount(c)}
        if c.get("original_step") is not None:
            row["step"] = c["original_step"]
        component_rows.append(row)
    product = reaction.get("product") if isinstance(reaction.get("product"), dict) else {}
    agent_input = {
        "product": _product_label(product),
        "product_mass": _product_mass(product, reaction.get("yield_percent")),
        "scale_reference": reaction.get("scale_reference"),
        "components": component_rows,
    }
    steps = _agent_visible_steps(reaction)
    if steps:
        agent_input["steps"] = steps
    if reaction.get("description"):
        agent_input["description"] = reaction["description"]
    return agent_input


def _component_phrase(comp: dict) -> str:
    """Build a natural-language phrase for a single component."""
    amount = format_amount(comp)
    return f"{amount} of {_visible_name(comp)}" if amount else _visible_name(comp)


def _multistep_description(reaction: dict, steps: list[dict]) -> str:
    """Step-by-step prose for a multi-step route."""
    sentences = [f"The following is a {len(steps)}-step route."]
    rows = _agent_visible_components(reaction)
    for st in steps:
        n = st["step_number"]
        phrases = [_component_phrase(c) for c in rows if c.get("original_step") == n]
        if phrases:
            sentences.append(f"Step {n}: " + "; ".join(phrases) + ".")
        last = n == steps[-1]["step_number"]
        what = "the final product" if last else f"the product of step {n}"
        made = f"{st['product_mass']} of {what}" if st.get("product_mass") else f"an unreported amount of {what}"
        named = f" ({st['product']})" if st.get("product") else ""
        ref = st.get("scale_reference") or "its main substrate"
        sentences.append(f"This gave {made}{named}; the step was run on {ref}.")
    return " ".join(sentences)


def prepare_agent_input_natural_language(reaction: dict) -> dict:
    """The same view as ``prepare_agent_input`` with the components written as prose in a
    ``description`` field (no structured component list). A description produced by format
    noise is kept as it is."""
    view = prepare_agent_input(reaction)
    view.pop("components", None)
    if reaction.get("description"):
        return view
    steps = view.get("steps")
    if steps:
        view["description"] = _multistep_description(reaction, steps)
        return view
    phrases = [_component_phrase(c) for c in _agent_visible_components(reaction)]
    sentences = []
    if phrases:
        joined = phrases[0] if len(phrases) == 1 else ", ".join(phrases[:-1]) + " and " + phrases[-1]
        sentences.append(f"The reaction used {joined}.")
    if view.get("product_mass"):
        sentences.append(f"This gave {view['product_mass']} of {view.get('product') or 'the product'}.")
    elif view.get("product"):
        sentences.append(f"The product is {view['product']} (isolated amount not reported).")
    if view.get("scale_reference"):
        sentences.append(f"The scale reference is {view['scale_reference']}.")
    view["description"] = " ".join(sentences)
    return view


def run_evaluation(
    agent: CostEstimationAgent,
    dataset_path: str | Path,
    output_path: str | Path | None = None,
    max_workers: int = 8,
    start: int = 0,
    end: int | None = None,
    input_format: Literal["structured", "natural_language"] = "structured",
    record_transform: Callable[[dict], dict] | None = None,
    save_trajectories: bool = False,
    run_config: dict | None = None,
) -> BenchmarkResults:
    """Run an agent on the dataset and compute metrics.

    start/end slice the raw dataset lines (0-indexed, end exclusive) before
    filtering for evaluable reactions — useful for staged runs.

    input_format controls how reaction data is presented to the agent:
      - ``"structured"`` (default): tabular dict with a ``components`` list
      - ``"natural_language"``: prose paragraph; ``components`` replaced by
        a ``description`` string

    run_config describes the run (agent, model, provider, input format, noise settings). It is
    stored at the top of the resume cache, and resuming from a cache written with a different
    configuration raises ValueError instead of silently mixing results.

    record_transform is an optional callable applied to each reaction record
    *before* it is passed to ``prepare_agent_input``.  This is used for noise
    injection — the transform modifies component names while preserving the
    ground-truth cost for evaluation.
    """
    _prepare = (
        prepare_agent_input_natural_language
        if input_format == "natural_language"
        else prepare_agent_input
    )
    dataset = load_dataset(dataset_path)
    if end is not None:
        dataset = dataset[start:end]
    elif start:
        dataset = dataset[start:]
    predictions: list[CostPrediction] = []
    n_skipped_missing_truth = 0

    evaluable = []
    for reaction in dataset:
        true_cost = _ground_truth_cost(reaction)
        if true_cost is None or true_cost <= 0:
            n_skipped_missing_truth += 1
        else:
            evaluable.append(reaction)

    trajectories_by_id: dict[str, list[dict]] = {}
    effective_workers = max_workers

    cache_path: Path | None = None
    results_map: dict[str, tuple] = {}
    if output_path:
        cache_path = Path(output_path).with_suffix(".partial.jsonl")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            eval_ids = {r["reaction_id"] for r in evaluable}
            good_lines, n_bad, header = [], 0, None
            with open(cache_path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        n_bad += 1  # e.g. a line cut short by a crash mid-write
                        continue
                    good_lines.append(line)
                    if "_run_config" in rec:
                        header = rec["_run_config"]
                        continue
                    rid = rec.get("reaction_id")
                    if rid not in eval_ids:
                        continue
                    if isinstance(rec.get("result"), dict) and rec["result"].get("_harness_error"):
                        continue  # failed call from an earlier run: retry it
                    reaction = next(r for r in evaluable if r["reaction_id"] == rid)
                    results_map[rid] = (reaction, rec["result"])
                    if rec.get("tool_calls"):
                        trajectories_by_id[rid] = rec["tool_calls"]
            if not good_lines:
                cache_path.unlink()  # empty file (e.g. a run killed before its first write): start fresh
            elif run_config is not None and header != run_config:
                raise ValueError(
                    f"{cache_path} was written by a different or unrecorded run configuration "
                    f"({header}); use another --output or delete it"
                )
            if n_bad and good_lines:
                logger.warning(f"Dropped {n_bad} unreadable line(s) from {cache_path}")
                cache_path.write_text("".join(line + "\n" for line in good_lines))
            logger.info(f"Resumed {len(results_map)} cached reactions from {cache_path}")

    cache_lock = threading.Lock()
    if cache_path is not None and not cache_path.exists():
        with open(cache_path, "w") as fh:
            fh.write(json.dumps({"_run_config": run_config}) + "\n")

    def _append_cache(rid: str, result: dict, tool_calls: list[dict]):
        if cache_path is None:
            return
        line = json.dumps({
            "reaction_id": rid,
            "result": result,
            "tool_calls": tool_calls,
        })
        with cache_lock, open(cache_path, "a") as fh:
            fh.write(line + "\n")

    def _run_one(reaction):
        transformed = record_transform(reaction) if record_transform else reaction
        agent_input = _prepare(transformed)

        reaction_tracker = None
        run_agent = agent
        # ReAct-style agent: clone + wrap TOOL_REGISTRY
        if save_trajectories and hasattr(agent, '_tools_override'):
            from copy import copy

            from ..tools.agent_tools import TOOL_REGISTRY as _base_tools
            run_agent = copy(agent)
            reaction_tracker = ToolTracker()
            run_agent._tools_override = reaction_tracker.wrap_tools(_base_tools)
        # SDK agent: per-instance tool_calls log, needs serial execution
        elif save_trajectories and hasattr(agent, '_last_tool_calls'):
            from copy import copy
            run_agent = copy(agent)
            run_agent._last_tool_calls = []

        try:
            result = run_agent.estimate_cost(agent_input)
            if not isinstance(result, dict):
                raise TypeError(f"agent returned {type(result).__name__}, expected dict")
        except Exception as e:
            logger.error(f"Agent failed on {reaction['reaction_id']}: {e}")
            result = {"predicted_cost_per_gram": None, "predicted_components": [],
                      "_harness_error": f"{type(e).__name__}: {e}"}

        tool_calls = []
        if reaction_tracker is not None:
            for tc in reaction_tracker.get_calls():
                tool_calls.append({
                    "tool_name": tc.tool_name,
                    "arguments": tc.arguments,
                    "result": tc.result,
                    "success": tc.success,
                    "step": tc.step_number,
                })
        elif save_trajectories and hasattr(run_agent, 'tool_calls'):
            tool_calls = list(run_agent.tool_calls)

        return reaction, result, tool_calls

    pending = [r for r in evaluable if r["reaction_id"] not in results_map]
    if pending:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {executor.submit(_run_one, rxn): rxn["reaction_id"] for rxn in pending}
            done = 0
            for future in as_completed(futures):
                reaction, result, tool_calls = future.result()
                rid = reaction["reaction_id"]
                results_map[rid] = (reaction, result)
                if tool_calls:
                    trajectories_by_id[rid] = tool_calls
                if not result.get("_harness_error"):
                    _append_cache(rid, result, tool_calls)
                done += 1
                if done % 10 == 0:
                    logger.info(f"  [{done}/{len(pending)}] reactions completed")

    # Preserve raw agent results for token tracking
    raw_results_by_id: dict[str, dict] = {}
    raw_metric_results: list[dict] = []
    for reaction in evaluable:
        reaction, result = results_map[reaction["reaction_id"]]
        raw_results_by_id[reaction["reaction_id"]] = result
        true_cost = _ground_truth_cost(reaction)
        raw_metric_results.append(
            {
                "reaction_id": reaction["reaction_id"],
                "predicted_cost_per_gram": _coerce_cost(result.get("predicted_cost_per_gram")),
                "true_cost": true_cost,
                "token_usage": result.get("token_usage")
                if isinstance(result.get("token_usage"), dict) else None,
                "predicted_components": _component_dicts(result.get("predicted_components")),
            }
        )

        predictions.append(score_prediction(reaction, result))

    error_ids = sorted(rid for rid, (_, res) in results_map.items() if res.get("_harness_error"))
    if error_ids:
        logger.warning(
            f"{len(error_ids)} reactions failed with agent errors and were scored as abstentions;"
            " rerun with the same output path to retry them"
        )

    results = evaluate(predictions)
    supplementary_metrics = {
        "token_efficiency": token_efficiency(raw_metric_results),
    }

    # Save detailed results
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(
                {
                    "metrics": results.to_dict(),
                    "supplementary_metrics": supplementary_metrics,
                    "n_input_records": len(dataset),
                    "n_skipped_missing_truth": n_skipped_missing_truth,
                    "n_evaluable_records": len(predictions),
                    "n_agent_errors": len(error_ids),
                    "agent_error_ids": error_ids,
                    "predictions": [
                        {
                            "reaction_id": p.reaction_id,
                            "predicted_cost": p.predicted_cost,
                            "true_cost": p.true_cost,
                            "tcre": abs(p.predicted_cost - p.true_cost) / p.true_cost
                            if p.predicted_cost is not None
                            and p.true_cost is not None
                            and p.true_cost > 0
                            else None,
                            "token_usage": raw_results_by_id.get(
                                p.reaction_id, {}
                            ).get("token_usage"),
                            "predicted_components": [
                                {
                                    "name": c.get("name", ""),
                                    "price_per_gram": c.get("price_per_gram"),
                                }
                                for c in _component_dicts(
                                    raw_results_by_id.get(p.reaction_id, {}).get(
                                        "predicted_components"
                                    )
                                )
                            ],
                            **({"tool_calls": trajectories_by_id[p.reaction_id]}
                               if p.reaction_id in trajectories_by_id else {}),
                        }
                        for p in predictions
                    ],
                },
                f,
                indent=2,
            )

    return results
