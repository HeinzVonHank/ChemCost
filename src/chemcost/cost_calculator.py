"""Core cost calculation logic for producing 1g of product."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chemcost.pricing.pricing_db import PricingDB

# Default scale for procurement cost: one equivalent = 1 mmol.
STANDARD_SCALE_MMOL: float = 1.0


@dataclass
class ComponentProcurement:
    """Procurement breakdown for a single reaction component."""

    name: str
    role: str
    equivalents: float
    mw: float  # g/mol
    required_mass_g: float  # at the chosen reaction scale
    total_cost_usd: float | None  # actual purchase cost
    cost_per_g: float | None  # effective $/g of the pack(s) purchased
    pack_quantity_g: float | None  # total grams purchased
    n_packs: int | None  # number of packs
    supplier: str | None
    tier: str  # "pack_based" | "unpriced" | "solvent_excluded"


@dataclass
class ProcurementResult:
    """Result of a procurement-oriented cost calculation."""

    product_mw: float
    yield_percent: float
    scale_mmol: float
    limiting_reagent: str  # name of the identified limiting reagent
    components: list[ComponentProcurement] = field(default_factory=list)
    procurement_cost_usd_per_g_product: float | None = None
    cost_tier: str = "unpriced"  # "pack_based" | "unpriced"


def _identify_limiting_reagent(components: list[dict]) -> str:
    """Return the name of the limiting reagent.

    Rule: the reactant with the lowest equivalents value.
    If multiple reactants are tied, the first listed is chosen.
    If there are no reactants, the first non-solvent component is used.
    """
    def _safe_equiv(c):
        try:
            return float(c.get("equivalents") or 0)
        except (TypeError, ValueError):
            return 0
    reactants = [c for c in components if c.get("role") == "reactant" and _safe_equiv(c) > 0]
    if reactants:
        return min(reactants, key=_safe_equiv).get("name", "unknown")
    non_solvent = [c for c in components if c.get("role") != "solvent"]
    if non_solvent:
        return non_solvent[0].get("name", "unknown")
    return "unknown"


def calculate_procurement_cost(
    product_mw: float,
    yield_percent: float,
    components: list[dict],
    db: "PricingDB",
    scale_mmol: float = STANDARD_SCALE_MMOL,
) -> ProcurementResult:
    """Calculate the procurement cost of a reaction at a fixed scale.

    Cost model rules (v2):
    - **Scale**: one equivalent = scale_mmol mmol; equivalents are used exactly as given.
    - **Limiting reagent**: reported for reference only (reactant with the minimum equivalents).
    - **Required mass**: equiv_i × MW_i × scale_mmol × 0.001 grams per component.
    - **Price lookup**: among qualifying packs (purity ≥ 95%, or a COMMERCIAL_GRADES grade), the cheapest way to buy
      at least required_mass_g in whole packs of any sizes (pricing_db.cheapest_purchase).
      If no qualifying pack quote is present in the DB, the component is unpriced.
    - **Solvents**: excluded from cost entirely.
    - **Yield**: affects denominator (grams of product produced) only, not purchasing decisions.
    - **Procurement cost per gram of product**:
        total_purchase_usd / (product_mw × scale_mmol × 0.001 × yield / 100)

    Cost tiers:
    - "pack_based" – every non-solvent component is priced from pack quotes
    - "unpriced"   – at least one non-solvent component has no qualifying pack quote
    """
    limiting_reagent = _identify_limiting_reagent(components)
    result = ProcurementResult(
        product_mw=product_mw,
        yield_percent=yield_percent,
        scale_mmol=scale_mmol,
        limiting_reagent=limiting_reagent,
    )

    if product_mw <= 0 or yield_percent <= 0:
        result.procurement_cost_usd_per_g_product = None
        result.cost_tier = "unpriced"
        return result

    comp_results: list[ComponentProcurement] = []
    for c in components:
        role = c.get("role", "reactant")

        if role == "solvent":
            comp_results.append(
                ComponentProcurement(
                    name=c.get("name", ""),
                    role=role,
                    equivalents=c.get("equivalents", 0),
                    mw=c.get("mw", 0),
                    required_mass_g=0.0,
                    total_cost_usd=0.0,
                    cost_per_g=None,
                    pack_quantity_g=None,
                    n_packs=None,
                    supplier=None,
                    tier="solvent_excluded",
                )
            )
            continue

        equiv = c.get("equivalents") or 0
        if equiv <= 0:
            mol_pct = c.get("mol_percent") or 0
            if mol_pct > 0:
                equiv = mol_pct / 100
            else:
                continue

        mw = c.get("mw") or 0
        if mw <= 0:
            continue

        required_mass_g = equiv * mw * scale_mmol * 0.001

        procurement = db.get_procurement_price(
            required_mass_g=required_mass_g,
            smiles=c.get("smiles") or None,
            name=c.get("name") or None,
        )

        comp_results.append(
            ComponentProcurement(
                name=c.get("name", ""),
                role=role,
                equivalents=equiv,
                mw=mw,
                required_mass_g=required_mass_g,
                total_cost_usd=procurement["total_cost_usd"],
                cost_per_g=procurement["cost_per_g"],
                pack_quantity_g=procurement["quantity_g"],
                n_packs=procurement["n_packs"],
                supplier=procurement["supplier"],
                tier=procurement["tier"],
            )
        )

    result.components = comp_results

    # Determine overall cost tier and total purchase cost
    costed = [c for c in comp_results if c.tier not in ("solvent_excluded",)]
    tiers = {c.tier for c in costed}

    if "unpriced" in tiers:
        result.cost_tier = "unpriced"
        result.procurement_cost_usd_per_g_product = None
        return result

    total_purchase_usd = sum(c.total_cost_usd for c in costed)
    result.cost_tier = "pack_based"

    grams_product = product_mw * scale_mmol * 0.001 * (yield_percent / 100)
    result.procurement_cost_usd_per_g_product = round(total_purchase_usd / grams_product, 4)
    return result


def _canonical_key(smiles: str | None) -> str | None:
    """Namespaced canonical SMILES used to match intermediates across steps."""
    if not smiles:
        return None
    from chemcost.pricing.pricing_db import _canonical_smiles

    canonical = _canonical_smiles(smiles)
    return f"canonical:{canonical}" if canonical else None


@dataclass
class StepProcurementResult:
    """Procurement result for a single step within a multi-step route."""

    step_number: int
    result: ProcurementResult
    intermediate_cost_per_g: float | None = None  # cost/g of intermediate input used


@dataclass
class MultistepProcurementResult:
    """Result of a multi-step procurement cost calculation."""

    n_steps: int
    step_results: list[StepProcurementResult]
    procurement_cost_usd_per_g_product: float | None = None
    cost_tier: str = "unpriced"


def calculate_multistep_procurement_cost(
    steps: list[dict],
    db: "PricingDB",
    scale_mmol: float = STANDARD_SCALE_MMOL,
) -> MultistepProcurementResult:
    """Calculate procurement cost for a multi-step synthesis via forward cascading.

    Each step is computed at the standard scale (one equivalent = 1 mmol).
    Intermediates produced by earlier steps are priced at their computed
    cost_per_g instead of querying the pricing DB.  External reagents use
    the normal pack-level DB lookup.

    Args:
        steps: Ordered list of step dicts, each with keys:
            reaction.product_smiles, reaction.product_name, reaction.product_mw,
            reaction.yield_percent, reaction.components (same format as single-step).
        db: PricingDB instance.
        scale_mmol: Reaction scale in mmol (default 1.0).

    Returns:
        MultistepProcurementResult with per-step breakdowns and final cost.
    """
    # Map product identifiers -> cost/g from earlier steps
    intermediate_costs: dict[str, float] = {}
    producer_step: dict[str, int] = {}  # intermediate key -> index of the step that makes it
    consumed: set[int] = set()
    step_results: list[StepProcurementResult] = []
    all_tiers: set[str] = set()

    for step_dict in steps:
        rxn = step_dict.get("reaction", step_dict)
        product_mw = rxn.get("product_mw") or 0
        yield_pct = rxn.get("yield_percent") or 0
        components = rxn.get("components", [])
        step_num = step_dict.get("step_number", len(step_results) + 1)

        # Resolve intermediate prices so downstream steps use the prior step's
        # computed cost instead of querying the external pricing database.
        resolved_components = []
        intermediate_used = None
        for comp in components:
            comp_copy = dict(comp)
            smiles = comp_copy.get("smiles") or ""
            name = comp_copy.get("name") or ""

            # Check if this component is an intermediate from a prior step
            matched_key = None
            if smiles and smiles in intermediate_costs:
                matched_key = smiles
            elif name and name in intermediate_costs:
                matched_key = name
            elif smiles and _canonical_key(smiles) in intermediate_costs:
                # Same molecule written as a different SMILES string
                matched_key = _canonical_key(smiles)

            if matched_key is not None:
                # Tag this component so we can handle it specially
                comp_copy["_intermediate_cost_per_g"] = intermediate_costs[matched_key]
                intermediate_used = intermediate_costs[matched_key]
                consumed.add(producer_step[matched_key])

            resolved_components.append(comp_copy)

        # Calculate step cost using the procurement model.
        # For intermediates we inject their cost directly rather than DB lookup.
        step_result = _calculate_step_with_intermediates(
            product_mw=product_mw,
            yield_percent=yield_pct,
            components=resolved_components,
            db=db,
            scale_mmol=scale_mmol,
        )

        step_results.append(StepProcurementResult(
            step_number=step_num,
            result=step_result,
            intermediate_cost_per_g=intermediate_used,
        ))

        # Collect tiers from external (non-intermediate) components
        for c in step_result.components:
            if c.tier not in ("solvent_excluded", "intermediate"):
                all_tiers.add(c.tier)

        # Record this step's product cost for downstream steps
        if step_result.procurement_cost_usd_per_g_product is not None:
            product_smiles = rxn.get("product_smiles", "")
            product_name = rxn.get("product_name", "")
            cost_per_g = step_result.procurement_cost_usd_per_g_product
            for key in (product_smiles, product_name, _canonical_key(product_smiles)):
                if key:
                    intermediate_costs[key] = cost_per_g
                    producer_step[key] = len(step_results) - 1

    # Final result
    final_cost = (
        step_results[-1].result.procurement_cost_usd_per_g_product
        if step_results else None
    )

    # A route is priced only if every step is priced and every non-final step's product
    # feeds a later step; otherwise part of the route's cost would silently drop out.
    every_step_priced = all(
        s.result.procurement_cost_usd_per_g_product is not None for s in step_results
    )
    route_connected = all(i in consumed for i in range(len(step_results) - 1))
    if (
        "unpriced" in all_tiers
        or final_cost is None
        or not every_step_priced
        or not route_connected
    ):
        overall_tier = "unpriced"
        final_cost = None
    else:
        overall_tier = "pack_based"

    return MultistepProcurementResult(
        n_steps=len(step_results),
        step_results=step_results,
        procurement_cost_usd_per_g_product=final_cost,
        cost_tier=overall_tier,
    )


def _calculate_step_with_intermediates(
    product_mw: float,
    yield_percent: float,
    components: list[dict],
    db: "PricingDB",
    scale_mmol: float = STANDARD_SCALE_MMOL,
) -> ProcurementResult:
    """Calculate procurement cost for one step, handling intermediates specially.

    Components with '_intermediate_cost_per_g' are costed at that rate
    (required_mass × cost_per_g) instead of querying the pricing DB.
    All other components use the standard pack-level DB lookup.
    """
    limiting_reagent = _identify_limiting_reagent(components)
    result = ProcurementResult(
        product_mw=product_mw,
        yield_percent=yield_percent,
        scale_mmol=scale_mmol,
        limiting_reagent=limiting_reagent,
    )

    if product_mw <= 0 or yield_percent <= 0:
        result.procurement_cost_usd_per_g_product = None
        result.cost_tier = "unpriced"
        return result

    comp_results: list[ComponentProcurement] = []
    for c in components:
        role = c.get("role", "reactant")

        if role == "solvent":
            comp_results.append(ComponentProcurement(
                name=c.get("name", ""), role=role,
                equivalents=c.get("equivalents", 0), mw=c.get("mw", 0),
                required_mass_g=0.0, total_cost_usd=0.0,
                cost_per_g=None, pack_quantity_g=None,
                n_packs=None, supplier=None, tier="solvent_excluded",
            ))
            continue

        try:
            equiv = float(c.get("equivalents") or 0)
        except (TypeError, ValueError):
            equiv = 0
        if equiv <= 0:
            try:
                mol_pct = float(c.get("mol_percent") or 0)
            except (TypeError, ValueError):
                mol_pct = 0
            if mol_pct > 0:
                equiv = mol_pct / 100
            else:
                continue

        try:
            mw = float(c.get("mw") or 0)
        except (TypeError, ValueError):
            mw = 0
        if mw <= 0:
            continue

        required_mass_g = equiv * mw * scale_mmol * 0.001

        intermediate_cpg = c.get("_intermediate_cost_per_g")
        if intermediate_cpg is not None:
            # Intermediate from a prior step: cost = mass × cost_per_g
            total_cost = required_mass_g * intermediate_cpg
            comp_results.append(ComponentProcurement(
                name=c.get("name", ""), role=role,
                equivalents=equiv, mw=mw,
                required_mass_g=required_mass_g,
                total_cost_usd=round(total_cost, 6),
                cost_per_g=round(intermediate_cpg, 4),
                pack_quantity_g=None, n_packs=None,
                supplier="intermediate", tier="intermediate",
            ))
            continue

        # External reagent: normal DB lookup
        procurement = db.get_procurement_price(
            required_mass_g=required_mass_g,
            smiles=c.get("smiles") or None,
            name=c.get("name") or None,
        )

        comp_results.append(ComponentProcurement(
            name=c.get("name", ""), role=role,
            equivalents=equiv, mw=mw,
            required_mass_g=required_mass_g,
            total_cost_usd=procurement["total_cost_usd"],
            cost_per_g=procurement["cost_per_g"],
            pack_quantity_g=procurement["quantity_g"],
            n_packs=procurement["n_packs"],
            supplier=procurement["supplier"],
            tier=procurement["tier"],
        ))

    result.components = comp_results

    costed = [c for c in comp_results if c.tier not in ("solvent_excluded",)]
    tiers = {c.tier for c in costed if c.tier != "intermediate"}

    if "unpriced" in tiers:
        result.cost_tier = "unpriced"
        result.procurement_cost_usd_per_g_product = None
        return result

    total_purchase_usd = sum(
        c.total_cost_usd for c in costed if c.total_cost_usd is not None
    )

    result.cost_tier = "pack_based"

    grams_product = product_mw * scale_mmol * 0.001 * (yield_percent / 100)
    result.procurement_cost_usd_per_g_product = round(total_purchase_usd / grams_product, 4)
    return result
