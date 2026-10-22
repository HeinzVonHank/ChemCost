"""Tests for the cost calculator."""

import pytest

from chemcost.cost_calculator import (
    _identify_limiting_reagent,
    calculate_multistep_procurement_cost,
    calculate_procurement_cost,
)
from chemcost.pricing.price_entry import PriceEntry
from chemcost.pricing.pricing_db import PricingDB

# ── Fixtures ────────────────────────────────────────────────────────────────

def _make_db(tmp_path):
    """Mini pricing DB with 1g pack @$10/g and 5g pack @$6/g for reactant A."""
    db = PricingDB(tmp_path / "p.sqlite")
    chem_id = db.add_chemical(name="reactant_a", smiles="C", mw=16.04)
    for qty, price, purity in [(1.0, 10.0, "99"), (5.0, 30.0, "99")]:
        db.add_price(
            chem_id,
            PriceEntry(
                chemical_name="reactant_a", cas=None, supplier="S",
                quantity_g=qty, price_usd=price, price_per_gram_usd=price / qty,
                purity=purity,
            ),
        )
    return db


def test_identify_limiting_reagent_picks_min_equiv():
    comps = [
        {"name": "A", "role": "reactant", "equivalents": 1.0},
        {"name": "B", "role": "reactant", "equivalents": 1.2},
        {"name": "cat", "role": "catalyst", "equivalents": 0.05},
    ]
    assert _identify_limiting_reagent(comps) == "A"


def test_identify_limiting_reagent_ignores_catalysts():
    comps = [
        {"name": "cat", "role": "catalyst", "equivalents": 0.01},
        {"name": "X", "role": "reactant", "equivalents": 1.0},
    ]
    assert _identify_limiting_reagent(comps) == "X"


def test_procurement_cost_pack_based(tmp_path):
    """At 1 mmol, reactant_a (MW=16.04) requires 0.01604 g → 1g pack at $10."""
    db = _make_db(tmp_path)
    result = calculate_procurement_cost(
        product_mw=16.04,
        yield_percent=100.0,
        components=[
            {"name": "reactant_a", "role": "reactant", "smiles": "C",
             "equivalents": 1.0, "mw": 16.04},
        ],
        db=db,
        scale_mmol=1.0,
    )
    assert result.cost_tier == "pack_based"
    # $10 pack / (16.04 g/mol × 0.001 mol × 1.0) = 10 / 0.01604 ≈ 623.44 $/g
    assert result.procurement_cost_usd_per_g_product == pytest.approx(10 / (16.04 * 0.001), rel=1e-3)
    db.close()


def test_procurement_cost_ignores_price_range_without_pack_quotes(tmp_path):
    """Chemical not in DB remains unpriced even if legacy price_range is present."""
    db = PricingDB(tmp_path / "empty.sqlite")
    result = calculate_procurement_cost(
        product_mw=100.0,
        yield_percent=100.0,
        components=[
            {
                "name": "exotic",
                "role": "reactant",
                "smiles": "CCCCCCC",
                "equivalents": 1.0,
                "mw": 100.0,
                "price_range": {"median": 50.0},
            },
        ],
        db=db,
        scale_mmol=1.0,
    )
    assert result.cost_tier == "unpriced"
    assert result.procurement_cost_usd_per_g_product is None
    assert result.components[0].tier == "unpriced"
    db.close()


def test_procurement_cost_solvent_excluded(tmp_path):
    """Solvents must not contribute to procurement cost."""
    db = _make_db(tmp_path)
    result = calculate_procurement_cost(
        product_mw=16.04,
        yield_percent=100.0,
        components=[
            {"name": "reactant_a", "role": "reactant", "smiles": "C",
             "equivalents": 1.0, "mw": 16.04},
            {"name": "THF", "role": "solvent", "equivalents": 0,
             "mw": 72.11, "price_per_gram_usd": 0.10},
        ],
        db=db,
        scale_mmol=1.0,
    )
    solvent_comps = [c for c in result.components if c.role == "solvent"]
    assert all(c.total_cost_usd == 0.0 for c in solvent_comps)
    db.close()


def test_procurement_cost_unpriced_when_no_data(tmp_path):
    """If a component has no qualifying pack quote, tier must be unpriced."""
    db = PricingDB(tmp_path / "empty.sqlite")
    result = calculate_procurement_cost(
        product_mw=100.0,
        yield_percent=80.0,
        components=[
            {"name": "unknown_chem", "role": "reactant",
             "equivalents": 1.0, "mw": 100.0},
        ],
        db=db,
    )
    assert result.cost_tier == "unpriced"
    assert result.procurement_cost_usd_per_g_product is None
    db.close()


def test_procurement_cost_zero_yield_returns_none(tmp_path):
    db = _make_db(tmp_path)
    result = calculate_procurement_cost(
        product_mw=100.0,
        yield_percent=0,
        components=[
            {"name": "reactant_a", "role": "reactant", "smiles": "C",
             "equivalents": 1.0, "mw": 16.04},
        ],
        db=db,
    )
    assert result.procurement_cost_usd_per_g_product is None
    db.close()


# ── Multi-step procurement cost tests ─────────────────────────────────────

def _make_multistep_db(tmp_path):
    """DB with two chemicals: reactant_a (1g@$10, 5g@$30) and reactant_b (1g@$20)."""
    db = PricingDB(tmp_path / "multi.sqlite")
    cid_a = db.add_chemical(name="reactant_a", smiles="C", mw=100.0)
    for qty, price in [(1.0, 10.0), (5.0, 30.0)]:
        db.add_price(cid_a, PriceEntry(
            chemical_name="reactant_a", cas=None, supplier="S",
            quantity_g=qty, price_usd=price, price_per_gram_usd=price / qty,
            purity="99",
        ))
    cid_b = db.add_chemical(name="reactant_b", smiles="CC", mw=80.0)
    db.add_price(cid_b, PriceEntry(
        chemical_name="reactant_b", cas=None, supplier="S",
        quantity_g=1.0, price_usd=20.0, price_per_gram_usd=20.0,
        purity="99",
    ))
    return db


def test_multistep_two_step_cascade(tmp_path):
    """Step 1 produces intermediate; step 2 uses it at cascaded cost."""
    db = _make_multistep_db(tmp_path)
    steps = [
        {
            "step_number": 1,
            "reaction": {
                "product_smiles": "P1",
                "product_name": "intermediate",
                "product_mw": 100.0,
                "yield_percent": 80.0,
                "components": [
                    {"name": "reactant_a", "smiles": "C", "role": "reactant",
                     "equivalents": 1.0, "mw": 100.0},
                ],
            },
        },
        {
            "step_number": 2,
            "reaction": {
                "product_smiles": "P2",
                "product_name": "final",
                "product_mw": 200.0,
                "yield_percent": 90.0,
                "components": [
                    {"name": "intermediate", "smiles": "P1", "role": "reactant",
                     "equivalents": 1.0, "mw": 100.0},
                    {"name": "reactant_b", "smiles": "CC", "role": "reactant",
                     "equivalents": 1.5, "mw": 80.0},
                ],
            },
        },
    ]
    result = calculate_multistep_procurement_cost(steps=steps, db=db)

    assert result.n_steps == 2
    # Step 1: required_mass = 1.0 * 100 * 0.001 = 0.1g -> 1g pack @ $10
    # cost_per_g_product = $10 / (100 * 0.001 * 0.8) = $10 / 0.08 = $125/g
    step1_cost = result.step_results[0].result.procurement_cost_usd_per_g_product
    assert step1_cost == pytest.approx(125.0, rel=1e-3)

    # Step 2: intermediate costs $125/g, required = 1.0 * 100 * 0.001 = 0.1g -> $12.50
    #          reactant_b: required = 1.5 * 80 * 0.001 = 0.12g -> 1g pack @ $20
    #          total = $12.50 + $20 = $32.50
    #          grams_product = 200 * 0.001 * 0.9 = 0.18g
    #          cost_per_g = $32.50 / 0.18 = $180.556
    step2_cost = result.step_results[1].result.procurement_cost_usd_per_g_product
    assert step2_cost == pytest.approx(32.50 / 0.18, rel=1e-2)

    assert result.procurement_cost_usd_per_g_product == step2_cost
    assert result.cost_tier == "pack_based"
    db.close()


def test_multistep_intermediate_not_from_db(tmp_path):
    """Intermediate should use cascaded cost, not DB price even if available."""
    db = _make_multistep_db(tmp_path)
    # Add intermediate to DB at a different price
    cid_p = db.add_chemical(name="intermediate", smiles="P1", mw=100.0)
    db.add_price(cid_p, PriceEntry(
        chemical_name="intermediate", cas=None, supplier="S",
        quantity_g=1.0, price_usd=999.0, price_per_gram_usd=999.0,
        purity="99",
    ))

    steps = [
        {
            "step_number": 1,
            "reaction": {
                "product_smiles": "P1", "product_name": "intermediate",
                "product_mw": 100.0, "yield_percent": 80.0,
                "components": [
                    {"name": "reactant_a", "smiles": "C", "role": "reactant",
                     "equivalents": 1.0, "mw": 100.0},
                ],
            },
        },
        {
            "step_number": 2,
            "reaction": {
                "product_smiles": "P2", "product_name": "final",
                "product_mw": 200.0, "yield_percent": 90.0,
                "components": [
                    {"name": "intermediate", "smiles": "P1", "role": "reactant",
                     "equivalents": 1.0, "mw": 100.0},
                ],
            },
        },
    ]
    result = calculate_multistep_procurement_cost(steps=steps, db=db)

    # Step 1 cost/g = $125 (same as above)
    # Step 2 should use $125/g for intermediate, NOT $999
    # intermediate cost = 0.1g * $125 = $12.50
    # cost_per_g = $12.50 / (200 * 0.001 * 0.9) = $12.50 / 0.18 ≈ $69.44
    step2_cost = result.step_results[1].result.procurement_cost_usd_per_g_product
    assert step2_cost == pytest.approx(12.50 / 0.18, rel=1e-2)
    db.close()


def test_multistep_price_range_only_external_component_is_unpriced(tmp_path):
    """A legacy price_range cannot price a missing external component."""
    db = _make_multistep_db(tmp_path)
    steps = [
        {
            "step_number": 1,
            "reaction": {
                "product_smiles": "P1", "product_name": "intermediate",
                "product_mw": 100.0, "yield_percent": 80.0,
                "components": [
                    {"name": "reactant_a", "smiles": "C", "role": "reactant",
                     "equivalents": 1.0, "mw": 100.0},
                ],
            },
        },
        {
            "step_number": 2,
            "reaction": {
                "product_smiles": "P2", "product_name": "final",
                "product_mw": 200.0, "yield_percent": 90.0,
                "components": [
                    {"name": "intermediate", "smiles": "P1", "role": "reactant",
                     "equivalents": 1.0, "mw": 100.0},
                    # exotic_chem is not in DB; legacy price_range must not price it.
                    {"name": "exotic_chem", "smiles": "CCCCC", "role": "reactant",
                     "equivalents": 1.0, "mw": 50.0,
                     "price_range": {"median": 100.0}},
                ],
            },
        },
    ]
    result = calculate_multistep_procurement_cost(steps=steps, db=db)
    assert result.cost_tier == "unpriced"
    assert result.procurement_cost_usd_per_g_product is None
    db.close()


def test_multistep_unpriced_propagates(tmp_path):
    """If any external component is unpriced, overall result should be unpriced."""
    db = _make_multistep_db(tmp_path)
    steps = [
        {
            "step_number": 1,
            "reaction": {
                "product_smiles": "P1", "product_name": "intermediate",
                "product_mw": 100.0, "yield_percent": 80.0,
                "components": [
                    # unknown_chem: not in DB
                    {"name": "unknown_chem", "role": "reactant",
                     "equivalents": 1.0, "mw": 100.0},
                ],
            },
        },
    ]
    result = calculate_multistep_procurement_cost(steps=steps, db=db)
    assert result.cost_tier == "unpriced"
    assert result.procurement_cost_usd_per_g_product is None
    db.close()


def test_multistep_solvents_excluded(tmp_path):
    """Solvents in multi-step should be excluded just like single-step."""
    db = _make_multistep_db(tmp_path)
    steps = [
        {
            "step_number": 1,
            "reaction": {
                "product_smiles": "P1", "product_name": "product",
                "product_mw": 100.0, "yield_percent": 100.0,
                "components": [
                    {"name": "reactant_a", "smiles": "C", "role": "reactant",
                     "equivalents": 1.0, "mw": 100.0},
                    {"name": "THF", "role": "solvent", "equivalents": 0, "mw": 72.11},
                ],
            },
        },
    ]
    result = calculate_multistep_procurement_cost(steps=steps, db=db)
    solvent_comps = [c for c in result.step_results[0].result.components if c.role == "solvent"]
    assert all(c.total_cost_usd == 0.0 for c in solvent_comps)
    assert result.cost_tier == "pack_based"
    db.close()


def test_multistep_step_without_product_mw_makes_route_unpriced(tmp_path):
    """A step that cannot be costed must not silently drop out of the route cost."""
    db = _make_multistep_db(tmp_path)
    steps = [
        {"step_number": 1, "reaction": {
            "product_smiles": None, "product_name": "intermediate",
            "product_mw": None, "yield_percent": 80.0,
            "components": [{"name": "reactant_a", "smiles": "C", "role": "reactant",
                            "equivalents": 1.0, "mw": 100.0}]}},
        {"step_number": 2, "reaction": {
            "product_smiles": "P2", "product_name": "final",
            "product_mw": 200.0, "yield_percent": 90.0,
            "components": [{"name": "reactant_b", "smiles": "CC", "role": "reactant",
                            "equivalents": 1.0, "mw": 80.0}]}},
    ]
    result = calculate_multistep_procurement_cost(steps=steps, db=db)
    assert result.cost_tier == "unpriced"
    assert result.procurement_cost_usd_per_g_product is None


def test_multistep_disconnected_route_is_unpriced(tmp_path):
    """If an intermediate is never consumed downstream, the route cost would be incomplete."""
    db = _make_multistep_db(tmp_path)
    steps = [
        {"step_number": 1, "reaction": {
            "product_smiles": "P1", "product_name": "intermediate",
            "product_mw": 100.0, "yield_percent": 80.0,
            "components": [{"name": "reactant_a", "smiles": "C", "role": "reactant",
                            "equivalents": 1.0, "mw": 100.0}]}},
        {"step_number": 2, "reaction": {
            "product_smiles": "P2", "product_name": "final",
            "product_mw": 200.0, "yield_percent": 90.0,
            "components": [{"name": "reactant_b", "smiles": "CC", "role": "reactant",
                            "equivalents": 1.0, "mw": 80.0}]}},
    ]
    result = calculate_multistep_procurement_cost(steps=steps, db=db)
    assert result.cost_tier == "unpriced"
