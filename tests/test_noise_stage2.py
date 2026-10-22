"""Quantity noise: laboratory amounts re-expressed as approximate masses, volumes, mmol, mol% or vague text."""

import copy

import pytest

from chemcost.noise.noise_injector import (
    _amount_to_mmol_str,
    _amount_to_volume_str,
    _equiv_to_mol_percent_str,
    _get_density,
    _make_approximate_str,
    _make_vague_description,
    inject_noise,
    inject_quantity_noise,
)


def _comp(name="benzene", role="reactant", equiv=1.0, mw=100.0, scale=2.0):
    return {"name": name, "role": role, "equivalents": equiv, "mw": mw,
            "amount_g": None if equiv is None else equiv * scale * mw / 1000}


def _record(components, cost=10.0):
    return {"reaction_id": "QN-TEST-001", "reaction_name": "Quantity Noise Test", "components": components,
            "procurement_cost_usd_per_g_product": cost, "yield_percent": 80.0}


class TestHelpers:
    def test_known_densities(self):
        assert _get_density("THF") == pytest.approx(0.889, abs=0.01)
        assert _get_density("dichloromethane") > 1.3
        assert _get_density("unknown stuff") == 1.0

    def test_mol_percent(self):
        assert _equiv_to_mol_percent_str(0.05) == "5 mol%"
        assert _equiv_to_mol_percent_str(0.025) == "2.5 mol%"
        assert _equiv_to_mol_percent_str(0.0001) == "0.01 mol%"

    def test_volume_and_mmol(self):
        assert _amount_to_volume_str(0.889, "THF") == "1.00 mL"
        assert _amount_to_volume_str(0.0000889, "THF").endswith("µL")
        assert _amount_to_mmol_str(0.2, 100.0) == "2 mmol"

    def test_approximate_keeps_two_significant_figures(self):
        import random

        text = _make_approximate_str(0.12345, random.Random(0))
        assert text.endswith("120 mg") and text[0] in "~acA"
        assert _make_approximate_str(1.2345, random.Random(0)).endswith("1.2 g")

    def test_vague(self):
        import random

        rng = random.Random(0)
        assert "excess" in _make_vague_description(8.0, "reagent", rng)
        assert _make_vague_description(0.05, "catalyst", rng) in ("catalytic amount", "cat.")
        assert _make_vague_description(1.0, "reactant", rng) is None


class TestInjectQuantityNoise:
    def test_catalyst_becomes_mol_percent_of_the_reference(self):
        rec = _record([_comp(), _comp("Pd(OAc)2", "catalyst", 0.05, 224.5)])
        cat = next(n["components"][1] for seed in range(50)
                   if "quantity_description" in (n := inject_quantity_noise(rec, noise_level="low", seed=seed))
                   ["components"][1])
        assert cat["quantity_description"] == "5 mol%" and cat["amount_g"] is None
        assert cat["original_amount_g"] == pytest.approx(0.02245)

    def test_medium_rewrites_amounts_without_revealing_equivalents(self):
        comps = [_comp(f"reagent {i}", "reagent", 1.0 + i / 10, 80.0 + i) for i in range(30)]
        noisy = inject_quantity_noise(_record(comps), noise_level="medium", seed=3)
        kinds = {c.get("quantity_noise_kind") for c in noisy["components"]}
        assert {"mmol", "approximate"} <= kinds
        for c in noisy["components"]:
            if c.get("quantity_description"):
                assert "equiv" not in c["quantity_description"]

    def test_liquids_with_known_density_can_switch_to_volume(self):
        comps = [_comp("THF", "reagent", 1.0 + i / 10, 72.1) for i in range(30)]
        noisy = inject_quantity_noise(_record(comps), noise_level="medium", seed=5)
        assert any(c.get("quantity_noise_kind") == "unit_switch" and "mL" in c["quantity_description"]
                   for c in noisy["components"])

    def test_density_uses_the_common_name(self):
        comps = [{**_comp("oxolane", "reagent", 1.0 + i / 10, 72.1), "name_common": "THF"} for i in range(30)]
        noisy = inject_quantity_noise(_record(comps), noise_level="medium", seed=5)
        assert any(c.get("quantity_noise_kind") == "unit_switch" for c in noisy["components"])

    def test_high_gives_vague_text_for_extremes(self):
        rec = _record([_comp("NaH", "reagent", 8.0, 24.0)])
        texts = {inject_quantity_noise(rec, noise_level="high", seed=s)["components"][0].get("quantity_description")
                 for s in range(30)}
        assert any(t and "excess" in t for t in texts)

    def test_solvents_and_missing_amounts_are_skipped(self):
        comps = [_comp("DMF", "solvent"), _comp("x", "reactant", None)]
        noisy = inject_quantity_noise(_record(comps), noise_level="high", seed=1)
        assert all("quantity_description" not in c for c in noisy["components"])

    def test_reproducible_and_pure(self):
        rec = _record([_comp("Pd(OAc)2", "catalyst", 0.05, 224.5), _comp("a", "reagent", 1.5, 90.0)])
        before = copy.deepcopy(rec)
        assert inject_quantity_noise(rec, "medium", 7) == inject_quantity_noise(rec, "medium", 7)
        assert rec == before

    def test_ground_truth_preserved(self):
        noisy = inject_quantity_noise(_record([_comp("Pd(OAc)2", "catalyst", 0.05, 224.5)], 99.9), "high", 1)
        assert noisy["procurement_cost_usd_per_g_product"] == 99.9


class TestInjectNoiseWithQuantity:
    def test_quantity_stage_through_inject_noise(self):
        rec = _record([_comp("Pd(OAc)2", "catalyst", 0.05, 224.5)])
        texts = {inject_noise(rec, noise_types=["quantity"], noise_level="low", seed=s)["components"][0]
                 .get("quantity_description") for s in range(50)}
        assert "5 mol%" in texts

    def test_seed_reproducibility(self):
        rec = _record([_comp("a", "reagent", 1.5, 90.0), _comp("THF", "reagent", 2.0, 72.1)])
        kwargs = {"noise_types": ["quantity", "name_variation"], "noise_level": "medium", "seed": 11}
        assert inject_noise(rec, **kwargs) == inject_noise(rec, **kwargs)


def test_name_variation_never_restores_the_common_name():
    from chemcost.noise.noise_injector import inject_name_variation

    rec = _record([{**_comp("ethanoic acid", "reagent"), "name_common": "acetic acid"}])
    for seed in range(20):
        name = inject_name_variation(rec, noise_level="high", seed=seed)["components"][0]["name"]
        assert name.lower() != "acetic acid"
