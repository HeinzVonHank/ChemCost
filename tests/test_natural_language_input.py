"""The agent views (structured and prose): names, laboratory amounts, formula, scale reference only."""

import json
import re

import pytest

from chemcost.evaluation.evaluator import (
    _component_phrase,
    format_amount,
    prepare_agent_input,
    prepare_agent_input_natural_language,
)

SUZUKI_REACTION = {
    "reaction_id": "CHEMCOST-9001",
    "reaction_name": "Suzuki Coupling",
    "type": "single_step",
    "product": {"name": "biphenyl", "smiles": "c1ccc(-c2ccccc2)cc1", "mw": 154.21, "formula": "C12H10",
                "display_name": "1,1'-biphenyl", "mass_g": 0.69395},
    "yield_percent": 90.0,
    "scale_reference": "bromobenzene",
    "lab_scale_mmol": 5.0,
    "components": [
        {"name": "bromobenzene", "name_common": "PhBr", "smiles": "Brc1ccccc1", "role": "reactant",
         "equivalents": 1.0, "mw": 157.01, "amount_g": 0.78505},
        {"name": "benzeneboronic acid", "name_common": "phenylboronic acid", "smiles": "OB(O)c1ccccc1",
         "role": "reactant", "equivalents": 1.3, "mw": 121.93, "amount_g": 0.792545},
        {"name": "tetrakis(triphenylphosphane)palladium", "name_common": "Pd(PPh3)4", "smiles": "[Pd]",
         "role": "catalyst", "equivalents": 0.05, "mw": 1155.56, "amount_g": 0.28889},
        {"name": "dipotassium carbonate", "name_common": "K2CO3", "smiles": "O=C([O-])[O-].[K+].[K+]",
         "role": "reagent", "equivalents": 2.0, "mw": 138.21, "amount_g": 1.3821},
        {"name": "DMF", "smiles": "CN(C)C=O", "role": "solvent", "equivalents": None, "mw": 73.09},
    ],
    "procurement_cost_usd_per_g_product": 123.45,
}


def _texts(record):
    return [json.dumps(prepare_agent_input(record)), json.dumps(prepare_agent_input_natural_language(record))]


def test_format_amount():
    assert format_amount({"role": "reactant", "amount_g": 1.3821}) == "1.3821 g"
    assert format_amount({"role": "reactant", "amount_g": 0.000123456}) == "0.12346 mg"
    assert format_amount({"role": "solvent", "amount_g": 1.0}) is None
    assert format_amount({"role": "reactant", "amount_g": 0.5, "quantity_description": "~0.5 g"}) == "~0.5 g"


def test_structured_view_has_amounts_formula_and_scale_reference():
    view = prepare_agent_input(SUZUKI_REACTION)
    assert view["product"] == "1,1'-biphenyl" and view["product_mass"] == "693.95 mg"
    assert view["scale_reference"] == "bromobenzene"
    rows = {c["name"]: c for c in view["components"]}
    assert rows["bromobenzene"]["amount"] == "785.05 mg"
    assert rows["dipotassium carbonate"]["amount"] == "1.3821 g"
    assert rows["DMF"]["amount"] is None
    assert set(view) == {"product", "product_mass", "scale_reference", "components"}  # no yield: derivable
    assert all(set(c) == {"name", "amount"} for c in view["components"])  # no roles: inferable


def test_prose_view_has_every_component_and_no_table():
    view = prepare_agent_input_natural_language(SUZUKI_REACTION)
    assert "components" not in view
    text = view["description"]
    for c in SUZUKI_REACTION["components"]:
        assert c["name"] in text
    assert "785.05 mg" in text and "693.95 mg of 1,1'-biphenyl" in text and "scale reference is bromobenzene" in text
    assert "yield" not in text and "C12H10" not in text


@pytest.mark.parametrize("hidden", ["equivalents", "\"mw\"", "g/mol", "157.01", "1.3 ", "Brc1ccccc1", "123.45",
                                    "CHEMCOST-9001", "PhBr", "Pd(PPh3)4", "lab_scale", "Suzuki",
                                    "reactant", "catalyst", "solvent", "\"role\""])
def test_views_hide_structures_equivalents_prices_and_common_names(hidden):
    for text in _texts(SUZUKI_REACTION):
        assert hidden not in text


def test_unnamed_component_does_not_reveal_smiles():
    rec = {**SUZUKI_REACTION, "components": [{"name": "", "smiles": "CCOCC(=O)Cl", "role": "reactant",
                                              "equivalents": 1.0, "mw": 122.5, "amount_g": 0.1225}]}
    for text in _texts(rec):
        assert "CCOCC" not in text and "UNKNOWN_COMPONENT" in text


def test_missing_yield_is_not_invented():
    rec = {**SUZUKI_REACTION, "yield_percent": None}
    assert "yield" not in prepare_agent_input_natural_language(rec)["description"]


def test_quantity_description_replaces_amount_in_both_views():
    comps = [dict(c) for c in SUZUKI_REACTION["components"]]
    comps[0] = {**comps[0], "amount_g": None, "quantity_description": "~0.79 g"}
    rec = {**SUZUKI_REACTION, "components": comps}
    for text in _texts(rec):
        assert "~0.79 g" in text and "785.05 mg" not in text


def test_product_rows_are_not_inputs():
    rec = {**SUZUKI_REACTION, "components": SUZUKI_REACTION["components"]
           + [{"name": "biphenyl", "role": "product", "smiles": "c1ccc(-c2ccccc2)cc1"}]}
    assert all("biphenyl" not in c["name"] for c in prepare_agent_input(rec)["components"])


def test_component_phrase():
    assert _component_phrase(SUZUKI_REACTION["components"][0]) == "785.05 mg of bromobenzene"
    assert _component_phrase(SUZUKI_REACTION["components"][4]) == "DMF"


def test_no_number_in_views_reveals_an_equivalent():
    numbers = set()
    for text in _texts(SUZUKI_REACTION):
        numbers |= set(re.findall(r"\d+(?:\.\d+)?", text))
    assert not {"1.3", "0.05", "2.0"} & numbers
