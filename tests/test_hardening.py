"""Regression tests for agent-facing tool safety and evaluator robustness."""

import json
import math

import pytest

from chemcost.evaluation.evaluator import _coerce_cost, run_evaluation
from chemcost.tools.agent_tools import TOOL_REGISTRY, calculate


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("1+2*3", 7.0),
        ("0.5*180.95*0.001", 0.090475),
        ("sqrt(16)+log(e)", 5.0),
        ("2**10", 1024.0),
        ("-3/4", -0.75),
        ("round(3.14159, 2)", 3.14),
    ],
)
def test_calculate_arithmetic(expression, expected):
    assert calculate(expression)["result"] == pytest.approx(expected)


@pytest.mark.parametrize(
    "expression",
    [
        "().__class__.__base__.__subclasses__()",
        "abs.__self__",
        "__import__('os')",
        "x + 1",
        "[1][0]",
        "'a' * 3",
        "9**9**9",
        "10**400",
        "exp(1000)",
        "1/0",
        "lambda: 1",
        "1" * 600,
    ],
)
def test_calculate_rejects_unsafe_or_unbounded(expression):
    assert "error" in calculate(expression)


def test_tool_registry_is_the_two_documented_tools():
    assert sorted(TOOL_REGISTRY) == ["calculate", "get_supplier_quotes"]  # no name search: names are reasoned out


def test_no_tool_hands_out_molecular_weights():
    for info in TOOL_REGISTRY.values():
        assert "molecular weight" not in info["description"].lower()


@pytest.mark.parametrize(
    "value, expected",
    [(12.5, 12.5), ("12.5", 12.5), (3, 3.0), (None, None), ("abc", None), ([1], None),
     ({}, None), (True, None), (float("nan"), None), (float("inf"), None)],
)
def test_coerce_cost(value, expected):
    assert _coerce_cost(value) == expected


def test_malformed_agent_output_does_not_crash_evaluation(tmp_path):
    record = {
        "reaction_id": "R1",
        "reaction_name": "test",
        "type": "single_step",
        "product": {"name": "p", "smiles": "C", "mw": 100.0},
        "yield_percent": 100.0,
        "components": [{"name": "a", "role": "reactant", "equivalents": 1.0, "mw": 50.0}],
        "procurement_cost_usd_per_g_product": 10.0,
    }
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("\n".join(json.dumps({**record, "reaction_id": f"R{i}"}) for i in range(4)))

    outputs = iter([
        {"predicted_cost_per_gram": "not a number", "predicted_components": "oops"},
        {"predicted_cost_per_gram": float("nan"), "predicted_components": [1, None]},
        {"predicted_cost_per_gram": [10.0], "predicted_components": [{"name": 3}]},
        {"predicted_cost_per_gram": 10.0, "predicted_components": [{"name": "a"}]},
    ])

    class Agent:
        def estimate_cost(self, reaction):
            return next(outputs)

    result = run_evaluation(Agent(), dataset, tmp_path / "out.json", max_workers=1)
    metrics = result.to_dict()
    assert math.isclose(metrics["cta@10"], 0.25)


def test_empty_component_name_never_reveals_smiles():
    from chemcost.evaluation.evaluator import (
        prepare_agent_input,
        prepare_agent_input_natural_language,
    )

    smiles = "Brc1ccc2c(ccn2Cc2ccccc2)c1"
    record = {
        "reaction_id": "R1",
        "reaction_name": "test",
        "product": {"name": "p", "smiles": "C", "mw": 100.0},
        "yield_percent": 50.0,
        "components": [
            {"name": "", "smiles": smiles, "role": "reactant", "equivalents": 1.0, "mw": 288.2},
            {"name": None, "smiles": "CCO", "role": "solvent", "equivalents": 1.0, "mw": 46.1},
        ],
    }
    structured = json.dumps(prepare_agent_input(record))
    natural = json.dumps(prepare_agent_input_natural_language(record))
    for text in (structured, natural):
        assert smiles not in text
        assert "CCO" not in text
        assert "UNKNOWN_COMPONENT" in text


def test_supplier_quotes_match_equivalent_smiles(tmp_path):
    from datetime import date

    from chemcost.pricing.price_entry import PriceEntry
    from chemcost.pricing.pricing_db import PricingDB

    db = PricingDB(tmp_path / "p.sqlite")
    cid = db.add_chemical("bromobenzene", smiles="c1ccc(Br)cc1", mw=157.01)
    db.add_price(cid, PriceEntry(chemical_name="bromobenzene", cas=None, supplier="S",
                                 catalog_number=None, quantity_g=5.0, price_usd=10.0,
                                 price_per_gram_usd=2.0, purity="98", grade=None, url=None,
                                 snapshot_date=date(2026, 3, 9)))
    for query in ("c1ccc(Br)cc1", "C1=CC=C(C=C1)Br", "Brc1ccccc1"):
        assert db.get_pack_quotes(smiles=query), query
    assert not db.get_pack_quotes(smiles="c1ccc(Cl)cc1")


MULTISTEP_RECORD = {
    "reaction_id": "R-MS",
    "reaction_name": "Multi-step synthesis route (test)",
    "type": "multi_step",
    "product": {"name": "final", "smiles": "CCCCO", "mw": 74.12},
    "yield_percent": 56.0,
    "components": [
        {"name": "aldehyde", "smiles": "CCC=O", "role": "reactant", "equivalents": 1.0,
         "mw": 58.08, "original_step": 1, "amount_g": 0.1162},
        {"name": "secret intermediate", "smiles": "CCCCO[Si](C)(C)C", "role": "reactant",
         "equivalents": 1.0, "mw": 146.3, "original_step": 2, "from_step": 1, "amount_g": 0.2926},
    ],
    "steps": [
        {"step_number": 1, "product": {"name": "secret intermediate", "formula": "C7H18OSi",
                                       "display_name": "butoxy(trimethyl)silane", "mass_g": 0.2341,
                                       "smiles": "CCCCO[Si](C)(C)C", "mw": 146.3},
         "yield_percent": 80.0, "scale_reference": "aldehyde", "components": []},
        {"step_number": 2, "product": {"name": "final", "smiles": "CCCCO", "mw": 74.12, "formula": "C4H10O",
                                       "mass_g": 0.1038},
         "yield_percent": 70.0, "scale_reference": "product of step 1", "components": []},
    ],
}


def test_multistep_agent_view_exposes_steps_but_not_identities():
    from chemcost.evaluation.evaluator import (
        prepare_agent_input,
        prepare_agent_input_natural_language,
    )

    view = prepare_agent_input(MULTISTEP_RECORD)
    assert [s["product"] for s in view["steps"]] == ["butoxy(trimethyl)silane", "compound of formula C4H10O"]
    assert [s["product_mass"] for s in view["steps"]] == ["234.1 mg", "103.8 mg"]
    assert all("yield_percent" not in s for s in view["steps"])  # the yield is derived, not given
    assert [s["scale_reference"] for s in view["steps"]] == ["aldehyde", "product of step 1"]
    assert [c["amount"] for c in view["components"]] == ["116.2 mg", "292.6 mg"]
    assert [c["step"] for c in view["components"]] == [1, 2]
    assert view["components"][1]["name"] == "product of step 1"
    prose = prepare_agent_input_natural_language(MULTISTEP_RECORD)["description"]
    assert "Step 1" in prose and "Step 2" in prose and "234.1 mg" in prose and "% yield" not in prose
    for text in (json.dumps(view), prose):
        assert "secret intermediate" not in text
        assert "CCCCO" not in text


def test_read_only_db_refuses_missing_path_and_writes(tmp_path):
    import sqlite3

    from chemcost.pricing.pricing_db import PricingDB

    with pytest.raises(FileNotFoundError):
        PricingDB(tmp_path / "missing.sqlite", read_only=True)
    PricingDB(tmp_path / "p.sqlite").close()
    db = PricingDB(tmp_path / "p.sqlite", read_only=True)
    with pytest.raises(sqlite3.OperationalError):
        db.conn.execute("INSERT INTO chemicals (name) VALUES ('x')")


def test_resume_cache_rejects_other_configuration(tmp_path):
    record = {
        "reaction_id": "R1", "reaction_name": "t", "type": "single_step",
        "product": {"name": "p", "smiles": "C", "mw": 100.0}, "yield_percent": 100.0,
        "components": [{"name": "a", "role": "reactant", "equivalents": 1.0, "mw": 50.0}],
        "procurement_cost_usd_per_g_product": 10.0,
    }
    dataset = tmp_path / "d.jsonl"
    dataset.write_text(json.dumps(record))

    class Agent:
        def estimate_cost(self, reaction):
            return {"predicted_cost_per_gram": 10.0, "predicted_components": []}

    out = tmp_path / "o.json"
    run_evaluation(Agent(), dataset, out, max_workers=1, run_config={"noise": "none"})
    run_evaluation(Agent(), dataset, out, max_workers=1, run_config={"noise": "none"})
    with pytest.raises(ValueError):
        run_evaluation(Agent(), dataset, out, max_workers=1, run_config={"noise": "medium"})


def _one_record_dataset(tmp_path):
    record = {
        "reaction_id": "R1", "reaction_name": "t", "type": "single_step",
        "product": {"name": "p", "smiles": "C", "mw": 100.0}, "yield_percent": 100.0,
        "components": [{"name": "a", "role": "reactant", "equivalents": 1.0, "mw": 50.0}],
        "procurement_cost_usd_per_g_product": 10.0,
    }
    dataset = tmp_path / "d.jsonl"
    dataset.write_text(json.dumps(record))
    return dataset


class _FixedAgent:
    calls = 0

    def estimate_cost(self, reaction):
        type(self).calls += 1
        return {"predicted_cost_per_gram": 10.0, "predicted_components": []}


def test_resume_survives_truncated_cache_line(tmp_path):
    dataset, out = _one_record_dataset(tmp_path), tmp_path / "o.json"
    run_evaluation(_FixedAgent(), dataset, out, max_workers=1, run_config={"noise": "none"})
    cache = out.with_suffix(".partial.jsonl")
    with open(cache, "a") as fh:
        fh.write('{"reaction_id": "R1", "resu')  # crash mid-write
    _FixedAgent.calls = 0
    run_evaluation(_FixedAgent(), dataset, out, max_workers=1, run_config={"noise": "none"})
    assert _FixedAgent.calls == 0  # the complete line was reused
    assert all(json.loads(line) for line in cache.read_text().splitlines())


def test_resume_refuses_cache_without_matching_header(tmp_path):
    dataset, out = _one_record_dataset(tmp_path), tmp_path / "o.json"
    run_evaluation(_FixedAgent(), dataset, out, max_workers=1)  # Python API, no run_config
    with pytest.raises(ValueError):
        run_evaluation(_FixedAgent(), dataset, out, max_workers=1, run_config={"noise": "none"})


@pytest.mark.parametrize(
    "text, expected",
    [
        ("cost per gram of step 1 product: $5.00 ... final cost per gram: $900", 900.0),
        ("Cost per gram (USD): 1,234.50", 1234.5),
        ("**Cost per gram** = $12.5", 12.5),
        ("the cost per gram of step 2 is unclear", None),
        (None, None),
        ("", None),
    ],
)
def test_plain_text_answer_parsing(text, expected):
    from chemcost.baselines.react_agent import ReActAgent

    parsed = ReActAgent._try_parse_final_answer(text)
    assert (parsed or {}).get("predicted_cost_per_gram") == expected


def test_read_only_db_path_with_uri_characters(tmp_path):
    from chemcost.pricing.pricing_db import PricingDB

    path = tmp_path / "we#ird?dir%20" / "p.sqlite"
    PricingDB(path).close()
    db = PricingDB(path, read_only=True)
    assert db.conn.execute("SELECT count(*) FROM chemicals").fetchone()[0] == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == ["we#ird?dir%20"]


def test_multistep_yield_noise_hides_step_yields():
    from chemcost.baselines.react_agent import build_components_text_for_prompt
    from chemcost.evaluation.evaluator import prepare_agent_input
    from chemcost.noise.noise_injector import inject_missing_info_noise

    hidden = False
    for seed in range(20):
        noisy = inject_missing_info_noise(MULTISTEP_RECORD, noise_level="high", seed=seed)
        view = prepare_agent_input(noisy)
        masses = [s["product_mass"] for s in view["steps"]]
        if None in masses:  # a dropped yield hides that step's isolated mass
            hidden = True
            assert "None" not in build_components_text_for_prompt(view)
    assert hidden


def test_model_error_field_is_not_a_harness_failure(tmp_path):
    class Agent:
        def estimate_cost(self, reaction):
            return {"predicted_cost_per_gram": 10.0, "predicted_components": [], "error": "one MW differed"}

    dataset, out = _one_record_dataset(tmp_path), tmp_path / "o.json"
    run_evaluation(Agent(), dataset, out, max_workers=1, run_config={"noise": "none"})
    report = json.loads(out.read_text())
    assert report["n_agent_errors"] == 0
    assert len(out.with_suffix(".partial.jsonl").read_text().splitlines()) == 2  # header + result


def test_empty_resume_cache_starts_fresh(tmp_path):
    dataset, out = _one_record_dataset(tmp_path), tmp_path / "o.json"
    out.with_suffix(".partial.jsonl").write_text("")
    run_evaluation(_FixedAgent(), dataset, out, max_workers=1, run_config={"noise": "none"})
    assert json.loads(out.read_text())["metrics"]["n_predicted"] == 1


@pytest.mark.parametrize(
    "text, expected",
    [
        ('Draft {"predicted_cost_per_gram": 10} ... final {"predicted_cost_per_gram": 20}', 20.0),
        ("cost per gram: 1.2e3", 1200.0),
    ],
)
def test_answer_parser_takes_final_answer(text, expected):
    from chemcost.baselines.react_agent import ReActAgent

    assert ReActAgent._try_parse_final_answer(text)["predicted_cost_per_gram"] == expected


def test_tiny_catalyst_amount_survives_prose():
    from chemcost.evaluation.evaluator import format_amount

    assert format_amount({"role": "catalyst", "amount_g": 0.0000123}) == "0.0123 mg"
    assert format_amount({"role": "catalyst", "amount_g": 0.012345678}) == "12.346 mg"


def test_openai_responses_loop_runs_tools_and_parses_answer():
    from types import SimpleNamespace as NS

    from chemcost.baselines.react_agent import ReActAgent, uses_responses_api

    assert uses_responses_api("gpt-5.3-codex") and uses_responses_api("gpt-5.5")
    assert not uses_responses_api("gpt-5.4") and not uses_responses_api("gpt-4.1")

    calls = []

    class Responses:
        def create(self, **kw):
            calls.append(kw)
            usage = NS(input_tokens=10, output_tokens=5)
            if len(calls) == 1:  # first turn: one tool call
                item = NS(type="function_call", call_id="c1", name="calculate", arguments='{"expression": "2*3"}')
                return NS(id="r1", output=[item], output_text="", usage=usage)
            return NS(id="r2", output=[NS(type="message")], usage=usage,
                      output_text='{"predicted_cost_per_gram": 6.0, "predicted_components": []}')

    agent = ReActAgent.__new__(ReActAgent)
    agent.model, agent.max_steps, agent._client = "gpt-5.5", 5, NS(responses=Responses())
    out = agent._run_openai_responses("prompt", {"calculate": TOOL_REGISTRY["calculate"]})
    assert out["predicted_cost_per_gram"] == 6.0
    assert out["token_usage"]["total_tokens"] == 30
    second = calls[1]
    assert second["previous_response_id"] == "r1"
    assert json.loads(second["input"][0]["output"])["result"] == 6.0


def test_supplier_quotes_are_raw_and_unfiltered(tmp_path, monkeypatch):
    from datetime import date

    from chemcost.pricing.price_entry import PriceEntry
    from chemcost.pricing.pricing_db import PricingDB
    from chemcost.tools import agent_tools

    db = PricingDB(tmp_path / "p.sqlite")
    cid = db.add_chemical("bromobenzene", smiles="c1ccc(Br)cc1", mw=157.01)
    for qty, price, purity in [(0.25, 9.0, "97"), (5.0, 10.0, "90"), (5.0, 12.0, None), (2500.0, 300.0, "0.99")]:
        db.add_price(cid, PriceEntry(chemical_name="bromobenzene", cas=None, supplier="S", catalog_number=None,
                                     quantity_g=qty, price_usd=price, price_per_gram_usd=price / qty, purity=purity,
                                     grade=None, url=None, snapshot_date=date(2026, 3, 9)))
    monkeypatch.setattr(agent_tools, "_get_pricing_db", lambda: db)
    agent_tools._get_supplier_quotes_cached.cache_clear()
    quotes = agent_tools.get_supplier_quotes("c1ccc(Br)cc1")["quotes"]
    agent_tools._get_supplier_quotes_cached.cache_clear()
    assert [q["pack"] for q in quotes] == ["250 mg", "5 g", "5 g", "2.5 kg"]
    assert [q["purity"] for q in quotes] == ["97", "90", None, "0.99"]  # nothing filtered or normalised
    assert db.get_pack_quotes(smiles="c1ccc(Br)cc1")[-1]["quantity_g"] == 2500.0  # the label rule still filters
    assert len(db.get_pack_quotes(smiles="c1ccc(Br)cc1")) == 2


def test_pubchem_throttling_is_retried_not_reported(monkeypatch):
    from types import SimpleNamespace as NS

    from chemcost.tools import agent_tools

    calls = []

    class Client:
        def get(self, url):
            calls.append(url)
            if len(calls) < 3:
                return NS(status_code=429, headers={"retry-after": "0"})
            return NS(status_code=200, headers={}, raise_for_status=lambda: None,
                      json=lambda: {"PropertyTable": {"Properties": [{"SMILES": "CCO", "IUPACName": "ethanol"}]}})

    monkeypatch.setattr(agent_tools, "_get_http_client", lambda: Client())
    monkeypatch.setattr(agent_tools, "_PUBCHEM_MIN_INTERVAL_S", 0.0)
    out = json.loads(agent_tools._search_chemical_uncached("ethanol"))
    assert out == {"name": "ethanol", "smiles": "CCO"} and len(calls) == 3


def test_codex_session_exposes_only_the_benchmark_tools():
    from chemcost.baselines.codex_agent import CodexBaseline

    agent = CodexBaseline(model="gpt-5.6-luna", tools=["get_supplier_quotes"], tool_budget=8)
    agent.codex = "codex"
    cmd = " ".join(agent._command("/tmp/x", "/tmp/x/a.txt"))
    features = ("shell_tool", "unified_exec", "standalone_web_search", "multi_agent", "browser_use", "computer_use")
    for feature in features:
        assert f"--disable {feature}" in cmd
    assert "-s read-only" in cmd and "--ignore-user-config" in cmd and 'web_search="disabled"' in cmd
    assert 'CHEMCOST_TOOLS="get_supplier_quotes"' in cmd and 'CHEMCOST_TOOL_BUDGET="8"' in cmd
    assert "tools.calculate.approval_mode" not in cmd
    assert "No calculator is available" in agent._instructions() and "at most 8 tool calls" in agent._instructions()
