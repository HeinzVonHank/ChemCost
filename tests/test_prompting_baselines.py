"""Tests for the zero-shot prompts (CoT, zero-shot, ReAct) and answer parsing.

These tests verify instantiation, prompt construction, and JSON parsing
without making actual LLM API calls.
"""

import re

import pytest

from chemcost.baselines.prompting_baselines import COT_SYSTEM_PROMPT, CoTBaseline, _build_user_prompt
from chemcost.baselines.react_agent import SYSTEM_PROMPT, TASK_DEFINITION, ReActAgent, build_task_prompt

# ---------------------------------------------------------------------------
# Fixture: sample reaction in the format produced by prepare_agent_input
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_reaction():
    return {
        "reaction_name": "Suzuki Coupling",
        "product": "1,1'-biphenyl",
        "product_mass": "693.9 mg",
        "scale_reference": "bromobenzene",
        "components": [
            {"name": "bromobenzene", "role": "reactant", "amount": "785 mg"},
            {"name": "phenylboronic acid", "role": "reactant", "amount": "792.5 mg"},
            {"name": "tetrakis(triphenylphosphine)palladium(0)", "role": "catalyst", "amount": "288.9 mg"},
            {"name": "potassium carbonate", "role": "base", "amount": "1.382 g"},
            {"name": "DMF", "role": "solvent", "amount": None},
        ],
    }


class TestInstantiation:
    def test_cot_baseline_default(self):
        agent = CoTBaseline()
        assert agent.provider == "anthropic"

    def test_cot_baseline_openai(self):
        agent = CoTBaseline(model="gpt-4.1", provider="openai")
        assert agent.model == "gpt-4.1"


class TestPromptsAreZeroShot:
    @pytest.mark.parametrize("prompt", [SYSTEM_PROMPT, COT_SYSTEM_PROMPT, TASK_DEFINITION])
    def test_no_worked_example_or_recipe(self, prompt):
        lowered = prompt.lower()
        assert "example" not in lowered
        assert "ceil" not in lowered and "× 0.001" not in prompt and "* 0.001" not in prompt
        assert "equivalents ×" not in prompt and "step-by-step procedure" not in lowered

    def test_task_definition_states_the_scale_and_purchase_rule(self):
        assert "1 mmol of the scale reference" in TASK_DEFINITION
        assert "any combination of pack sizes" in TASK_DEFINITION and "Solvents are not bought" in TASK_DEFINITION


class TestPromptConstruction:
    def test_task_prompt_shows_amounts_not_equivalents(self, sample_reaction):
        prompt = build_task_prompt(sample_reaction)
        assert "785 mg" in prompt and "1.382 g" in prompt
        assert "Product: 1,1'-biphenyl" in prompt and "693.9 mg" in prompt and "Scale reference: bromobenzene" in prompt
        assert "yield" not in prompt.lower()
        assert not re.search(r"equiv|MW:|g/mol", prompt)

    def test_build_user_prompt_cot_flag(self, sample_reaction):
        assert "step by step" in _build_user_prompt(sample_reaction, include_cot=True)
        assert "step by step" not in _build_user_prompt(sample_reaction, include_cot=False)

    def test_build_user_prompt_asks_for_json_without_tools(self, sample_reaction):
        prompt = _build_user_prompt(sample_reaction)
        assert "predicted_cost_per_gram" in prompt and "No tools are available" in prompt

    def test_prose_description_replaces_component_table(self, sample_reaction):
        prompt = build_task_prompt({**sample_reaction, "description": "Use 785 mg of bromobenzene ..."})
        assert "Use 785 mg of bromobenzene" in prompt and "| role:" not in prompt


class TestParsing:
    def test_parse_valid_json(self):
        text = '{"predicted_cost_per_gram": 42.5, "predicted_components": []}'
        result = ReActAgent._try_parse_final_answer(text)
        assert result is not None
        assert result["predicted_cost_per_gram"] == 42.5

    def test_parse_json_embedded_in_text(self):
        text = (
            "After careful analysis, the total cost is:\n\n"
            '{"predicted_cost_per_gram": 150.3, '
            '"predicted_components": [{"name": "bromobenzene", "price_per_gram": 0.4}]}'
            "\n\nThat concludes my estimate."
        )
        result = ReActAgent._try_parse_final_answer(text)
        assert result is not None
        assert result["predicted_cost_per_gram"] == 150.3
        assert len(result["predicted_components"]) == 1

    def test_parse_default_to_null(self):
        """When text has no parseable cost, _try_parse returns None."""
        assert ReActAgent._try_parse_final_answer("I have no idea") is None


def test_cot_has_estimate_cost():
    assert callable(CoTBaseline().estimate_cost)
