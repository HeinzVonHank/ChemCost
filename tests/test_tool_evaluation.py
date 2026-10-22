"""Tests for tool-use evaluation framework: tracker, ablation, error analysis."""

from chemcost.evaluation.ablation import (
    ABLATIONS,
    AblationConfig,
    apply_ablation,
    run_ablation,
)
from chemcost.evaluation.error_analysis import (
    ErrorCategory,
    categorize_errors,
    error_distribution,
)
from chemcost.evaluation.metrics import (
    ComponentMatch,
    CostPrediction,
    tool_efficiency,
    tool_usage_summary,
)
from chemcost.evaluation.tool_tracker import ToolTracker, ToolUsageStats

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool_registry():
    """Minimal tool registry for testing."""
    return {
        "search_chemical": {
            "function": lambda query: {"name": query, "smiles": "C"},
            "description": "Search chemical",
            "parameters": {"query": "name"},
        },
        "get_supplier_quotes": {
            "function": lambda smiles_or_name: {"quotes": [], "source": "test"},
            "description": "Get supplier quotes",
            "parameters": {"smiles_or_name": "name, CAS, or SMILES"},
        },
        "calculate": {
            "function": lambda expression: {"result": eval(expression)},
            "description": "Calculator",
            "parameters": {"expression": "math expression"},
        },
    }


def _make_failing_tool_registry():
    """Registry where search_chemical returns an error."""
    reg = _make_tool_registry()
    reg["search_chemical"]["function"] = lambda query: {"error": "not found"}
    return reg


# ---------------------------------------------------------------------------
# ToolTracker tests
# ---------------------------------------------------------------------------

class TestToolTracker:
    def test_wrap_preserves_keys(self):
        tools = _make_tool_registry()
        tracker = ToolTracker()
        wrapped = tracker.wrap_tools(tools)

        for name in tools:
            assert name in wrapped
            assert "description" in wrapped[name]
            assert "parameters" in wrapped[name]
            assert "function" in wrapped[name]

    def test_wrapped_function_returns_same_result(self):
        tools = _make_tool_registry()
        tracker = ToolTracker()
        wrapped = tracker.wrap_tools(tools)

        original = tools["search_chemical"]["function"](query="methane")
        tracked = wrapped["search_chemical"]["function"](query="methane")
        assert original == tracked

    def test_calls_are_logged(self):
        tools = _make_tool_registry()
        tracker = ToolTracker()
        wrapped = tracker.wrap_tools(tools)

        wrapped["search_chemical"]["function"](query="methane")
        wrapped["get_supplier_quotes"]["function"](smiles_or_name="methane")

        calls = tracker.get_calls()
        assert len(calls) == 2
        assert calls[0].tool_name == "search_chemical"
        assert calls[0].arguments == {"query": "methane"}
        assert calls[0].success is True
        assert calls[1].tool_name == "get_supplier_quotes"

    def test_error_result_tracked_as_failure(self):
        tools = _make_failing_tool_registry()
        tracker = ToolTracker()
        wrapped = tracker.wrap_tools(tools)

        wrapped["search_chemical"]["function"](query="unknown")
        calls = tracker.get_calls()
        assert len(calls) == 1
        assert calls[0].success is False

    def test_stats_basic(self):
        tools = _make_tool_registry()
        tracker = ToolTracker()
        wrapped = tracker.wrap_tools(tools)

        tracker.mark_reaction_boundary()
        wrapped["search_chemical"]["function"](query="A")
        wrapped["get_supplier_quotes"]["function"](smiles_or_name="A")
        tracker.mark_reaction_boundary()
        wrapped["search_chemical"]["function"](query="B")
        wrapped["get_supplier_quotes"]["function"](smiles_or_name="B")

        stats = tracker.get_stats()
        assert stats.total_calls == 4
        assert stats.calls_per_tool["search_chemical"] == 2
        assert stats.calls_per_tool["get_supplier_quotes"] == 2
        assert stats.avg_calls_per_reaction == 2.0  # 4 calls / 2 reactions
        assert stats.redundant_calls == 0

    def test_redundant_calls_detected(self):
        tools = _make_tool_registry()
        tracker = ToolTracker()
        wrapped = tracker.wrap_tools(tools)

        # Call search_chemical with same args twice
        wrapped["search_chemical"]["function"](query="methane")
        wrapped["search_chemical"]["function"](query="methane")

        stats = tracker.get_stats()
        assert stats.redundant_calls == 1

    def test_tool_sequence(self):
        tools = _make_tool_registry()
        tracker = ToolTracker()
        wrapped = tracker.wrap_tools(tools)

        wrapped["search_chemical"]["function"](query="A")
        wrapped["get_supplier_quotes"]["function"](smiles_or_name="A")
        wrapped["calculate"]["function"](expression="1+1")

        stats = tracker.get_stats()
        assert stats.tool_sequence == ["search_chemical", "get_supplier_quotes", "calculate"]

    def test_reset(self):
        tools = _make_tool_registry()
        tracker = ToolTracker()
        wrapped = tracker.wrap_tools(tools)

        wrapped["search_chemical"]["function"](query="A")
        assert len(tracker.get_calls()) == 1

        tracker.reset()
        assert len(tracker.get_calls()) == 0
        stats = tracker.get_stats()
        assert stats.total_calls == 0


# ---------------------------------------------------------------------------
# Ablation tests
# ---------------------------------------------------------------------------

class TestAblation:
    def test_apply_ablation_full(self):
        tools = _make_tool_registry()
        config = AblationConfig("full", [], "All tools")
        result = apply_ablation(tools, config)
        assert set(result.keys()) == set(tools.keys())

    def test_apply_ablation_removes_tools(self):
        tools = _make_tool_registry()
        config = AblationConfig("no_search", ["search_chemical"], "No search")
        result = apply_ablation(tools, config)
        assert "search_chemical" not in result
        assert "get_supplier_quotes" in result
        assert "calculate" in result

    def test_apply_ablation_no_tools(self):
        tools = _make_tool_registry()
        config = AblationConfig(
            "no_tools",
            ["search_chemical", "get_supplier_quotes", "calculate"],
            "Zero-shot",
        )
        result = apply_ablation(tools, config)
        assert len(result) == 0

    def test_predefined_ablations_valid(self):
        all_tool_names = {
            "search_chemical",
            "get_supplier_quotes",
            "calculate",
        }
        for config in ABLATIONS:
            assert isinstance(config.name, str)
            assert isinstance(config.disabled_tools, list)
            for tool in config.disabled_tools:
                assert tool in all_tool_names, f"Unknown tool {tool} in {config.name}"

    def test_apply_ablation_search_only(self):
        tools = _make_tool_registry()
        config = AblationConfig(
            "search_only",
            ["get_supplier_quotes", "calculate"],
            "Only search",
        )
        result = apply_ablation(tools, config)
        assert set(result.keys()) == {"search_chemical"}

    def test_run_ablation_uses_current_component_matching(self):
        dataset = [
            {
                "reaction_id": "rxn-1",
                "reaction_name": "Test",
                "procurement_cost_usd_per_g_product": 10.0,
                "components": [
                    {"name": "triethylamine", "role": "reactant"},
                    {"name": "DMF", "role": "solvent"},
                    {"name": "product", "role": "product"},
                ],
            }
        ]

        class DummyAgent:
            def estimate_cost(self, reaction):
                return {
                    "predicted_cost_per_gram": "10.0",
                    "predicted_components": [{"name": "Et3N", "price_per_gram": 5.0}, "junk"],
                }

        results = run_ablation(
            agent_factory=lambda tools: DummyAgent(),
            dataset=dataset,
            ablations=[AblationConfig("full", [], "All tools")],
            tools=_make_tool_registry(),
        )

        prediction = results.runs[0].predictions[0]
        assert prediction.true_component_names == ["triethylamine"]
        assert prediction.predicted_components[0].found is True
        assert prediction.predicted_cost == 10.0


# ---------------------------------------------------------------------------
# Error analysis tests
# ---------------------------------------------------------------------------

class TestErrorAnalysis:
    def test_no_errors_for_perfect_prediction(self):
        pred = CostPrediction(
            reaction_id="r1",
            predicted_cost=10.0,
            true_cost=10.0,
            predicted_components=[
                ComponentMatch(name="A", found=True, price_pred=5.0, price_true=5.0),
            ],
            true_component_names=["A"],
        )
        gt = {"components": [{"name": "A", "role": "reactant", "price_per_gram_usd": 5.0}]}
        errors = categorize_errors(pred, gt)
        assert errors == []

    def test_missing_component_detected(self):
        pred = CostPrediction(
            reaction_id="r1",
            predicted_cost=5.0,
            true_cost=10.0,
            predicted_components=[],
            true_component_names=["A", "B"],
        )
        gt = {"components": [
            {"name": "A", "role": "reactant"},
            {"name": "B", "role": "reactant"},
        ]}
        errors = categorize_errors(pred, gt)
        assert ErrorCategory.MISSING_COMPONENT in errors
        # Two components missing
        assert errors.count(ErrorCategory.MISSING_COMPONENT) == 2

    def test_extra_component_detected(self):
        pred = CostPrediction(
            reaction_id="r1",
            predicted_cost=10.0,
            true_cost=10.0,
            predicted_components=[
                ComponentMatch(name="A", found=True, price_pred=5.0, price_true=5.0),
                ComponentMatch(name="X", found=False, price_pred=1.0, price_true=None),
            ],
            true_component_names=["A"],
        )
        gt = {"components": [{"name": "A", "role": "reactant", "price_per_gram_usd": 5.0}]}
        errors = categorize_errors(pred, gt)
        assert ErrorCategory.EXTRA_COMPONENT in errors

    def test_wrong_price_detected(self):
        pred = CostPrediction(
            reaction_id="r1",
            predicted_cost=10.0,
            true_cost=10.0,
            predicted_components=[
                ComponentMatch(name="A", found=True, price_pred=10.0, price_true=5.0),
            ],
            true_component_names=["A"],
        )
        gt = {"components": [{"name": "A", "role": "reactant", "price_per_gram_usd": 5.0}]}
        errors = categorize_errors(pred, gt)
        assert ErrorCategory.WRONG_PRICE in errors

    def test_chemical_not_found_detected(self):
        pred = CostPrediction(
            reaction_id="r1",
            predicted_cost=5.0,
            true_cost=10.0,
            predicted_components=[
                ComponentMatch(name="A", found=False, price_pred=None, price_true=None),
            ],
            true_component_names=["A"],
        )
        gt = {"components": [{"name": "A", "role": "reactant"}]}
        errors = categorize_errors(pred, gt)
        # "A" is in true_component_names AND in predicted_components but found=False
        assert ErrorCategory.CHEMICAL_NOT_FOUND in errors

    def test_calculation_error_detected(self):
        """If components match but final cost is way off, it's a calculation error."""
        pred = CostPrediction(
            reaction_id="r1",
            predicted_cost=20.0,
            true_cost=10.0,
            predicted_components=[
                ComponentMatch(name="A", found=True, price_pred=5.0, price_true=5.0),
            ],
            true_component_names=["A"],
        )
        gt = {"components": [{"name": "A", "role": "reactant", "price_per_gram_usd": 5.0}]}
        errors = categorize_errors(pred, gt)
        assert ErrorCategory.CALCULATION_ERROR in errors

    def test_error_distribution(self):
        preds = [
            CostPrediction(
                reaction_id="r1",
                predicted_cost=5.0,
                true_cost=10.0,
                predicted_components=[],
                true_component_names=["A"],
            ),
            CostPrediction(
                reaction_id="r2",
                predicted_cost=5.0,
                true_cost=10.0,
                predicted_components=[],
                true_component_names=["B", "C"],
            ),
        ]
        gts = [
            {"components": [{"name": "A", "role": "reactant"}]},
            {"components": [{"name": "B", "role": "reactant"}, {"name": "C", "role": "reactant"}]},
        ]
        dist = error_distribution(preds, gts)
        # 3 total missing components across both reactions
        assert dist.get("missing_component", 0) == 3

    def test_alias_match_is_not_marked_missing_or_extra(self):
        pred = CostPrediction(
            reaction_id="r1",
            predicted_cost=10.0,
            true_cost=10.0,
            predicted_components=[
                ComponentMatch(name="Et3N", found=True, price_pred=5.0, price_true=5.0),
            ],
            true_component_names=["triethylamine"],
        )
        gt = {
            "components": [
                {
                    "name": "triethylamine",
                    "role": "reactant",
                    "price_per_gram_usd": 5.0,
                }
            ]
        }
        errors = categorize_errors(pred, gt)
        assert ErrorCategory.MISSING_COMPONENT not in errors
        assert ErrorCategory.EXTRA_COMPONENT not in errors


# ---------------------------------------------------------------------------
# Metrics additions tests
# ---------------------------------------------------------------------------

class TestMetricsAdditions:
    def test_tool_efficiency_basic(self):
        preds = [
            CostPrediction("r1", predicted_cost=2.0, true_cost=2.0),  # correct
            CostPrediction("r2", predicted_cost=5.0, true_cost=2.0),  # wrong
        ]
        # 1 correct out of 10 tool calls
        eff = tool_efficiency(preds, total_tool_calls=10, tolerance_k=25)
        assert eff == 0.1

    def test_tool_efficiency_zero_calls(self):
        preds = [CostPrediction("r1", predicted_cost=2.0, true_cost=2.0)]
        assert tool_efficiency(preds, total_tool_calls=0) == 0.0

    def test_tool_efficiency_all_correct(self):
        preds = [
            CostPrediction("r1", predicted_cost=2.0, true_cost=2.0),
            CostPrediction("r2", predicted_cost=2.0, true_cost=2.0),
        ]
        eff = tool_efficiency(preds, total_tool_calls=4, tolerance_k=25)
        assert eff == 0.5  # 2 correct / 4 calls

    def test_tool_usage_summary_empty(self):
        result = tool_usage_summary([])
        assert result["total_calls"] == 0

    def test_tool_usage_summary_aggregates(self):
        s1 = ToolUsageStats(
            total_calls=5,
            calls_per_tool={"search_chemical": 3, "get_supplier_quotes": 2},
            success_rate_per_tool={"search_chemical": 1.0, "get_supplier_quotes": 0.5},
            avg_calls_per_reaction=5.0,
            redundant_calls=1,
            tool_sequence=[
                "search_chemical",
                "get_supplier_quotes",
                "search_chemical",
                "get_supplier_quotes",
                "search_chemical",
            ],
        )
        s2 = ToolUsageStats(
            total_calls=3,
            calls_per_tool={"search_chemical": 1, "calculate": 2},
            success_rate_per_tool={"search_chemical": 1.0, "calculate": 1.0},
            avg_calls_per_reaction=3.0,
            redundant_calls=0,
            tool_sequence=["search_chemical", "calculate", "calculate"],
        )
        result = tool_usage_summary([s1, s2])
        assert result["total_calls"] == 8
        assert result["calls_per_tool"]["search_chemical"] == 4
        assert result["calls_per_tool"]["get_supplier_quotes"] == 2
        assert result["calls_per_tool"]["calculate"] == 2
        assert result["total_redundant_calls"] == 1
        assert result["mean_calls_per_reaction"] == 4.0  # (5+3)/2
