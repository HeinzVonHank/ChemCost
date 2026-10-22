# ChemCost

A benchmark for chemical reaction procurement-cost estimation by LLM agents.

Given a reaction as it was run in the laboratory (component names and amounts used, the product's molecular formula, and the yield), an agent must return the US-dollar cost of buying enough material to run the reaction at 1 mmol of a named scale reference compound, normalized per gram of product. Supplier prices, structures, equivalents, and molecular weights are withheld, so solving a reaction requires identifying each chemical from a deliberately unfamiliar name, deriving the stoichiometry from the laboratory amounts, retrieving supplier quotes, choosing the right packs, and aggregating cost. Ground truth is computed deterministically from a frozen supplier-quote snapshot, so evaluation is judge-free.

- **Dataset:** [huggingface.co/datasets/YoungerWu/ChemCost](https://huggingface.co/datasets/YoungerWu/ChemCost)
- **Paper:** _link to be added_

The dataset contains single-step reactions and multi-step synthesis routes, each with a deterministic procurement-cost label, plus a frozen database of pack-level supplier quotes (a frozen 2026 snapshot).

## Installation

Python 3.10 or newer.

```bash
git clone https://github.com/HeinzVonHank/ChemCost.git
cd ChemCost
pip install -e ".[eval,tests]"
python scripts/download_data.py      # dataset + pricing DB into data/
python scripts/verify_labels.py      # recompute every label from the frozen DB
```

`download_data.py` writes `data/chemcost.jsonl` and `data/pricing_db.sqlite`. Set `CHEMCOST_PRICING_DB` to use a database at a different path.

## Running an evaluation

```bash
export ANTHROPIC_API_KEY=...
python scripts/run_evaluation.py \
  --agent react \
  --provider anthropic \
  --model claude-sonnet-4-6 \
  --output results/react_sonnet46.json
```

Agents: `react` (tool use), `sdk` (Claude Agent SDK with the same tools), `codex` (OpenAI models through the Codex CLI and its login, with every built-in Codex tool disabled and only the benchmark tools exposed over MCP; `--disable-tools` and `--tool-budget` set harder variants), and the no-tool baselines `zero-shot` and `cot`. All prompts are zero-shot: they define the task but contain no worked examples or calculation recipe. Providers: `anthropic`, `openai`, `qwen` (`DASHSCOPE_API_KEY`), `deepseek`, `kimi`, `glm` (`ZHIPU_API_KEY`), `openrouter`; each reads `<PROVIDER>_API_KEY` from the environment unless noted. With `react`, OpenAI Codex models and gpt-5.5 or later run on the Responses API (function tools with low reasoning effort); other OpenAI models use Chat Completions.

Noise settings:

```bash
python scripts/run_evaluation.py --agent react \
  --provider anthropic --model claude-sonnet-4-6 \
  --noise medium --noise-stages name,quantity,missing,format
```

From Python:

```python
from chemcost.baselines.react_agent import ReActAgent
from chemcost.evaluation.evaluator import run_evaluation

agent = ReActAgent(model="claude-sonnet-4-6", provider="anthropic")
result = run_evaluation(agent, "data/chemcost.jsonl", "results/react.json")
print(result.to_dict())
```

## Task and tools

ReAct agents get four tools:

- `get_supplier_quotes(smiles_or_name)`: every supplier quote for the chemical in the frozen pricing database, unfiltered: `(supplier, pack, price_usd, purity)` with pack sizes in mg, g, or kg and purity as reported (null when not reported).
- `calculate(expression)`: arithmetic only (`+ - * / **`, parentheses, `sqrt log log10 exp abs round`); parsed with an AST whitelist, never executed as code.

Agents receive only the view produced by `chemcost.evaluation.evaluator.prepare_agent_input`: each component's name and the amount used in the laboratory (in g or mg, five significant figures; none for solvents), the product by name (or its molecular formula) and the mass of it that was isolated, and the scale reference: the compound whose 1 mmol defines the costed run. There is no name-search tool: the agent has to work out each structure from its name. Names are systematic names or synonyms rather than the most familiar name, and the structure leads to the label's quotes. `get_supplier_quotes` returns every supplier quote as listed (all purities, unreported purity as null, pack sizes in mg, g, or kg); the agent decides which offers count under the task's purity rule. For multi-step routes the view also gives each component's step and each step's product name and isolated mass and its scale reference; intermediates appear as "product of step k". Nothing an agent can work out itself is given: no reaction class, no component roles, no equivalents, no yields (they follow from the isolated mass), and no molecular weights (no tool returns them either). The prompts are zero-shot: they define the task (scale, whole packs in any combination of sizes, solvents not bought, per gram of product) but give no formula, procedure, or worked example, so an agent must work out which components are solvents, the molecular weights, the stoichiometry, and the pack arithmetic itself. Record ids are not shown, but the dataset itself is public: an agent with web or Hugging Face access could still find a record by its component names, so evaluate without such access (the bundled agents only have the two benchmark tools). The raw records also contain SMILES, equivalents, molecular weights, selected packs, and the label, so do not pass raw records to an agent.

## Ground-truth rule

For each non-solvent component *i*, the required mass is *m_i = e_i · M_i · 10⁻³* g (one equivalent = 1 mmol; equivalents are used as given; *M* is the average molecular weight). Its purchase cost is the cheapest way to buy at least *m_i* in whole packs of any combination of sizes (and suppliers): *q_i = min Σ_p n_p · price_p* over non-negative integers *n_p* with *Σ_p n_p · size_p ≥ m_i*, over the qualifying packs *p* (for each size, its cheapest qualifying offer). A pack qualifies if its purity is reported and is at least 95% (fractions such as 0.98 are read as 98%; a supplier that gives one identical purity on every quote, across at least 10 chemicals, is treated as not reporting purity; reagents whose ordinary commercial grade is below 95% use that grade instead: potassium hydroxide ≥ 85%, sodium hydride ≥ 55% (60% dispersion), and mCPBA ≥ 70%) and it is not an isotope-labeled or reference material. Solvents are excluded. The label is *Σ q_i / (M_product · 10⁻³ · yield/100)*. There is no per-gram price fallback. Multi-step routes cascade forward: each step is costed at 1 mmol scale, and an intermediate is charged at the per-gram cost of the step that produces it. A route is included only if every step can be costed and every intermediate is used by a later step.

The implementation is `chemcost.cost_calculator.calculate_procurement_cost`; `scripts/verify_labels.py` reproduces every released label from the database.

## Metrics

The headline metric is **CTA@k**: the fraction of reactions whose predicted cost is within *k*% of ground truth, with abstentions counted as failures. We report CTA@10 and CTA@25. Component precision and recall compare the agent's listed purchases with the non-solvent components that must be bought (intermediates and solvents excluded), using an alias-aware name match; because component names are part of the input, they measure bookkeeping rather than chemical resolution and should be read alongside CTA.

## Reference results

All 1,427 records, `--agent codex` (Codex CLI 0.162.1 with a ChatGPT login, every built-in Codex tool disabled, only `get_supplier_quotes` and `calculate` exposed, zero-shot prompt, no tool-call limit), run on 2026-10-22:

| Model | CTA@10 | CTA@25 | CTA@10 single-step (1,359) | CTA@10 multi-step (68) |
|---|---|---|---|---|
| gpt-5.6-luna | 64.2% | 72.7% | 65.3% | 41.2% |
| gpt-6-luna | 47.5% | 57.7% | 48.8% | 22.1% |

```bash
python scripts/run_evaluation.py --dataset data/chemcost.jsonl --agent codex --model gpt-5.6-luna --workers 20 --output results/codex.json
```

## Repository layout

```
src/chemcost/
  baselines/         ReAct, Agent SDK, zero-shot and CoT agents
  evaluation/        evaluator, metrics, tool tracker, ablation utilities
  noise/             four-stage noise injection (name, quantity, missing, format)
  pricing/           read-only interface to the frozen pricing SQLite
  tools/             agent tools
  cost_calculator.py ground-truth cost rule
scripts/             download_data.py, run_evaluation.py, verify_labels.py
tests/               unit tests (pytest)
```

## License

Code: MIT. Dataset and pricing database: CC BY-SA 4.0, because most reactions are derived from the Open Reaction Database (CC BY-SA 4.0). See `LICENSE`. The CC BY-SA 4.0 grant covers the ChemCost compilation: the records, labels, and the selection and structure of the pricing database. Individual supplier prices are factual listings published by the suppliers; ChemCost claims no rights in them and includes them as a frozen research snapshot. The pricing snapshot is frozen and is not intended for real procurement decisions.

## Citation

```bibtex
@inproceedings{chemcost2026,
  title     = {ChemCost: A Benchmark for Chemical Reaction Procurement Cost Estimation by LLM Agents},
  author    = {TODO},
  booktitle = {TODO},
  year      = {2026}
}
```
