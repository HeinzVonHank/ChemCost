"""ReAct agent baseline for ChemCost benchmark."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile

logger = logging.getLogger(__name__)

JSON_ONLY_REPROMPT = (
    "You are in a non-interactive benchmark. Never ask clarifying questions. "
    'Return ONLY one valid JSON object with keys "predicted_cost_per_gram" and '
    '"predicted_components". If you cannot complete the estimate, return '
    '{"predicted_cost_per_gram": null, "predicted_components": []}.'
)


# Newer Claude models reject sampling parameters and always think; thinking counts toward max_tokens.
_NO_SAMPLING_MODEL_PREFIXES = (
    "claude-opus-5", "claude-sonnet-5", "claude-haiku-5", "claude-fable",
    "claude-opus-4-7", "claude-opus-4-8",
)


def anthropic_request_kwargs(model: str, max_tokens: int) -> dict:
    """max_tokens/temperature suited to the Claude model being called."""
    if model.startswith(_NO_SAMPLING_MODEL_PREFIXES):
        return {"max_tokens": max(max_tokens, 16000)}
    return {"max_tokens": max_tokens, "temperature": 0}


def uses_responses_api(model: str) -> bool:
    """OpenAI models that take function tools with reasoning only on the Responses API."""
    return "codex" in model or model.startswith(("gpt-5.5", "gpt-5.6"))


def shown(value) -> str:
    """Render a possibly missing input value for a prompt (numbers to 4 significant figures)."""
    if value is None or value == "":
        return "unknown"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def run_tool(tools: dict, name: str, raw_args) -> dict:
    """Execute one tool call; any problem comes back to the model as an error result."""
    if name not in tools:
        return {"error": f"Unknown tool '{name}'. Available tools: {', '.join(sorted(tools))}"}
    if isinstance(raw_args, str):
        try:
            raw_args = json.loads(raw_args or "{}")
        except json.JSONDecodeError:
            return {"error": "Tool arguments are not valid JSON"}
    if not isinstance(raw_args, dict):
        return {"error": "Tool arguments must be a JSON object"}
    params = set(tools[name]["parameters"])
    args = {k: str(v) for k, v in raw_args.items() if k in params}
    missing = sorted(params - set(args))
    if missing:
        return {"error": f"Missing argument(s) for {name}: {', '.join(missing)}"}
    try:
        return tools[name]["function"](**args)
    except Exception as e:  # tool bugs or bad inputs must not abort the whole reaction
        return {"error": f"{type(e).__name__}: {e}"}


def build_components_text_for_prompt(reaction: dict) -> str:
    """Render the component rows (and, for multi-step routes, the step table)."""
    lines = []
    for c in reaction.get("components", []):
        step = f"[step {c['step']}] " if c.get("step") is not None else ""
        lines.append(f"  - {step}{c['name'] or '(unnamed)'} | amount used: {c.get('amount') or 'not recorded'}")
    steps = reaction.get("steps")
    if steps:
        lines.append("")
        lines.append("Steps:")
        for st in steps:
            lines.append(
                f"  - step {st.get('step_number')}: product {shown(st.get('product'))};"
                f" isolated {shown(st.get('product_mass'))}; scale reference {shown(st.get('scale_reference'))}"
            )
    return "\n".join(lines)


def build_task_prompt(reaction: dict, *, tools: bool = True) -> str:
    """The user message for any agent: the reaction as run in the lab, and what to estimate."""
    body = reaction.get("description") or build_components_text_for_prompt(reaction)
    head = (
        f"Product: {shown(reaction.get('product'))}\n"
        f"Product isolated: {shown(reaction.get('product_mass'))}\n"
        f"Scale reference: {shown(reaction.get('scale_reference'))}\n"
    )
    if reaction.get("steps"):
        head = f"Multi-step route ({len(reaction['steps'])} steps)\n"
    how = ("Work out each chemical's structure from its name, then use the tools to get supplier quotes."
           if tools else "No tools are available: rely on your knowledge of chemical prices.")
    return (
        "Estimate the procurement cost, in US dollars per gram of product, for this reaction.\n\n"
        f"{head}\nComponents as used in the laboratory:\n{body}\n\n{how}\n"
    )


TASK_DEFINITION = """\
## Task
You are given a reaction as it was run in the laboratory: each component's name and the amount used
(not recorded for some components), the product and the mass of it that was isolated, and the scale
reference compound. Estimate what it costs to buy the materials for this reaction, per gram of product.

## Definition of the answer
- Scale: the reaction is run with exactly 1 mmol of the scale reference compound; every other
  component is scaled in proportion to the laboratory amounts.
- Purchases: for each component that has to be bought, buy whole packs (any combination of pack sizes
  and suppliers), the cheapest way that gives at least the amount needed. Solvents are not bought.
- Which offers count: only offers whose purity is reported and is at least 95% (a purity written as
  a fraction, such as 0.98, means 98%). Reagents whose ordinary commercial grade is below that count
  at their grade: potassium hydroxide from 85%, sodium hydride from 55%, mCPBA from 70%.
- Product: the product obtained scales with the run, in proportion to the laboratory run.
- Multi-step routes: every step is run with 1 mmol of that step's scale reference. A component named
  "product of step k" is made in step k, not bought: it costs what step k costs per gram of its product.
  Report the cost per gram of the final product.
- The answer is the total purchase cost divided by the grams of product obtained."""

SYSTEM_PROMPT = TASK_DEFINITION + """

## Tools
- get_supplier_quotes(smiles_or_name): every supplier offer for a chemical as listed (supplier, pack
  size with unit, price_usd, reported purity). These are the only offers you may buy from; not all of
  them count (see the definition above).
- calculate(expression): evaluate a mathematical expression.

This is a non-interactive benchmark: never ask questions and always end with a final answer. If you
cannot complete the estimate, answer {"predicted_cost_per_gram": null, "predicted_components": []}.

Final answer, as JSON only:
{"predicted_cost_per_gram": <number>, "predicted_components": [{"name": <str>, "price_per_gram": <number>}]}
"""

# Stronger variant for models that produce overly conservative final answers
# (only Qwen3-235B-A22B needs this; other models already populate components fully).
SYSTEM_PROMPT_FORCE_COMPONENTS = SYSTEM_PROMPT.rstrip().rstrip('"""').rstrip() + """

ADDITIONAL OUTPUT REQUIREMENT:
- You MUST list every non-solvent component you considered in predicted_components,
  including ones whose price you estimated or could not fully resolve. For unresolved
  components, use your best estimated price_per_gram (e.g., based on a similar
  chemical or chemical class) — do not omit them.
- Even when predicted_cost_per_gram is null, predicted_components must NOT be empty.
"""

class ReActAgent:
    """ReAct-style agent using an LLM with tool calls for cost estimation."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        provider: str = "anthropic",
        max_steps: int | None = None,
        tools: dict | None = None,
    ) -> None:
        self.model = model
        self.provider = provider
        # qwen3 series uses more steps due to verbose reasoning; default higher
        # GLM-4.x defaults to serial (1 tool/step) and needs many more steps.
        # DeepSeek V4 emits verbose reasoning_content + DSML tool calls; needs more steps.
        if max_steps is None:
            if provider == "glm" and model.startswith("glm-4"):
                self.max_steps = 40
            elif model.startswith("deepseek-v4"):
                self.max_steps = 30
            elif model.startswith("qwen3"):
                self.max_steps = 20
            else:
                self.max_steps = 15
        else:
            self.max_steps = max_steps
        self._client = None
        self._tools_override = tools  # If set, use instead of TOOL_REGISTRY

    @property
    def client(self):
        if self._client is None:
            if self.provider == "anthropic":
                import anthropic
                self._client = anthropic.Anthropic()
            elif self.provider == "openai":
                import openai
                self._client = openai.OpenAI()
            elif self.provider == "qwen":
                import os

                import openai
                self._client = openai.OpenAI(
                    api_key=os.environ["DASHSCOPE_API_KEY"],
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                )
            elif self.provider == "deepseek":
                import os

                import openai
                self._client = openai.OpenAI(
                    api_key=os.environ["DEEPSEEK_API_KEY"],
                    base_url="https://api.deepseek.com",
                    timeout=600.0,
                    max_retries=1,
                )
            elif self.provider == "kimi":
                import os

                import openai
                self._client = openai.OpenAI(
                    api_key=os.environ["KIMI_API_KEY"],
                    base_url="https://api.moonshot.cn/v1",
                )
            elif self.provider == "glm":
                import os

                import openai
                self._client = openai.OpenAI(
                    api_key=os.environ["ZHIPU_API_KEY"],
                    base_url="https://open.bigmodel.cn/api/paas/v4/",
                )
            elif self.provider == "openrouter":
                import os

                import openai
                self._client = openai.OpenAI(
                    api_key=os.environ["OPENROUTER_API_KEY"],
                    base_url="https://openrouter.ai/api/v1",
                )
        return self._client

    def estimate_cost(self, reaction: dict) -> dict:
        """Estimate procurement cost for a single reaction using ReAct loop."""
        if self._tools_override is not None:
            TOOL_REGISTRY = self._tools_override
        else:
            from ..tools.agent_tools import TOOL_REGISTRY

        user_prompt = build_task_prompt(reaction)

        if self.provider == "anthropic":
            return self._run_anthropic(user_prompt, TOOL_REGISTRY)
        elif self.provider == "openai" and uses_responses_api(self.model):
            return self._run_openai_responses(user_prompt, TOOL_REGISTRY)
        elif self.provider in ("openai", "qwen", "deepseek", "kimi", "glm", "openrouter"):
            return self._run_openai(user_prompt, TOOL_REGISTRY)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    @staticmethod
    def _attach_token_usage(result: dict, token_usage: dict) -> dict:
        """Attach token_usage to a result dict without mutating the original."""
        result["token_usage"] = token_usage
        return result

    def _run_anthropic(self, user_prompt: str, tools: dict) -> dict:
        """Run ReAct loop with Anthropic Claude."""
        tool_defs = [
            {
                "name": name,
                "description": info["description"],
                "input_schema": {
                    "type": "object",
                    "properties": {
                        param: {"type": "string", "description": desc}
                        for param, desc in info["parameters"].items()
                    },
                    "required": list(info["parameters"].keys()),
                },
            }
            for name, info in tools.items()
        ]

        messages = [{"role": "user", "content": user_prompt}]
        total_input_tokens = 0
        total_output_tokens = 0

        for step in range(self.max_steps):
            response = self.client.messages.create(
                model=self.model,
                system=SYSTEM_PROMPT,
                tools=tool_defs,
                messages=messages,
                **anthropic_request_kwargs(self.model, 4096),
            )

            # Accumulate token usage from Anthropic response
            if hasattr(response, "usage") and response.usage is not None:
                total_input_tokens += getattr(response.usage, "input_tokens", 0)
                total_output_tokens += getattr(response.usage, "output_tokens", 0)

            # Collect text and tool_use blocks
            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            token_usage = {
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens,
            }

            # Check if we need to handle tool calls
            tool_uses = [b for b in assistant_content if b.type == "tool_use"]
            if not tool_uses:
                # No tool calls — extract final answer from text
                text = "".join(b.text for b in assistant_content if b.type == "text")
                parsed = self._try_parse_final_answer(text)
                if parsed is not None:
                    return self._attach_token_usage(parsed, token_usage)
                result = self._coerce_json_anthropic(messages)
                return self._attach_token_usage(result, token_usage)

            # Execute tool calls
            tool_results = []
            for tool_use in tool_uses:
                result = run_tool(tools, tool_use.name, tool_use.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": json.dumps(result),
                })

            messages.append({"role": "user", "content": tool_results})

            if response.stop_reason == "end_turn":
                text = "".join(b.text for b in assistant_content if b.type == "text")
                parsed = self._try_parse_final_answer(text)
                if parsed is not None:
                    return self._attach_token_usage(parsed, token_usage)
                result = self._coerce_json_anthropic(messages)
                return self._attach_token_usage(result, token_usage)

        logger.warning("ReAct agent reached max steps without final answer")
        token_usage = {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
        }
        result = self._coerce_json_anthropic(messages)
        return self._attach_token_usage(result, token_usage)

    def _run_openai(self, user_prompt: str, tools: dict) -> dict:
        """Run ReAct loop with OpenAI."""
        tool_defs = [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": info["description"],
                    "parameters": {
                        "type": "object",
                        "properties": {
                            param: {"type": "string", "description": desc}
                            for param, desc in info["parameters"].items()
                        },
                        "required": list(info["parameters"].keys()),
                    },
                },
            }
            for name, info in tools.items()
        ]

        # Qwen3-235B-A22B-Instruct returns overly minimal final JSON by default
        # (predicted_components stays empty even when components were investigated).
        # DeepSeek V4 has similar conservative output behavior. Force a stronger
        # system prompt for both families.
        active_system = SYSTEM_PROMPT
        if self.model.startswith("qwen3-235b") or self.model.startswith("deepseek-v4"):
            active_system = SYSTEM_PROMPT_FORCE_COMPONENTS

        messages = [
            {"role": "system", "content": active_system},
            {"role": "user", "content": user_prompt},
        ]

        # qwen3-max uses heavy thinking by default that exhausts the max_tokens budget
        # (response truncates to "."). Disable thinking only for that model.
        # qwen3.5-plus uses lighter thinking that helps it reach a valid final answer.
        extra_kwargs = {}
        if self.model in ("qwen3-max", "qwen3-max-latest"):
            extra_kwargs["extra_body"] = {"enable_thinking": False}
        # Kimi K2.x defaults to reasoning mode that saturates output (4096 max) per step
        # and inflates tool call count under noise. Disable for ~3x speedup.
        if self.provider == "kimi" and self.model.startswith("kimi-k2"):
            extra_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        # GLM-4.x defaults: thinking on (mixes reasoning into tool calls) + serial tool
        # calls (only 1 per step). Disable thinking and force parallel tool calling.
        if self.provider == "glm" and self.model.startswith("glm-4"):
            extra_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            extra_kwargs["tool_choice"] = "auto"
            extra_kwargs["parallel_tool_calls"] = True
        # Newer OpenAI models (gpt-5*, o1, o3) require max_completion_tokens
        is_reasoning = self.model.startswith(("gpt-5", "o1", "o3", "o4"))
        # Gemini 2.5+ exposed via OpenAI-compatible gateways is also a reasoning model
        # that saturates the output budget on internal reasoning unless reasoning_effort
        # is set low; trigger the low-reasoning path when the model name signals Gemini.
        is_gemini_reasoning = self.model.startswith("gemini-2") \
            or self.model.startswith("gemini/gemini-3")
        # DeepSeek V4 is also a reasoning model: thinking emitted in `reasoning_content`
        # field shares the output budget with the actual final answer.
        is_deepseek_reasoning = self.model.startswith("deepseek-v4")
        tokens_key = "max_completion_tokens" if (is_reasoning or is_gemini_reasoning) else "max_tokens"
        # gpt-5.4* + tools in chat.completions rejects reasoning_effort (must use /v1/responses).
        tools_no_effort = self.model.startswith("gpt-5.4")
        if is_reasoning and not tools_no_effort:
            extra_kwargs["reasoning_effort"] = "low"
        if is_gemini_reasoning:
            extra_kwargs["reasoning_effort"] = "minimal"
        if is_deepseek_reasoning:
            tokens_budget = 16384  # accommodate verbose reasoning_content + final JSON
        elif tools_no_effort:
            tokens_budget = 16384
        elif is_reasoning or is_gemini_reasoning:
            tokens_budget = 8192
        else:
            tokens_budget = 4096

        total_input_tokens = 0
        total_output_tokens = 0

        # Reasoning models and Kimi K2.x reject temperature=0
        skip_temp = (
            self.model.startswith(("gpt-5", "o1", "o3", "o4"))
            or self.provider == "kimi"
            or is_gemini_reasoning
        )
        temp_kwargs = {} if skip_temp else {"temperature": 0}

        for step in range(self.max_steps):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tool_defs,
                **temp_kwargs,
                **{tokens_key: tokens_budget},
                **extra_kwargs,
            )

            # Accumulate token usage from OpenAI response
            if hasattr(response, "usage") and response.usage is not None:
                total_input_tokens += getattr(response.usage, "prompt_tokens", 0) or 0
                total_output_tokens += getattr(response.usage, "completion_tokens", 0) or 0

            choice = response.choices[0]
            messages.append(choice.message)

            token_usage = {
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens,
            }

            # DeepSeek V4 emits DSML-style tool calls in content text instead of
            # the OpenAI tool_calls field. Detect and parse them via secondary recovery.
            content_text = choice.message.content or ""
            dsml_calls = (
                self._parse_dsml_tool_calls(content_text)
                if "DSML" in content_text and not choice.message.tool_calls
                else []
            )

            if choice.message.tool_calls:  # answer every call, whatever finish_reason says
                for tc in choice.message.tool_calls:
                    result = run_tool(tools, tc.function.name, tc.function.arguments)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result),
                    })
            elif dsml_calls:
                # Synthesize an assistant message with proper tool_calls so the
                # next-turn message history is OpenAI-shaped, then execute and
                # append tool results.
                synthetic_tool_calls = []
                for i, (name, args) in enumerate(dsml_calls):
                    if name not in tools:
                        continue
                    call_id = f"dsml_{step}_{i}"
                    synthetic_tool_calls.append({
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    })
                # Replace last assistant message with the synthesized one so DeepSeek
                # accepts the tool_call_id references on the next turn.
                messages[-1] = {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": synthetic_tool_calls,
                }
                for stc in synthetic_tool_calls:
                    name = stc["function"]["name"]
                    result = run_tool(tools, name, stc["function"]["arguments"])
                    messages.append({
                        "role": "tool",
                        "tool_call_id": stc["id"],
                        "content": json.dumps(result),
                    })
            else:
                parsed = self._try_parse_final_answer(content_text)
                if parsed is not None:
                    return self._attach_token_usage(parsed, token_usage)
                result = self._coerce_json_openai(
                    messages,
                    tokens_key=tokens_key,
                    extra_kwargs=extra_kwargs,
                )
                return self._attach_token_usage(result, token_usage)

        logger.warning("ReAct agent reached max steps without final answer")
        token_usage = {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
        }
        result = self._coerce_json_openai(
            messages, tokens_key=tokens_key, extra_kwargs=extra_kwargs
        )
        return self._attach_token_usage(result, token_usage)

    def _run_openai_responses(self, user_prompt: str, tools: dict) -> dict:
        """ReAct loop on the OpenAI Responses API (Codex models and gpt-5.5+, which accept function
        tools with reasoning only there). Reasoning effort is low, as for other reasoning models."""
        tool_defs = [
            {
                "type": "function",
                "name": name,
                "description": info["description"],
                "parameters": {
                    "type": "object",
                    "properties": {p: {"type": "string", "description": d} for p, d in info["parameters"].items()},
                    "required": list(info["parameters"].keys()),
                    "additionalProperties": False,
                },
                "strict": True,
            }
            for name, info in tools.items()
        ]
        common = {"model": self.model, "instructions": SYSTEM_PROMPT, "reasoning": {"effort": "low"},
                  "max_output_tokens": 16384}
        usage = {"input_tokens": 0, "output_tokens": 0}

        def call(**kwargs):
            response = self.client.responses.create(**common, **kwargs)
            u = getattr(response, "usage", None)
            usage["input_tokens"] += (getattr(u, "input_tokens", 0) or 0) if u else 0
            usage["output_tokens"] += (getattr(u, "output_tokens", 0) or 0) if u else 0
            return response

        def token_usage():
            return {**usage, "total_tokens": usage["input_tokens"] + usage["output_tokens"]}

        response = call(input=[{"role": "user", "content": user_prompt}], tools=tool_defs)
        for _ in range(self.max_steps):
            calls = [item for item in response.output if getattr(item, "type", None) == "function_call"]
            if not calls:
                parsed = self._try_parse_final_answer(response.output_text or "")
                if parsed is not None:
                    return self._attach_token_usage(parsed, token_usage())
                break
            outputs = [{"type": "function_call_output", "call_id": c.call_id,
                        "output": json.dumps(run_tool(tools, c.name, c.arguments))} for c in calls]
            response = call(previous_response_id=response.id, input=outputs, tools=tool_defs)
        else:
            logger.warning("ReAct agent reached max steps without final answer")
        # One last request for strict JSON. Any open tool calls are closed first (the API requires it).
        pending = [{"type": "function_call_output", "call_id": item.call_id,
                    "output": json.dumps({"error": "step limit reached; give your final answer"})}
                   for item in response.output if getattr(item, "type", None) == "function_call"]
        response = call(previous_response_id=response.id, tools=tool_defs, tool_choice="none",
                        input=pending + [{"role": "user", "content": JSON_ONLY_REPROMPT}])
        parsed = self._try_parse_final_answer(response.output_text or "")
        if parsed is None:
            logger.warning("Could not coerce OpenAI response into JSON: %s", (response.output_text or "")[:200])
            parsed = {"predicted_cost_per_gram": None, "predicted_components": []}
        return self._attach_token_usage(parsed, token_usage())

    @staticmethod
    def _parse_dsml_tool_calls(text: str) -> list[tuple[str, dict[str, str]]]:
        """Parse DeepSeek V4 DSML-style tool calls from message content.

        Format example::

            <｜DSML｜tool_calls>
            <｜DSML｜invoke name="calculate">
            <｜DSML｜parameter name="expression" string="true">8.40 / 0.142</｜DSML｜parameter>
            </｜DSML｜invoke>
            </｜DSML｜tool_calls>

        Returns a list of (tool_name, args_dict) tuples.
        """
        # Use the unicode full-width vertical bar that DeepSeek emits (U+FF5C)
        marker = "｜"  # ｜
        pat_invoke = re.compile(
            rf"<{marker}DSML{marker}invoke name=\"([^\"]+)\">(.*?)</{marker}DSML{marker}invoke>",
            re.DOTALL,
        )
        pat_param = re.compile(
            rf"<{marker}DSML{marker}parameter name=\"([^\"]+)\"[^>]*>(.*?)</{marker}DSML{marker}parameter>",
            re.DOTALL,
        )
        calls = []
        for m in pat_invoke.finditer(text):
            tool_name = m.group(1)
            body = m.group(2)
            args = {p.group(1): p.group(2).strip() for p in pat_param.finditer(body)}
            calls.append((tool_name, args))
        return calls

    def _coerce_json_anthropic(self, messages: list[dict]) -> dict:
        """Ask Anthropic one last time to emit strict JSON only."""
        retry_messages = messages + [{"role": "user", "content": JSON_ONLY_REPROMPT}]
        response = self.client.messages.create(
            model=self.model,
            system=SYSTEM_PROMPT,
            messages=retry_messages,
            **anthropic_request_kwargs(self.model, 512),
        )
        text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text")
        parsed = self._try_parse_final_answer(text)
        if parsed is not None:
            return parsed
        logger.warning("Could not coerce Anthropic response into JSON: %s", text[:200])
        return {"predicted_cost_per_gram": None, "predicted_components": []}

    def _coerce_json_openai(self, messages: list[dict], tokens_key: str, extra_kwargs: dict) -> dict:
        """Ask OpenAI-compatible models one last time to emit strict JSON only."""
        retry_messages = messages + [{"role": "user", "content": JSON_ONLY_REPROMPT}]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=retry_messages,
            **{tokens_key: 8192 if tokens_key == "max_completion_tokens" else 512},  # reasoning models think first
            **extra_kwargs,
        )
        text = response.choices[0].message.content or ""
        parsed = self._try_parse_final_answer(text)
        if parsed is not None:
            return parsed
        logger.warning("Could not coerce OpenAI response into JSON: %s", text[:200])
        return {"predicted_cost_per_gram": None, "predicted_components": []}

    @staticmethod
    def _try_parse_final_answer(text: str) -> dict | None:
        """Extract structured cost prediction from model text, if possible."""
        if not isinstance(text, str) or not text:
            return None
        # JSON objects containing predicted_cost_per_gram (nested objects/arrays allowed); the last
        # one is the final answer, earlier ones are drafts.
        found = None
        for match in re.finditer(r"\{", text):
            start = match.start()
            depth = 0
            for i, ch in enumerate(text[start:], start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : i + 1]
                        if "predicted_cost_per_gram" in candidate:
                            try:
                                parsed = json.loads(candidate)
                                if isinstance(parsed, dict):
                                    parsed.setdefault("predicted_components", [])
                                    parsed.pop("_harness_error", None)  # reserved for the evaluator
                                    found = parsed
                            except json.JSONDecodeError:
                                pass
                        break
        if found is not None:
            return found

        # Try to extract just the number
        # Plain-text fallback: only a per-gram figure counts (a total purchase cost is not the
        # answer); accept thousands separators such as "$1,234.56".
        # The figure must follow a ":" or "=" directly, so "cost per gram of step 1 product" does not
        # yield "1"; the last such figure is the final answer.
        cost_matches = re.findall(
            r"(?:cost\s+per\s+gram|per[-\s]gram\s+cost|predicted_cost_per_gram)"
            r"[\s\"*]*(?:\([^)]{0,20}\))?[\s\"*]*[:=][\s*]*(?:USD\s*)?\$?\s*"
            r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
            text,
            re.IGNORECASE,
        )
        if cost_matches:
            return {
                "predicted_cost_per_gram": float(cost_matches[-1].replace(",", "")),
                "predicted_components": [],
            }

        return None

    def _parse_final_answer(self, text: str) -> dict:
        """Parse model output and fall back to a null prediction on failure."""
        parsed = self._try_parse_final_answer(text)
        if parsed is not None:
            return parsed
        logger.warning("Could not parse agent response: %s", text[:200])
        return {"predicted_cost_per_gram": None, "predicted_components": []}


class AgentSDKBaseline:
    """ReAct agent using claude-agent-sdk (OAuth, no API key required)."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_turns: int = 20,
    ) -> None:
        self.model = model
        self.max_turns = max_turns
        self._last_tool_calls: list[dict] = []

    @property
    def tool_calls(self):
        """Return tool calls from the most recent estimate_cost invocation."""
        return list(self._last_tool_calls)

    def estimate_cost(self, reaction: dict) -> dict:
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    return ex.submit(asyncio.run, self._estimate_async(reaction)).result()
        except RuntimeError:
            pass
        return asyncio.run(self._estimate_async(reaction))

    async def _estimate_async(self, reaction: dict) -> dict:
        import json
        import os

        import claude_agent_sdk as sdk

        from ..tools.agent_tools import TOOL_REGISTRY

        # Allow nested claude -p invocations inside a Claude Code session
        os.environ.pop("CLAUDECODE", None)

        user_prompt = build_task_prompt(reaction)

        # Reset tool call log for this invocation
        self._last_tool_calls = []
        step_counter = {"n": 0}

        # Build MCP tools from TOOL_REGISTRY with call tracking
        tool_calls_log = self._last_tool_calls

        def _make_handler(fn, valid_params, tool_name):
            async def handler(args):
                filtered = {k: str(v) for k, v in args.items() if k in valid_params}
                step_counter["n"] += 1
                step_n = step_counter["n"]
                try:
                    result = fn(**filtered)
                    success = not (isinstance(result, dict) and "error" in result)
                    result_str = json.dumps(result) if result is not None else None
                except Exception as exc:
                    result_str = str(exc)
                    success = False
                    result = {"error": str(exc)}
                tool_calls_log.append({
                    "tool_name": tool_name,
                    "arguments": dict(filtered),
                    "result": result_str,
                    "success": success,
                    "step": step_n,
                })
                return {"content": [{"type": "text", "text": json.dumps(result)}]}
            return handler

        mcp_tools = []
        for name, info in TOOL_REGISTRY.items():
            func = info["function"]
            params = info["parameters"]

            mcp_tools.append(sdk.SdkMcpTool(
                name=name,
                description=info["description"],
                input_schema={
                    "type": "object",
                    "properties": {p: {"type": "string", "description": d} for p, d in params.items()},
                    "required": list(params.keys()),
                },
                handler=_make_handler(func, set(params.keys()), name),
            ))

        mcp_server = sdk.create_sdk_mcp_server("chemcost", tools=mcp_tools)

        # Sandbox: no built-in tools (Bash, Read, Web, ...), only the benchmark tools, no other MCP
        # servers or claude.ai connectors, no user/project settings, and an empty working directory.
        _agent_cwd = tempfile.mkdtemp(prefix="chemcost_agent_")
        try:
            final_text = await self._query(sdk, user_prompt, mcp_server, list(TOOL_REGISTRY), _agent_cwd)
        finally:
            shutil.rmtree(_agent_cwd, ignore_errors=True)

        parsed = ReActAgent._try_parse_final_answer(final_text)
        if parsed is not None:
            return parsed
        logger.warning("AgentSDK: could not parse response: %s", final_text[:200])
        return {"predicted_cost_per_gram": None, "predicted_components": []}

    async def _query(self, sdk, user_prompt: str, mcp_server, tool_names: list[str], cwd: str) -> str:
        final_text = ""
        async for msg in sdk.query(
            prompt=user_prompt,
            options=sdk.ClaudeAgentOptions(
                model=self.model,
                max_turns=self.max_turns,
                mcp_servers={"chemcost": mcp_server},
                tools=[],
                allowed_tools=[f"mcp__chemcost__{name}" for name in tool_names],
                setting_sources=[],
                extra_args={"strict-mcp-config": None},
                env={"ENABLE_CLAUDEAI_MCP_SERVERS": "false"},
                system_prompt=SYSTEM_PROMPT,
                cwd=cwd,
                # The CLI bundled with older SDK releases may not support new models: prefer an
                # installed `claude` (or CLAUDE_CLI_PATH) when there is one.
                cli_path=os.environ.get("CLAUDE_CLI_PATH") or shutil.which("claude"),
            ),
        ):
            if isinstance(msg, sdk.AssistantMessage):
                for block in msg.content:
                    if hasattr(block, "text") and block.text:
                        final_text = block.text  # keep last text block
        return final_text


class ZeroShotBaseline:
    """Zero-shot baseline: LLM estimates cost without tools."""

    def __init__(self, model: str = "claude-sonnet-4-20250514", provider: str = "anthropic") -> None:
        self.model = model
        self.provider = provider
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if self.provider == "anthropic":
                import anthropic
                self._client = anthropic.Anthropic()
            elif self.provider == "openai":
                import openai
                self._client = openai.OpenAI()
            elif self.provider == "qwen":
                import os

                import openai
                self._client = openai.OpenAI(
                    api_key=os.environ["DASHSCOPE_API_KEY"],
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                )
            elif self.provider == "deepseek":
                import os

                import openai
                self._client = openai.OpenAI(
                    api_key=os.environ["DEEPSEEK_API_KEY"],
                    base_url="https://api.deepseek.com",
                    timeout=600.0,
                    max_retries=1,
                )
            elif self.provider == "kimi":
                import os

                import openai
                self._client = openai.OpenAI(
                    api_key=os.environ["KIMI_API_KEY"],
                    base_url="https://api.moonshot.cn/v1",
                )
            elif self.provider == "glm":
                import os

                import openai
                self._client = openai.OpenAI(
                    api_key=os.environ["ZHIPU_API_KEY"],
                    base_url="https://open.bigmodel.cn/api/paas/v4/",
                )
            elif self.provider == "openrouter":
                import os

                import openai
                self._client = openai.OpenAI(
                    api_key=os.environ["OPENROUTER_API_KEY"],
                    base_url="https://openrouter.ai/api/v1",
                )
        return self._client

    def estimate_cost(self, reaction: dict) -> dict:
        """Estimate cost without any tools — pure LLM knowledge."""
        prompt = TASK_DEFINITION + "\n\n" + build_task_prompt(reaction, tools=False) + (
            "\nRespond with ONLY a JSON object:\n"
            '{"predicted_cost_per_gram": <number>, "predicted_components": [{"name": "<str>", "price_per_gram": <number>}]}'
        )

        if self.provider == "anthropic":
            resp = self.client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **anthropic_request_kwargs(self.model, 1024),
            )
            text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
            usage = {
                "input_tokens": getattr(resp.usage, "input_tokens", 0),
                "output_tokens": getattr(resp.usage, "output_tokens", 0),
            }
        else:
            extra_kwargs = {}
            if self.model.startswith("qwen3"):
                extra_kwargs["extra_body"] = {"enable_thinking": False}
            # OpenAI reasoning models take max_completion_tokens and need room to think.
            reasoning = self.model.startswith(("gpt-5", "o1", "o3", "o4"))
            token_arg = {"max_completion_tokens": 8192} if reasoning else {"max_tokens": 1024}
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **token_arg,
                **extra_kwargs,
            )
            text = resp.choices[0].message.content or ""
            u = getattr(resp, "usage", None)
            usage = {
                "input_tokens": (getattr(u, "prompt_tokens", 0) or 0) if u else 0,
                "output_tokens": (getattr(u, "completion_tokens", 0) or 0) if u else 0,
            }

        parsed = ReActAgent._try_parse_final_answer(text)
        if parsed is None:
            logger.warning("Could not parse ZeroShot response: %s", text[:200])
            parsed = {"predicted_cost_per_gram": None, "predicted_components": []}
        parsed["token_usage"] = usage
        return parsed
