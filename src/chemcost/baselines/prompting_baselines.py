"""Chain-of-Thought baseline (no tools) for the ChemCost benchmark."""

from __future__ import annotations

import logging

from .react_agent import TASK_DEFINITION, ReActAgent, anthropic_request_kwargs, build_task_prompt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared helper: build a client for any supported provider
# ---------------------------------------------------------------------------

def _make_client(provider: str):
    """Lazily build an API client for the given provider."""
    if provider == "anthropic":
        import anthropic
        return anthropic.Anthropic()
    elif provider == "openai":
        import openai
        return openai.OpenAI()
    elif provider == "qwen":
        import os

        import openai
        return openai.OpenAI(
            api_key=os.environ["DASHSCOPE_API_KEY"],
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    elif provider == "deepseek":
        import os

        import openai
        # DeepSeek V4 reasoning calls can hang silently; cap with explicit timeouts
        # and one client-level retry so a stalled connection is dropped, not waited on.
        return openai.OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
            timeout=600.0,
            max_retries=1,
        )
    elif provider == "kimi":
        import os

        import openai
        return openai.OpenAI(
            api_key=os.environ["KIMI_API_KEY"],
            base_url="https://api.moonshot.cn/v1",
        )
    elif provider == "glm":
        import os

        import openai
        return openai.OpenAI(
            api_key=os.environ["ZHIPU_API_KEY"],
            base_url="https://open.bigmodel.cn/api/paas/v4/",
        )
    elif provider == "openrouter":
        import os

        import openai
        return openai.OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
    raise ValueError(f"Unknown provider: {provider}")


# ---------------------------------------------------------------------------
# CoT prompt (no tools, no worked examples)
# ---------------------------------------------------------------------------

COT_SYSTEM_PROMPT = TASK_DEFINITION + """

No tools are available: rely on your knowledge of typical laboratory chemical prices and pack sizes.
Think step by step, then give the final answer as JSON only:
{"predicted_cost_per_gram": <number>, "predicted_components": [{"name": "<str>", "price_per_gram": <number>}]}
"""


def _build_user_prompt(reaction: dict, *, include_cot: bool = False) -> str:
    """User prompt for the no-tool baselines."""
    prompt = build_task_prompt(reaction, tools=False)
    if include_cot:
        prompt += "\nThink step by step through each component before giving your answer.\n"
    return prompt + (
        '\nRespond with ONLY a JSON object:\n{"predicted_cost_per_gram": <number>, '
        '"predicted_components": [{"name": "<str>", "price_per_gram": <number>}]}'
    )


# ---------------------------------------------------------------------------
# LLM call helpers (Anthropic / OpenAI / Qwen)
# ---------------------------------------------------------------------------

def _call_anthropic(client, model: str, system: str, user_msg: str, max_tokens: int = 2048) -> str:
    """Single Anthropic API call, return (text, token_usage)."""
    resp = client.messages.create(
        model=model,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
        **anthropic_request_kwargs(model, max_tokens),
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    usage = {
        "input_tokens": getattr(resp.usage, "input_tokens", 0),
        "output_tokens": getattr(resp.usage, "output_tokens", 0),
    }
    return text, usage


def _call_openai(
    client,
    model: str,
    system: str,
    user_msg: str,
    max_tokens: int = 2048,
    extra_kwargs: dict | None = None,
) -> tuple[str, dict]:
    """Single OpenAI-compatible API call, return (text, token_usage)."""
    extra_kwargs = extra_kwargs or {}
    tokens_key = (
        "max_completion_tokens"
        if model.startswith(("gpt-5", "o1", "o3", "o4"))
        else "max_tokens"
    )
    # Kimi K2.x rejects temperature=0
    skip_temp = (
        model.startswith(("gpt-5", "o1", "o3", "o4"))
        or model.startswith("kimi-k")
    )
    call_kwargs = {} if skip_temp else {"temperature": 0}
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        **call_kwargs,
        **{tokens_key: max_tokens},
        **extra_kwargs,
    )
    text = resp.choices[0].message.content or ""
    u = getattr(resp, "usage", None)
    usage = {
        "input_tokens": (getattr(u, "prompt_tokens", 0) or 0) if u else 0,
        "output_tokens": (getattr(u, "completion_tokens", 0) or 0) if u else 0,
    }
    return text, usage


def _call_llm(client, model: str, provider: str, system: str, user_msg: str) -> tuple[str, dict]:
    """Dispatch an LLM call to the appropriate provider. Returns (text, token_usage)."""
    if provider == "anthropic":
        return _call_anthropic(client, model, system, user_msg)
    else:
        extra_kwargs = {}
        if model in ("qwen3-max", "qwen3-max-latest"):
            extra_kwargs = {"extra_body": {"enable_thinking": False}}
        # Reasoning models burn output budget on reasoning_content; raise the cap.
        if model.startswith("deepseek-v4"):
            max_tokens = 16384
        elif model.startswith(("gpt-5", "o1", "o3", "o4")):
            max_tokens = 8192  # OpenAI reasoning models: same budget as the zero-shot baseline
        else:
            max_tokens = 2048
        return _call_openai(
            client, model, system, user_msg,
            max_tokens=max_tokens, extra_kwargs=extra_kwargs,
        )


# ============================================================================
# CoTBaseline — Chain-of-Thought, no tools
# ============================================================================

class CoTBaseline:
    """Chain-of-Thought baseline: LLM reasons step-by-step but has no tools."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        provider: str = "anthropic",
    ) -> None:
        self.model = model
        self.provider = provider
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = _make_client(self.provider)
        return self._client

    def estimate_cost(self, reaction: dict) -> dict:
        """Estimate cost using chain-of-thought prompting, no tools."""
        user_msg = _build_user_prompt(reaction, include_cot=True)
        text, usage = _call_llm(
            self.client, self.model, self.provider,
            COT_SYSTEM_PROMPT, user_msg,
        )
        parsed = ReActAgent._try_parse_final_answer(text)
        if parsed is None:
            parsed = {"predicted_cost_per_gram": None, "predicted_components": []}
        parsed["token_usage"] = usage
        return parsed
