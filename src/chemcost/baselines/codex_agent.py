"""Codex CLI baseline: an OpenAI model run by `codex exec` with only the ChemCost tools (MCP).

Every built-in Codex capability (shell, code execution against the filesystem, web search, apps,
plugins, browser, sub-agents) is switched off; the session runs read-only in an empty temporary
directory, so the agent can reach nothing but the three benchmark tools. Uses the Codex login
(e.g. a ChatGPT account) instead of an API key.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from ..pricing.pricing_db import DEFAULT_DB_PATH
from ..tools.agent_tools import TOOL_REGISTRY
from .react_agent import SYSTEM_PROMPT, ReActAgent, build_task_prompt

logger = logging.getLogger(__name__)

_DISABLED_FEATURES = (
    "shell_tool", "unified_exec", "view_image", "standalone_web_search", "multi_agent", "apps", "plugins",
    "browser_use", "browser_use_external", "computer_use", "image_generation", "in_app_browser", "goals",
    "hooks", "memories",
)


class CodexBaseline:
    """Run each reaction as one non-interactive Codex session."""

    def __init__(self, model: str = "gpt-5.6-luna", reasoning_effort: str | None = None,
                 tools: list[str] | None = None, tool_budget: int | None = None, timeout_s: int = 1200) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.tools = list(tools or TOOL_REGISTRY)
        self.tool_budget = tool_budget
        self.timeout_s = timeout_s
        self.codex = os.environ.get("CODEX_CLI_PATH") or shutil.which("codex")
        self.tool_calls: list[dict] = []

    def _command(self, workdir: str, last_message: str) -> list[str]:
        src = str(Path(__file__).resolve().parents[2])  # the directory that holds the chemcost package
        env = {"PYTHONPATH": src, "CHEMCOST_PRICING_DB": str(DEFAULT_DB_PATH), "CHEMCOST_TOOLS": ",".join(self.tools)}
        if self.tool_budget is not None:
            env["CHEMCOST_TOOL_BUDGET"] = str(self.tool_budget)
        cmd = [self.codex, "exec", "--ephemeral", "--ignore-user-config", "--ignore-rules", "--skip-git-repo-check",
               "-s", "read-only", "-C", workdir, "-m", self.model, "--json", "-o", last_message]
        for feature in _DISABLED_FEATURES:
            cmd += ["--disable", feature]
        cmd += ["-c", 'web_search="disabled"',
                "-c", f"mcp_servers.chemcost.command={json.dumps(sys.executable)}",
                "-c", 'mcp_servers.chemcost.args=["-m","chemcost.tools.mcp_server"]',
                "-c", "mcp_servers.chemcost.env={" + ",".join(f"{k}={json.dumps(v)}" for k, v in env.items()) + "}"]
        for tool in self.tools:
            cmd += ["-c", f'mcp_servers.chemcost.tools.{tool}.approval_mode="approve"']
        if self.reasoning_effort:
            cmd += ["-c", f'model_reasoning_effort="{self.reasoning_effort}"']
        return cmd + ["-"]

    def _instructions(self) -> str:
        notes = []
        if "calculate" not in self.tools:
            notes.append("No calculator is available: do all arithmetic yourself.")
        if self.tool_budget is not None:
            notes.append(f"You may make at most {self.tool_budget} tool calls in total for this reaction.")
        return "\n".join(notes)

    def estimate_cost(self, reaction: dict) -> dict:
        if not self.codex:
            raise RuntimeError("codex CLI not found (install it or set CODEX_CLI_PATH)")
        extra = self._instructions()
        prompt = SYSTEM_PROMPT + "\n\n" + build_task_prompt(reaction) + ("\n" + extra if extra else "")
        workdir = tempfile.mkdtemp(prefix="chemcost_codex_")
        last = os.path.join(workdir, "answer.txt")
        try:
            proc = subprocess.run(self._command(workdir, last), input=prompt, capture_output=True, text=True,
                                  timeout=self.timeout_s)
            events = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]
            text = Path(last).read_text() if os.path.exists(last) else ""
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        failed = [e for e in events if e.get("type") == "turn.failed"]
        if failed:
            raise RuntimeError(f"codex turn failed: {json.dumps(failed[0].get('error'))[:300]}")
        self.tool_calls = [
            {"tool_name": e["item"].get("tool"), "arguments": e["item"].get("arguments"),
             "result": json.dumps(e["item"].get("result"))[:4000], "success": e["item"].get("error") is None,
             "step": i}
            for i, e in enumerate(ev for ev in events if ev.get("type") == "item.completed"
                                  and ev.get("item", {}).get("type") == "mcp_tool_call")
        ]
        usage = {"input_tokens": 0, "output_tokens": 0}
        for e in events:
            if e.get("type") == "turn.completed":
                usage["input_tokens"] += e["usage"].get("input_tokens", 0)
                usage["output_tokens"] += e["usage"].get("output_tokens", 0)
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        parsed = ReActAgent._try_parse_final_answer(text) or {"predicted_cost_per_gram": None,
                                                                 "predicted_components": []}
        parsed["token_usage"] = usage
        return parsed
