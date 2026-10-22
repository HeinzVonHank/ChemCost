"""Tool usage tracking for ChemCost benchmark evaluation.

Wraps agent tools to transparently log every call, enabling
detailed analysis of tool usage patterns and ablation experiments.
"""

from __future__ import annotations

import functools
import json
import time
from collections import Counter
from dataclasses import dataclass


@dataclass
class ToolCall:
    """Record of a single tool invocation."""

    tool_name: str
    arguments: dict
    result: str | None
    success: bool
    timestamp: float
    step_number: int


@dataclass
class ToolUsageStats:
    """Aggregated statistics about tool usage."""

    total_calls: int
    calls_per_tool: dict[str, int]
    success_rate_per_tool: dict[str, float]
    avg_calls_per_reaction: float
    redundant_calls: int  # same tool+args called twice
    tool_sequence: list[str]  # ordered list of tool names used

    def to_dict(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "calls_per_tool": self.calls_per_tool,
            "success_rate_per_tool": {
                k: round(v, 4) for k, v in self.success_rate_per_tool.items()
            },
            "avg_calls_per_reaction": round(self.avg_calls_per_reaction, 4),
            "redundant_calls": self.redundant_calls,
            "tool_sequence": self.tool_sequence,
        }


class ToolTracker:
    """Wraps agent tools to track all calls without modifying them.

    Usage::

        tracker = ToolTracker()
        wrapped_tools = tracker.wrap_tools(TOOL_REGISTRY)
        # Pass wrapped_tools to agent instead of TOOL_REGISTRY
        # After agent runs:
        stats = tracker.get_stats()
    """

    def __init__(self) -> None:
        self._calls: list[ToolCall] = []
        self._step: int = 0
        self._n_reactions: int = 0

    def reset(self) -> None:
        """Clear all recorded calls."""
        self._calls = []
        self._step = 0
        self._n_reactions = 0

    def mark_reaction_boundary(self) -> None:
        """Call between reactions to track per-reaction averages."""
        self._n_reactions += 1

    def wrap_tools(self, tools: dict) -> dict:
        """Return a copy of the tool registry with wrapped functions.

        Each tool's ``function`` value is replaced by a wrapper that logs
        the call before delegating to the original function.  All other
        keys (description, parameters) are preserved unchanged.

        Args:
            tools: Tool registry dict (name -> {function, description, parameters}).

        Returns:
            New dict with the same structure but wrapped functions.
        """
        wrapped = {}
        for name, info in tools.items():
            wrapped[name] = dict(info)  # shallow copy
            wrapped[name]["function"] = self._make_wrapper(name, info["function"])
        return wrapped

    def _make_wrapper(self, tool_name: str, func):
        """Create a logging wrapper around a tool function."""

        @functools.wraps(func)
        def wrapper(**kwargs):
            self._step += 1
            ts = time.time()

            try:
                result = func(**kwargs)
                result_str = json.dumps(result) if result is not None else None
                success = "error" not in (result or {})
            except Exception as exc:
                result_str = str(exc)
                success = False
                result = {"error": str(exc)}

            self._calls.append(
                ToolCall(
                    tool_name=tool_name,
                    arguments=dict(kwargs),
                    result=result_str,
                    success=success,
                    timestamp=ts,
                    step_number=self._step,
                )
            )
            return result

        return wrapper

    def get_calls(self) -> list[ToolCall]:
        """Return all recorded tool calls in order."""
        return list(self._calls)

    def get_stats(self) -> ToolUsageStats:
        """Compute aggregated statistics from recorded calls."""
        total = len(self._calls)

        # Calls per tool
        calls_per_tool: dict[str, int] = Counter()
        successes_per_tool: dict[str, int] = Counter()
        for call in self._calls:
            calls_per_tool[call.tool_name] += 1
            if call.success:
                successes_per_tool[call.tool_name] += 1

        # Success rate per tool
        success_rate: dict[str, float] = {}
        for name, count in calls_per_tool.items():
            success_rate[name] = successes_per_tool[name] / count if count > 0 else 0.0

        # Redundant calls: same (tool_name, arguments) pair seen more than once
        seen: Counter = Counter()
        for call in self._calls:
            key = (call.tool_name, json.dumps(call.arguments, sort_keys=True))
            seen[key] += 1
        redundant = sum(count - 1 for count in seen.values() if count > 1)

        # Tool sequence
        sequence = [call.tool_name for call in self._calls]

        # Avg calls per reaction
        n_reactions = max(self._n_reactions, 1)
        avg_per_reaction = total / n_reactions

        return ToolUsageStats(
            total_calls=total,
            calls_per_tool=dict(calls_per_tool),
            success_rate_per_tool=success_rate,
            avg_calls_per_reaction=avg_per_reaction,
            redundant_calls=redundant,
            tool_sequence=sequence,
        )
