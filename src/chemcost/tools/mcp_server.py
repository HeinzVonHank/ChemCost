"""The benchmark tools as a stdio MCP server, for agent CLIs that take MCP tools (e.g. Codex).

Run: python -m chemcost.tools.mcp_server
"""

from __future__ import annotations

import inspect
import os

from mcp.server.fastmcp import FastMCP

from .agent_tools import TOOL_REGISTRY

server = FastMCP("chemcost")


def _register(name: str, info: dict) -> None:
    fn = info["function"]
    (param,) = info["parameters"]  # every benchmark tool takes one string argument

    def tool(**kwargs) -> dict:
        try:
            return fn(**{param: str(kwargs[param])})
        except Exception as e:  # a tool failure goes back to the model, not up the stack
            return {"error": f"{type(e).__name__}: {e}"}

    tool.__name__ = name
    tool.__doc__ = info["description"]
    tool.__annotations__ = {param: str, "return": dict}
    tool.__signature__ = inspect.Signature(
        [inspect.Parameter(param, inspect.Parameter.KEYWORD_ONLY, annotation=str)], return_annotation=dict)
    server.tool(name=name, description=info["description"])(tool)


# CHEMCOST_TOOLS: comma-separated subset to expose; CHEMCOST_TOOL_BUDGET: total calls allowed per process
# (one process serves one reaction), after which every call returns an error.
_allowed = [t for t in os.environ.get("CHEMCOST_TOOLS", ",".join(TOOL_REGISTRY)).split(",") if t in TOOL_REGISTRY]
_budget = int(os.environ["CHEMCOST_TOOL_BUDGET"]) if os.environ.get("CHEMCOST_TOOL_BUDGET") else None
_calls = [0]


def _counted(fn):
    def wrapper(**kwargs):
        _calls[0] += 1
        if _budget is not None and _calls[0] > _budget:
            return {"error": f"tool budget of {_budget} calls exhausted: give your final answer"}
        return fn(**kwargs)
    return wrapper


for _name in _allowed:
    _register(_name, {**TOOL_REGISTRY[_name], "function": _counted(TOOL_REGISTRY[_name]["function"])})


if __name__ == "__main__":
    server.run()
