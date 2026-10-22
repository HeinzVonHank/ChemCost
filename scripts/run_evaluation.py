#!/usr/bin/env python3
"""Run baseline agents on the ChemCost dataset and compute metrics."""

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from chemcost.baselines.codex_agent import CodexBaseline
from chemcost.baselines.prompting_baselines import (
    CoTBaseline,
)
from chemcost.baselines.react_agent import AgentSDKBaseline, ReActAgent, ZeroShotBaseline
from chemcost.evaluation.evaluator import run_evaluation
from chemcost.tools.agent_tools import TOOL_REGISTRY


def main():
    parser = argparse.ArgumentParser(description="Run ChemCost evaluation")
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).parents[1] / "data" / "chemcost.jsonl"),
        help="Path to the ChemCost JSONL file",
    )
    parser.add_argument(
        "--agent",
        choices=[
            "react",
            "zero-shot",
            "cot",
            "sdk",
            "codex",
        ],
        default="react",
        help="Agent type (sdk = claude-agent-sdk, codex = Codex CLI; neither needs an API key)",
    )
    parser.add_argument("--disable-tools", default="",
                        help="codex agent: comma-separated tools to withhold (e.g. calculate)")
    parser.add_argument("--tool-budget", type=int, default=None,
                        help="codex agent: maximum number of tool calls per reaction")
    parser.add_argument("--reasoning-effort", default=None,
                        help="codex agent: model reasoning effort (low/medium/high); default: the model's own")
    parser.add_argument("--model", default="claude-sonnet-4-6", help="LLM model name")
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "qwen", "deepseek", "kimi", "glm", "openrouter"],
        default="anthropic",
    )
    parser.add_argument("--output", default=None, help="Output JSON path for detailed results")
    parser.add_argument(
        "--start", type=int, default=0, help="Start index (0-based) into dataset lines"
    )
    parser.add_argument(
        "--end", type=int, default=None, help="End index (exclusive) into dataset lines"
    )
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers (default 4)")
    parser.add_argument(
        "--input-format",
        choices=["structured", "natural_language"],
        default="structured",
        help="Input format for agent: structured (default) or natural_language (prose)",
    )
    parser.add_argument(
        "--noise",
        choices=["none", "low", "medium", "high", "rich"],
        default="none",
        help="Noise level controlling per-stage intensity (default: none)",
    )
    parser.add_argument(
        "--noise-stages",
        default=None,
        help="Comma-separated stages to enable: name,quantity,missing,format. "
             "If omitted, stages are selected by --noise level.",
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=42,
        help="Random seed for noise injection (default: 42)",
    )
    parser.add_argument(
        "--format-kind",
        choices=["nl_only", "ocr_only", "nl_plus_ocr"],
        default=None,
        help="Override the +Fmt sub-type. Only meaningful when --noise-stages includes 'format'.",
    )
    parser.add_argument(
        "--save-trajectories",
        action="store_true",
        help="Save per-reaction tool-call trajectories in output JSON",
    )
    args = parser.parse_args()

    if args.agent == "react":
        agent = ReActAgent(model=args.model, provider=args.provider)
    elif args.agent == "cot":
        agent = CoTBaseline(model=args.model, provider=args.provider)
    elif args.agent == "sdk":
        agent = AgentSDKBaseline(model=args.model)
    elif args.agent == "codex":
        off = {t.strip() for t in (args.disable_tools or "").split(",") if t.strip()}
        agent = CodexBaseline(model=args.model, tools=[t for t in TOOL_REGISTRY if t not in off],
                              tool_budget=args.tool_budget, reasoning_effort=args.reasoning_effort)
    else:
        agent = ZeroShotBaseline(model=args.model, provider=args.provider)

    if args.output:
        output_path = args.output
    else:
        model_tag = args.model.split("/")[-1]
        stage = f"_s{args.start}-{args.end}" if (args.start or args.end) else ""
        # Every setting that changes the agent input goes into the name, so resume caches never mix.
        setting = "" if args.input_format == "structured" else "_nl"
        if args.noise != "none" or args.noise_stages:
            stages = (args.noise_stages or "default").replace(",", "+")
            setting += f"_noise-{args.noise}-{stages}-seed{args.noise_seed}"
            if args.format_kind:
                setting += f"-{args.format_kind}"
        provider = "" if args.agent in ("sdk", "codex") else f"{args.provider}_"
        output_path = f"results/{args.agent}_{provider}{model_tag}{setting}{stage}.json"

    suffix = f" [lines {args.start}:{args.end}]" if (args.start or args.end) else ""
    fmt_tag = f" [{args.input_format}]" if args.input_format != "structured" else ""
    noise_tag = ""
    if args.noise_stages:
        noise_tag = f" [noise-stages={args.noise_stages}]"
    elif args.noise != "none":
        noise_tag = f" [noise={args.noise}]"
    print(
        f"Running {args.agent} agent ({args.model}) on "
        f"{args.dataset}{suffix}{fmt_tag}{noise_tag}..."
    )

    # Apply noise injection if requested
    noise_fn = None
    if args.noise != "none" or args.noise_stages:
        from chemcost.noise import default_noise_types_for_level, inject_noise

        noise_level = args.noise if args.noise != "none" else "medium"
        noise_seed = args.noise_seed

        # Stage mapping: short names -> internal noise type names
        stage_name_map = {
            "name": ["isomer", "name_variation"],
            "quantity": ["quantity"],
            "missing": ["missing_info"],
            "format": ["format"],
        }

        if args.noise_stages:
            # Explicit stage selection (controlled single-factor experiment)
            noise_types = []
            for s in args.noise_stages.split(","):
                s = s.strip()
                if s in stage_name_map:
                    noise_types.extend(stage_name_map[s])
                elif s in {t for ts in stage_name_map.values() for t in ts}:
                    noise_types.append(s)
                else:
                    parser.error(f"unknown --noise-stages value {s!r}; use name, quantity, missing, format")
        else:
            noise_types = default_noise_types_for_level(noise_level)

        format_kind_arg = args.format_kind

        def noise_fn(record):
            reaction_id = record.get("reaction_id", "")
            seed_material = f"{noise_seed}:{reaction_id}".encode("utf-8")
            record_seed = int(hashlib.sha256(seed_material).hexdigest()[:8], 16)
            return inject_noise(
                record, noise_types=noise_types,
                noise_level=noise_level, seed=record_seed,
                format_kind=format_kind_arg,
            )

    results = run_evaluation(
        agent,
        args.dataset,
        output_path,
        max_workers=args.workers,
        start=args.start,
        end=args.end,
        input_format=args.input_format,
        record_transform=noise_fn,
        save_trajectories=args.save_trajectories,
        run_config={
            "dataset": str(Path(args.dataset).resolve()), "agent": args.agent, "model": args.model,
            "provider": None if args.agent in ("sdk", "codex") else args.provider,
            "disable_tools": args.disable_tools or None, "tool_budget": args.tool_budget,
            "reasoning_effort": args.reasoning_effort,
            "input_format": args.input_format, "noise": args.noise,
            "noise_stages": args.noise_stages, "noise_seed": args.noise_seed,
            "format_kind": args.format_kind,
        },
    )

    print("\n=== Results ===")
    for key, value in results.to_dict().items():
        print(f"  {key}: {value}")
    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    main()
