"""Command-line entry point: `chemcost download` and `chemcost mcp`."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

DEFAULT_REPO = os.environ.get("CHEMCOST_HF_REPO", "YoungerWu/ChemCost")
CACHE_DIR = Path(os.environ.get("CHEMCOST_CACHE", Path.home() / ".cache" / "chemcost"))


def download(out: Path, repo: str = DEFAULT_REPO, revision: str | None = None) -> Path:
    """Fetch chemcost.jsonl and pricing_db.sqlite from Hugging Face into `out`."""
    from huggingface_hub import snapshot_download

    out.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        revision=revision,
        local_dir=out,
        allow_patterns=["data/chemcost.jsonl", "pricing/pricing_db.sqlite"],
    )
    (out / "data" / "chemcost.jsonl").replace(out / "chemcost.jsonl")
    (out / "pricing" / "pricing_db.sqlite").replace(out / "pricing_db.sqlite")
    (out / "data").rmdir()
    (out / "pricing").rmdir()
    return out


def _serve_mcp() -> None:
    if "CHEMCOST_PRICING_DB" not in os.environ:
        db = CACHE_DIR / "pricing_db.sqlite"
        if not db.exists():
            download(CACHE_DIR)
        os.environ["CHEMCOST_PRICING_DB"] = str(db)
    from .tools.mcp_server import server

    server.run()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="chemcost", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    dl = sub.add_parser("download", help="download the dataset and frozen pricing database")
    dl.add_argument("--repo", default=DEFAULT_REPO, help="Hugging Face dataset repo id")
    dl.add_argument("--revision", default=None, help="dataset revision (tag or commit)")
    dl.add_argument("--out", default="data", help="output directory (default: ./data)")

    sub.add_parser(
        "mcp",
        help="serve the benchmark tools over stdio MCP (downloads the pricing DB to "
        "~/.cache/chemcost on first run unless CHEMCOST_PRICING_DB is set)",
    )

    args = parser.parse_args(argv)
    if args.command == "download":
        out = download(Path(args.out), args.repo, args.revision)
        print(f"Data ready under {out}/")
    else:
        _serve_mcp()


if __name__ == "__main__":
    main()
