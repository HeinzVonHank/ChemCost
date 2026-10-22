#!/usr/bin/env python3
"""Download the ChemCost dataset and frozen pricing database from Hugging Face into data/."""

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_REPO = os.environ.get("CHEMCOST_HF_REPO", "YoungerWu/ChemCost")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Hugging Face dataset repo id")
    parser.add_argument("--revision", default=None, help="Dataset revision (tag or commit)")
    parser.add_argument("--out", default=str(Path(__file__).parents[1] / "data"))
    args = parser.parse_args()

    out = Path(args.out)
    snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        revision=args.revision,
        local_dir=out,
        allow_patterns=["data/chemcost.jsonl", "pricing/pricing_db.sqlite"],
    )
    (out / "data" / "chemcost.jsonl").replace(out / "chemcost.jsonl")
    (out / "pricing" / "pricing_db.sqlite").replace(out / "pricing_db.sqlite")
    (out / "data").rmdir()
    (out / "pricing").rmdir()
    print(f"Data ready under {out}/")


if __name__ == "__main__":
    main()
