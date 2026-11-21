#!/usr/bin/env python3
"""Download the ChemCost dataset and frozen pricing database from Hugging Face into data/."""

import argparse
from pathlib import Path

from chemcost.cli import DEFAULT_REPO, download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Hugging Face dataset repo id")
    parser.add_argument("--revision", default=None, help="Dataset revision (tag or commit)")
    parser.add_argument("--out", default=str(Path(__file__).parents[1] / "data"))
    args = parser.parse_args()
    out = download(Path(args.out), args.repo, args.revision)
    print(f"Data ready under {out}/")


if __name__ == "__main__":
    main()
