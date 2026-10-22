#!/usr/bin/env python3
"""Recompute every ground-truth label from the frozen pricing database and check it matches."""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from chemcost.cost_calculator import (
    calculate_multistep_procurement_cost,
    calculate_procurement_cost,
)
from chemcost.pricing.pricing_db import DEFAULT_DB_PATH, PricingDB


def _steps_as_input(record: dict) -> list[dict]:
    return [
        {
            "step_number": s["step_number"],
            "reaction": {
                "product_name": s["product"]["name"],
                "product_smiles": s["product"]["smiles"],
                "product_mw": s["product"]["mw"],
                "yield_percent": s["yield_percent"],
                "components": s["components"],
            },
        }
        for s in record["steps"]
    ]


def recompute(record: dict, db: PricingDB) -> float | None:
    if record["type"] == "multi_step":
        res = calculate_multistep_procurement_cost(_steps_as_input(record), db)
    else:
        res = calculate_procurement_cost(
            record["product"]["mw"], record["yield_percent"], record["components"], db
        )
    return res.procurement_cost_usd_per_g_product


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset", nargs="?", default=str(Path(__file__).parents[1] / "data" / "chemcost.jsonl")
    )
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    args = parser.parse_args()

    db = PricingDB(args.db, read_only=True)
    n = bad = 0
    for line in open(args.dataset):
        r = json.loads(line)
        got, want = recompute(r, db), r["procurement_cost_usd_per_g_product"]
        n += 1
        if got is None or not math.isclose(got, want, rel_tol=1e-6, abs_tol=1e-4):
            bad += 1
            print(f"MISMATCH {r['reaction_id']}: stored={want} recomputed={got}")
    print(f"{n - bad}/{n} labels reproduced from {args.db}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
