#!/usr/bin/env python3
"""Merge Modal results into scoreboard.csv/.tsv (root + benchmarks)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from record import (
    NOTES_BPMDS,
    NOTES_CUOPT,
    NOTES_OURS,
    NOTES_OURS_GPU,
    apply_bpmds_results,
    apply_cuopt_results,
    apply_ours_gpu_results,
    apply_ours_results,
    parse_sol,
    results_from_jsonl,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--solver", choices=("ours", "bpmds", "cuopt", "ours_gpu"), required=True
    )
    ap.add_argument("--jsonl", type=Path)
    ap.add_argument("--sol-dir", type=Path)
    ap.add_argument("--patch-csv", type=Path)
    args = ap.parse_args()

    results: list[dict] = []
    cost_key = {
        "ours": "ours_cost",
        "bpmds": "bpmds_cost",
        "cuopt": "cuopt_cost",
        "ours_gpu": "ours_gpu_cost",
    }[args.solver]
    time_key = {
        "ours": "ours_time_s",
        "bpmds": "bpmds_time_s",
        "cuopt": "cuopt_time_s",
        "ours_gpu": "ours_gpu_time_s",
    }[args.solver]

    if args.jsonl and args.jsonl.is_file():
        for r in results_from_jsonl(args.jsonl):
            if args.solver in {"cuopt", "ours_gpu"}:
                results.append(r)
            else:
                results.append(
                    {
                        "instance": r.get("instance"),
                        "cost": r.get(cost_key, r.get("cost")),
                        "time_s": r.get(
                            time_key, r.get("time_s", r.get("solve_time_s"))
                        ),
                    }
                )
    if args.sol_dir and args.sol_dir.is_dir():
        for sol in sorted(args.sol_dir.glob("*.sol")):
            p = parse_sol(sol)
            results.append(
                {"instance": sol.stem, "cost": p["cost"], "time_s": p["time_s"]}
            )
    if args.patch_csv and args.patch_csv.is_file():
        with args.patch_csv.open(newline="") as f:
            for r in csv.DictReader(f):
                if args.solver == "cuopt":
                    results.append(
                        {
                            "instance": r["instance"],
                            "cuopt_cost": r.get("cuopt_cost") or None,
                            "cuopt_time_s": r.get("cuopt_time_s") or None,
                            "ok": str(r.get("ok", "")).lower() in {"1", "true", "yes"},
                        }
                    )
                else:
                    c = r.get(cost_key) or r.get("ours_cost") or r.get("bpmds_cost")
                    t = r.get(time_key) or r.get("ours_time_s") or r.get("bpmds_time_s")
                    results.append(
                        {
                            "instance": r["instance"],
                            "cost": float(c) if c else None,
                            "time_s": float(t) if t else None,
                        }
                    )

    if not results:
        raise SystemExit("no results: pass --jsonl and/or --sol-dir and/or --patch-csv")

    if args.solver == "ours":
        n = apply_ours_results(results)
        notes = NOTES_OURS
    elif args.solver == "bpmds":
        n = apply_bpmds_results(results)
        notes = NOTES_BPMDS
    elif args.solver == "ours_gpu":
        n = apply_ours_gpu_results(results)
        notes = NOTES_OURS_GPU
    else:
        n = apply_cuopt_results(results)
        notes = NOTES_CUOPT
    print(f"updated {n} scoreboard row-writes; notes={notes!r}")
    print("files: scoreboard.csv scoreboard.tsv benchmarks/scoreboard.*")
    print("next: paste scoreboard.tsv into the Google Sheet (one tab)")


if __name__ == "__main__":
    main()
