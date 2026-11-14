#!/usr/bin/env python3
"""
Production runner: BP-MDS .sol (or Modal jsonl/patch) → scoreboard rows.

Timed region from .sol = "Execution time for solving (sec)" (= Solver::solve).
Writes root + benchmarks scoreboard.csv/.tsv. Paste TSV into the Google Sheet.

Examples:
  python modal_exp/record_sols.py --solver ours --sol-dir /path/to/Output/ours
  python modal_exp/record_sols.py --solver ours --sol Ghent2.sol --notes 'run=R012 ...'
  python modal_exp/record_sols.py --solver ours --jsonl modal_exp/results/R011_ours_ls1.jsonl \\
      --both-win-vs bpmds --dry-run
  python modal_exp/record_sols.py --solver ours_gpu --jsonl modal_exp/results/R010_ours_gpu_cuda.jsonl
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

# Allow `python modal_exp/record_sols.py` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from record import (  # noqa: E402
    NOTES_BPMDS,
    NOTES_CUOPT,
    NOTES_OURS,
    NOTES_OURS_CUDA,
    NOTES_OURS_GPU,
    SCOREBOARD_PATHS,
    apply_bpmds_results,
    apply_cuopt_results,
    apply_ours_gpu_results,
    apply_ours_results,
    gap_pct,
    parse_sol,
    results_from_jsonl,
    _load_rows,
)

# Phase-B / tagged sols: Friuli-Venezia-Giulia_a14.75_r75000.sol
_TAG_RE = re.compile(r"_a[0-9]+(?:\.[0-9]+)?_r[0-9]+$", re.I)

SOLVERS = ("ours", "bpmds", "cuopt", "ours_gpu")

DEFAULT_NOTES = {
    "ours": NOTES_OURS,
    "bpmds": NOTES_BPMDS,
    "cuopt": NOTES_CUOPT,
    "ours_gpu": NOTES_OURS_GPU,
}


def instance_from_stem(stem: str) -> str:
    """Map .sol stem → scoreboard instance (strip α/ρ tags)."""
    return _TAG_RE.sub("", stem)


def collect_sols(
    *,
    sol: Path | None,
    sol_dir: Path | None,
    recursive: bool,
) -> list[dict]:
    paths: list[Path] = []
    if sol is not None:
        if not sol.is_file():
            raise SystemExit(f"missing --sol {sol}")
        paths.append(sol)
    if sol_dir is not None:
        if not sol_dir.is_dir():
            raise SystemExit(f"missing --sol-dir {sol_dir}")
        pattern = "**/*.sol" if recursive else "*.sol"
        paths.extend(sorted(sol_dir.glob(pattern)))
    out: list[dict] = []
    for path in paths:
        parsed = parse_sol(path)
        inst = instance_from_stem(path.stem)
        out.append(
            {
                "instance": inst,
                "cost": parsed["cost"],
                "time_s": parsed["time_s"],
                "mem_mb": parsed["mem_mb"],
                "sol_path": str(path),
                "ok": parsed["cost"] is not None and parsed["time_s"] is not None,
            }
        )
    return out


def collect_jsonl(path: Path, solver: str) -> list[dict]:
    cost_key = {
        "ours": "ours_cost",
        "bpmds": "bpmds_cost",
        "cuopt": "cuopt_cost",
        "ours_gpu": "ours_gpu_cost",
    }[solver]
    time_key = {
        "ours": "ours_time_s",
        "bpmds": "bpmds_time_s",
        "cuopt": "cuopt_time_s",
        "ours_gpu": "ours_gpu_time_s",
    }[solver]
    rows: list[dict] = []
    for r in results_from_jsonl(path):
        if solver in {"cuopt", "ours_gpu"}:
            # Keep full row (skipped / use_cuda / cuopt_* keys).
            if "cost" not in r and cost_key in r:
                r = {**r, "cost": r.get(cost_key)}
            if "time_s" not in r and time_key in r:
                r = {**r, "time_s": r.get(time_key)}
            rows.append(r)
            continue
        rows.append(
            {
                "instance": r.get("instance"),
                "cost": r.get(cost_key, r.get("cost")),
                "time_s": r.get(time_key, r.get("time_s", r.get("solve_time_s"))),
                "ok": r.get("ok", True),
                "notes": r.get("notes"),
                "use_cuda": r.get("use_cuda"),
            }
        )
    return rows


def collect_patch_csv(path: Path, solver: str) -> list[dict]:
    cost_key = {
        "ours": "ours_cost",
        "bpmds": "bpmds_cost",
        "cuopt": "cuopt_cost",
        "ours_gpu": "ours_gpu_cost",
    }[solver]
    time_key = {
        "ours": "ours_time_s",
        "bpmds": "bpmds_time_s",
        "cuopt": "cuopt_time_s",
        "ours_gpu": "ours_gpu_time_s",
    }[solver]
    rows: list[dict] = []
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            if solver == "cuopt":
                rows.append(
                    {
                        "instance": r["instance"],
                        "cuopt_cost": r.get("cuopt_cost") or None,
                        "cuopt_time_s": r.get("cuopt_time_s") or None,
                        "cost": float(r["cuopt_cost"]) if r.get("cuopt_cost") else None,
                        "time_s": float(r["cuopt_time_s"])
                        if r.get("cuopt_time_s")
                        else None,
                        "ok": str(r.get("ok", "true")).lower()
                        in {"1", "true", "yes", ""},
                    }
                )
            elif solver == "ours_gpu":
                c = r.get(cost_key) or r.get("ours_gpu_cost")
                t = r.get(time_key) or r.get("ours_gpu_time_s")
                rows.append(
                    {
                        "instance": r["instance"],
                        "ours_gpu_cost": c,
                        "ours_gpu_time_s": t,
                        "cost": float(c) if c else None,
                        "time_s": float(t) if t else None,
                        "ok": str(r.get("ok", "true")).lower()
                        in {"1", "true", "yes", ""},
                        "use_cuda": str(r.get("use_cuda", "")).lower()
                        in {"1", "true", "yes"},
                    }
                )
            else:
                c = r.get(cost_key) or r.get("cost")
                t = r.get(time_key) or r.get("time_s")
                rows.append(
                    {
                        "instance": r["instance"],
                        "cost": float(c) if c not in (None, "") else None,
                        "time_s": float(t) if t not in (None, "") else None,
                        "ok": True,
                    }
                )
    return rows


def _f(x) -> float | None:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def filter_both_win(
    results: list[dict],
    solver: str,
    vs: str,
) -> tuple[list[dict], list[dict]]:
    """Keep rows where solver gap≤ and time≤ contender on the live scoreboard."""
    _, sb_rows = _load_rows()
    by_sb = {r["instance"]: r for r in sb_rows}
    kept: list[dict] = []
    dropped: list[dict] = []
    gap_col = {
        "bpmds": "bpmds_gap_pct",
        "cuopt": "cuopt_gap_pct",
        "ours": "ours_gap_pct",
    }[vs]
    time_col = {
        "bpmds": "bpmds_time_s",
        "cuopt": "cuopt_time_s",
        "ours": "ours_time_s",
    }[vs]

    for r in results:
        inst = r.get("instance")
        sb = by_sb.get(inst or "")
        if not sb:
            dropped.append({**r, "drop_reason": "not_on_scoreboard"})
            continue
        cost = _f(r.get("cost", r.get(f"{solver}_cost")))
        time_s = _f(r.get("time_s", r.get(f"{solver}_time_s")))
        cg = _f(sb.get(gap_col))
        ct = _f(sb.get(time_col))
        if cost is None or time_s is None or cg is None or ct is None:
            dropped.append({**r, "drop_reason": "missing_cost_time_or_contender"})
            continue
        g = float(gap_pct(cost, sb.get("bks", "")) or "nan")
        if g != g:  # NaN
            dropped.append({**r, "drop_reason": "no_bks"})
            continue
        if g <= cg + 1e-9 and time_s <= ct + 1e-9:
            kept.append(r)
        else:
            dropped.append(
                {
                    **r,
                    "drop_reason": "not_both_win",
                    "gap": g,
                    "contender_gap": cg,
                    "time_s": time_s,
                    "contender_time": ct,
                }
            )
    return kept, dropped


def filter_known_instances(results: list[dict]) -> tuple[list[dict], list[str]]:
    _, sb_rows = _load_rows()
    known = {r["instance"] for r in sb_rows}
    kept, missing = [], []
    for r in results:
        inst = r.get("instance")
        if inst in known:
            kept.append(r)
        else:
            missing.append(str(inst))
    return kept, missing


def preview(results: list[dict], solver: str, sb_by: dict[str, dict]) -> None:
    print(
        f"{'instance':28} {'cost':>14} {'gap%':>10} {'time_s':>10} {'bks':>12}"
    )
    for r in sorted(results, key=lambda x: str(x.get("instance"))):
        inst = r.get("instance") or "?"
        cost = _f(r.get("cost", r.get(f"{solver}_cost")))
        time_s = _f(r.get("time_s", r.get(f"{solver}_time_s")))
        bks = (sb_by.get(inst) or {}).get("bks", "")
        g = gap_pct(cost, bks) if cost is not None else ""
        c_s = "" if cost is None else f"{cost:.4f}"
        t_s = "" if time_s is None else f"{time_s:.4f}"
        print(f"{inst:28} {c_s:>14} {g:>10} {t_s:>10} {bks:>12}")


def stamp_notes(results: list[dict], notes: str) -> list[dict]:
    out = []
    for r in results:
        rr = dict(r)
        rr["notes"] = notes
        out.append(rr)
    return out


def apply(
    solver: str,
    results: list[dict],
    *,
    notes: str | None,
) -> int:
    # Optional notes override via hit["notes"] for ours_gpu path; for ours/bpmds
    # the apply_* functions use module NOTES_*. Temporarily patch when override set.
    import record as rec

    if notes:
        results = stamp_notes(results, notes)
        if solver == "ours":
            old = rec.NOTES_OURS
            rec.NOTES_OURS = notes
            try:
                return apply_ours_results(results)
            finally:
                rec.NOTES_OURS = old
        if solver == "bpmds":
            old = rec.NOTES_BPMDS
            rec.NOTES_BPMDS = notes
            try:
                return apply_bpmds_results(results)
            finally:
                rec.NOTES_BPMDS = old
        if solver == "ours_gpu":
            # apply_ours_gpu_results picks CUDA vs GPU note from use_cuda / notes
            for r in results:
                r.setdefault("ok", True)
            return apply_ours_gpu_results(results)
        if solver == "cuopt":
            old = rec.NOTES_CUOPT
            rec.NOTES_CUOPT = notes
            try:
                return apply_cuopt_results(results, mark_skipped=False)
            finally:
                rec.NOTES_CUOPT = old

    if solver == "ours":
        return apply_ours_results(results)
    if solver == "bpmds":
        return apply_bpmds_results(results)
    if solver == "ours_gpu":
        for r in results:
            r.setdefault("ok", True)
            # Prefer CUDA note string if caller stamped run=R010
            if notes and "cuda" in notes.lower():
                r["use_cuda"] = True
        return apply_ours_gpu_results(results)
    return apply_cuopt_results(results, mark_skipped=False)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Record .sol / jsonl / patch into scoreboard (production)."
    )
    ap.add_argument("--solver", choices=SOLVERS, required=True)
    ap.add_argument("--sol", type=Path, help="Single .sol file")
    ap.add_argument("--sol-dir", type=Path, help="Directory of .sol files")
    ap.add_argument(
        "--recursive",
        action="store_true",
        help="With --sol-dir, glob **/*.sol",
    )
    ap.add_argument("--jsonl", type=Path, help="Modal results jsonl")
    ap.add_argument("--patch-csv", type=Path, help="Modal patch CSV")
    ap.add_argument(
        "--notes",
        type=str,
        default="",
        help="Override notes stamp (else solver default from record.py)",
    )
    ap.add_argument(
        "--both-win-vs",
        choices=("bpmds", "cuopt", "ours"),
        help="Only apply rows that beat contender on gap and time",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse/filter and print; do not write scoreboard",
    )
    ap.add_argument(
        "--large-only",
        action="store_true",
        help="Skip instances whose scoreboard size is not L/XL/XXL/XXXL",
    )
    args = ap.parse_args()

    results: list[dict] = []
    if args.sol or args.sol_dir:
        results.extend(
            collect_sols(sol=args.sol, sol_dir=args.sol_dir, recursive=args.recursive)
        )
    if args.jsonl:
        if not args.jsonl.is_file():
            raise SystemExit(f"missing --jsonl {args.jsonl}")
        results.extend(collect_jsonl(args.jsonl, args.solver))
    if args.patch_csv:
        if not args.patch_csv.is_file():
            raise SystemExit(f"missing --patch-csv {args.patch_csv}")
        results.extend(collect_patch_csv(args.patch_csv, args.solver))

    if not results:
        raise SystemExit(
            "no results: pass --sol and/or --sol-dir and/or --jsonl and/or --patch-csv"
        )

    # Dedupe by instance (last wins).
    by_inst: dict[str, dict] = {}
    for r in results:
        inst = r.get("instance")
        if not inst:
            continue
        by_inst[inst] = r
    results = list(by_inst.values())

    results, missing = filter_known_instances(results)
    if missing:
        print(f"warn: {len(missing)} sol(s) not on scoreboard (skipped): {missing[:8]}")

    if args.large_only:
        _, sb_rows = _load_rows()
        large = {
            r["instance"]
            for r in sb_rows
            if (r.get("size") or "").upper() in {"L", "XL", "XXL", "XXXL"}
        }
        results = [r for r in results if r.get("instance") in large]

    dropped: list[dict] = []
    if args.both_win_vs:
        results, dropped = filter_both_win(results, args.solver, args.both_win_vs)
        print(
            f"both-win vs {args.both_win_vs}: keep {len(results)} drop {len(dropped)}"
        )

    _, sb_rows = _load_rows()
    sb_by = {r["instance"]: r for r in sb_rows}
    preview(results, args.solver, sb_by)

    notes = args.notes.strip() or None
    if args.dry_run:
        print(
            f"dry-run: would update {len(results)} row(s); "
            f"notes={notes or DEFAULT_NOTES[args.solver]!r}"
        )
        print(f"targets: {[str(p) for p in SCOREBOARD_PATHS]}")
        return

    n = apply(args.solver, results, notes=notes)
    print(f"updated {n} scoreboard row-writes")
    print(f"notes={notes or DEFAULT_NOTES[args.solver]!r}")
    print("files: scoreboard.csv scoreboard.tsv benchmarks/scoreboard.*")
    print("next: paste scoreboard.tsv into the Google Sheet (one tab)")


if __name__ == "__main__":
    main()
