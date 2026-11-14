"""Parse BP-MDS .sol files and merge into scoreboard (production recording)."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTES_OURS = (
    "run=R006 bpmds@5b9e24f ours@a7a7533 cuopt@na "
    "hw=Modal-48c-256G timed=Solver::solve rho=10000 topk=2"
)
NOTES_BPMDS = (
    "run=R002 bpmds@5b9e24f ours@1190a34 cuopt@na "
    "hw=Modal-48c-256G timed=Solver::solve rho=10000"
)
NOTES = NOTES_OURS  # default for ours apply

COST_RE = re.compile(r"^Cost:\s*([0-9.eE+-]+)\s*$", re.I | re.M)
TIME_RE = re.compile(
    r"^Execution time for solving \(sec\):\s*([0-9.eE+-]+)\s*$", re.I | re.M
)
MEM_RE = re.compile(
    r"^Maximum memory difference during solving \(MB\):\s*([0-9.eE+-]+)\s*$",
    re.I | re.M,
)

SCOREBOARD_PATHS = [
    REPO / "scoreboard.csv",
    REPO / "benchmarks" / "scoreboard.csv",
]


def parse_sol(path: Path) -> dict:
    text = path.read_text(errors="replace")
    cost_m = COST_RE.search(text)
    time_m = TIME_RE.search(text)
    mem_m = MEM_RE.search(text)
    return {
        "cost": float(cost_m.group(1)) if cost_m else None,
        "time_s": float(time_m.group(1)) if time_m else None,
        "mem_mb": float(mem_m.group(1)) if mem_m else None,
        "sol_path": str(path),
    }


def gap_pct(cost: float | None, bks: str) -> str:
    if cost is None:
        return ""
    try:
        b = float(bks)
    except (TypeError, ValueError):
        return ""
    if b == 0:
        return ""
    return f"{100.0 * (cost - b) / b:.6f}"


def _write_pair(csv_path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    tsv_path = csv_path.with_suffix(".tsv")
    with tsv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        w.writerows(rows)


def _load_rows() -> tuple[list[str], list[dict]]:
    root = REPO / "scoreboard.csv"
    with root.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def apply_ours_results(results: list[dict]) -> int:
    by_inst = {r["instance"]: r for r in results if r.get("instance")}
    fieldnames, rows = _load_rows()
    updated = 0
    for row in rows:
        hit = by_inst.get(row["instance"])
        if not hit:
            continue
        if hit.get("cost") is not None:
            row["ours_cost"] = f"{float(hit['cost']):.6f}"
            row["ours_gap_pct"] = gap_pct(float(hit["cost"]), row.get("bks", ""))
        if hit.get("time_s") is not None:
            row["ours_time_s"] = f"{float(hit['time_s']):.6f}"
        # keep R001 ours note unless bpmds note already present from R002
        if "run=R002" not in (row.get("notes") or ""):
            row["notes"] = NOTES_OURS
        updated += 1
    for path in SCOREBOARD_PATHS:
        _write_pair(path, fieldnames, rows)
    return updated


def apply_bpmds_results(results: list[dict]) -> int:
    """Fill bpmds_cost / bpmds_gap_pct / bpmds_time_s from matched Modal upstream runs."""
    by_inst = {r["instance"]: r for r in results if r.get("instance")}
    fieldnames, rows = _load_rows()
    updated = 0
    for row in rows:
        hit = by_inst.get(row["instance"])
        if not hit:
            continue
        if hit.get("cost") is not None:
            row["bpmds_cost"] = f"{float(hit['cost']):.6f}"
            row["bpmds_gap_pct"] = gap_pct(float(hit["cost"]), row.get("bks", ""))
        if hit.get("time_s") is not None:
            row["bpmds_time_s"] = f"{float(hit['time_s']):.6f}"
        # merge notes: keep ours R001 tag + R002
        prev = row.get("notes") or ""
        if "run=R001" in prev and "run=R002" not in prev:
            row["notes"] = prev + " | " + NOTES_BPMDS
        else:
            row["notes"] = NOTES_BPMDS
        updated += 1
    for path in SCOREBOARD_PATHS:
        _write_pair(path, fieldnames, rows)
    return updated


def results_from_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows
