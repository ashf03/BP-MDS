#!/usr/bin/env python3
"""Rebuild the single scoreboard (all instances) from Inputs + Results catalogs."""

from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "Results" / "README.md"
INSTANCES = ROOT / "Inputs" / "instances.csv"
OUT_CSV = Path(__file__).resolve().parent / "scoreboard.csv"
OUT_TSV = Path(__file__).resolve().parent / "scoreboard.tsv"

COLS = [
    "sno",
    "size",
    "source",
    "set",
    "instance",
    "path",
    "customers",
    "n",
    "q",
    "bks",
    "alpha_bpmds",
    "bpmds_cost",
    "bpmds_gap_pct",
    "bpmds_time_s",
    "ours_cost",
    "ours_gap_pct",
    "ours_time_s",
    "cuopt_cost",
    "cuopt_gap_pct",
    "cuopt_time_s",
    "notes",
]

ROW_RE = re.compile(
    r"\|\s*(\d+)\s*\|\s*([^|]+)\|\s*(\w+)\s*\|\s*([^|]+)\|\s*(\d+)\s*\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|"
)


def load_results() -> dict[str, dict]:
    text = RESULTS.read_text()
    out: dict[str, dict] = {}
    for m in ROW_RE.finditer(text):
        _sno, _size, _src, inst, _n, alpha, bks, cost, gap = [x.strip() for x in m.groups()]
        gap_s = gap.replace("%", "").strip()
        if gap_s in {"—", "-", ""}:
            gap_s = ""
        bks_s = bks.strip()
        if bks_s in {"—", "-", ""}:
            bks_s = ""
        out[inst] = {
            "alpha_bpmds": alpha,
            "bpmds_cost": cost,
            "bpmds_gap_pct": gap_s,
            "bks_results": bks_s,
        }
    return out


def main() -> None:
    results = load_results()
    rows: list[dict] = []
    with INSTANCES.open() as f:
        for row in csv.DictReader(f):
            name = row["Name"]
            r = results.get(name, {})
            bks = row["BKS"] or r.get("bks_results", "")
            if bks in {"—", "-"}:
                bks = ""
            rows.append(
                {
                    "sno": row["S.No"],
                    "size": row["Size"],
                    "source": row["Source"],
                    "set": row["Set"],
                    "instance": name,
                    "path": row["Path"],
                    "customers": row["Customers"],
                    "n": row["N"],
                    "q": row["Q"],
                    "bks": bks,
                    "alpha_bpmds": r.get("alpha_bpmds", ""),
                    "bpmds_cost": r.get("bpmds_cost", ""),
                    "bpmds_gap_pct": r.get("bpmds_gap_pct", ""),
                    "bpmds_time_s": "",
                    "ours_cost": "",
                    "ours_gap_pct": "",
                    "ours_time_s": "",
                    "cuopt_cost": "",
                    "cuopt_gap_pct": "",
                    "cuopt_time_s": "",
                    "notes": "",
                }
            )

    rows.sort(key=lambda x: int(x["sno"]))
    for path, delim in ((OUT_CSV, ","), (OUT_TSV, "\t")):
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS, delimiter=delim)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
