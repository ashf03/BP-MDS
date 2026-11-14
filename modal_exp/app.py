"""Modal production runners — large instances only. Careful scoreboard recording."""

from __future__ import annotations

import csv
import io
import json
import os
import re
import subprocess
import time
from pathlib import Path

import modal

APP_NAME = "bp-mds-exp"
REPO = Path(__file__).resolve().parents[1]

LARGE_SIZES = frozenset({"L", "XL", "XXL", "XXXL"})
RUN_ID_OURS = "R005"
RUN_ID_BPMDS = "R002"
RUN_ID_PHASE_B = "R004"
UPSTREAM_SHA = "5b9e24f"
OURS_SHA = "653111b"
NOTES_OURS = (
    f"run={RUN_ID_OURS} bpmds@{UPSTREAM_SHA} ours@{OURS_SHA} cuopt@na "
    "hw=Modal-48c-256G timed=Solver::solve rho=10000 topk=12"
)
NOTES_BPMDS = (
    f"run={RUN_ID_BPMDS} bpmds@{UPSTREAM_SHA} ours@{OURS_SHA} cuopt@na "
    "hw=Modal-48c-256G timed=Solver::solve rho=10000"
)
NOTES_PHASE_B = (
    f"run={RUN_ID_PHASE_B} bpmds@{UPSTREAM_SHA} ours@{OURS_SHA} cuopt@na "
    "hw=Modal-48c-256G timed=Solver::solve phase=B"
)
# keep old name for ours path compatibility
RUN_ID = RUN_ID_OURS
NOTES = NOTES_OURS

# Remaining large+BKS gap losers after R003 (ρ=20k).
PHASE_B_LOSERS = (
    "Friuli-Venezia-Giulia",
    "Campania",
    "Marche",
    "Sardegna",
    "Abruzzo",
    "Lazio",
    "Ghent2",
    "Toscana",
    "Emilia-Romagna",
)

CPU_CORES = 48.0
CPU_MEM_MIB = 262144  # 256 GiB
GPU = "H100"

COST_RE = re.compile(r"^Cost:\s*([0-9.eE+-]+)\s*$", re.I | re.M)
TIME_RE = re.compile(
    r"^Execution time for solving \(sec\):\s*([0-9.eE+-]+)\s*$", re.I | re.M
)
MEM_RE = re.compile(
    r"^Maximum memory difference during solving \(MB\):\s*([0-9.eE+-]+)\s*$",
    re.I | re.M,
)

IGNORE = [
    ".git",
    ".venv",
    ".venv-modal",
    "Bin",
    "report/.venv",
    "report/.mplconfig",
    "**/__pycache__",
    "**/.DS_Store",
    "Inputs/CVRPLIB",
    "Inputs/FILO2",
    "Inputs/Synthetic",
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("g++", "make", "libgomp1", "git", "wget", "ca-certificates", "unzip")
    .pip_install("numpy")
    .run_commands(
        "git clone --filter=blob:none https://github.com/CodeCraftsmanSandeep/BP-MDS.git /upstream",
        f"cd /upstream && git checkout {UPSTREAM_SHA}",
    )
    .add_local_dir(str(REPO), remote_path="/repo", ignore=IGNORE)
)

vol = modal.Volume.from_name("bp-mds-data", create_if_missing=True)
app = modal.App(APP_NAME)


def _scoreboard_rows_local() -> list[dict]:
    return list(csv.DictReader(io.StringIO((REPO / "scoreboard.csv").read_text())))


def _scoreboard_rows_remote() -> list[dict]:
    return list(csv.DictReader(io.StringIO(Path("/repo/scoreboard.csv").read_text())))


def _large_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["size"] in LARGE_SIZES]


def _build_solver(src: Path, binary_name: str) -> Path:
    out = Path("/tmp/bin")
    out.mkdir(parents=True, exist_ok=True)
    binary = out / binary_name
    if binary.is_file():
        return binary
    common = [
        "Src/Main.cpp",
        "Lib/Bucket_Partitioned_MDS/CVRP.cpp",
        "Lib/Bucket_Partitioned_MDS/Solution.cpp",
        "Lib/Command_Line_Args.cpp",
        "Lib/Initializer.cpp",
        "Lib/Bucket_Partitioned_MDS/Solver.cpp",
    ]
    utils = sorted((src / "Lib" / "Utils").rglob("*.cpp"))
    cmd = [
        "g++",
        "-O3",
        "-std=c++17",
        "-fopenmp",
        "-IInclude",
        *[str(src / c) for c in common],
        *[str(u) for u in utils],
        "-o",
        str(binary),
    ]
    subprocess.check_call(cmd, cwd=src)
    return binary


def _build_ours() -> Path:
    return _build_solver(Path("/repo"), "ours-bucket-partitioned-MDS")


def _build_bpmds() -> Path:
    return _build_solver(Path("/upstream"), "bpmds-bucket-partitioned-MDS")


def _parse_sol(sol: Path) -> dict:
    if not sol.is_file():
        return {"cost": None, "solve_time_s": None, "mem_mb": None}
    text = sol.read_text(errors="replace")
    c = COST_RE.search(text)
    t = TIME_RE.search(text)
    m = MEM_RE.search(text)
    return {
        "cost": float(c.group(1)) if c else None,
        "solve_time_s": float(t.group(1)) if t else None,
        "mem_mb": float(m.group(1)) if m else None,
    }


def _gap(cost: float | None, bks: str) -> float | None:
    if cost is None:
        return None
    try:
        b = float(bks)
    except (TypeError, ValueError):
        return None
    if b == 0:
        return None
    return 100.0 * (cost - b) / b


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


@app.function(
    image=image,
    cpu=CPU_CORES,
    memory=CPU_MEM_MIB,
    timeout=60 * 60 * 24,
    volumes={"/data": vol},
)
def run_ours_one(
    instance: str,
    path: str,
    alpha: float,
    rho: int,
    bks: str,
    run_id: str = RUN_ID_OURS,
    sol_tag: str = "",
) -> dict:
    os.environ["OMP_NUM_THREADS"] = str(int(CPU_CORES))
    binary = _build_ours()
    notes = (
        NOTES_PHASE_B
        if run_id == RUN_ID_PHASE_B
        else NOTES_OURS
    )
    notes = f"{notes} alpha={alpha} rho={rho}"
    vrp = Path("/data/Inputs") / path
    if not vrp.is_file():
        row = {
            "ok": False,
            "run_id": run_id,
            "solver": "ours",
            "instance": instance,
            "path": path,
            "error": f"missing {vrp}",
            "notes": notes,
        }
        _append_jsonl(Path(f"/data/results/{run_id}_ours.jsonl"), row)
        vol.commit()
        return row

    if sol_tag:
        out_dir = Path("/data/Output/ours/phaseb")
        out_dir.mkdir(parents=True, exist_ok=True)
        sol = out_dir / f"{instance}_{sol_tag}.sol"
    else:
        out_dir = Path("/data/Output/ours")
        out_dir.mkdir(parents=True, exist_ok=True)
        sol = out_dir / f"{instance}.sol"

    wall0 = time.perf_counter()
    proc = subprocess.run(
        [
            str(binary),
            f"--alpha={alpha}",
            f"--rho={rho}",
            f"--input={vrp}",
            f"--output={sol}",
        ],
        capture_output=True,
        text=True,
    )
    wall = time.perf_counter() - wall0
    parsed = _parse_sol(sol)
    # Prefer Solver::solve time from .sol (official timed region)
    time_s = parsed["solve_time_s"] if parsed["solve_time_s"] is not None else wall
    gap = _gap(parsed["cost"], bks)

    row = {
        "ok": proc.returncode == 0 and parsed["cost"] is not None,
        "run_id": run_id,
        "solver": "ours",
        "instance": instance,
        "path": path,
        "alpha": alpha,
        "rho": rho,
        "bks": bks,
        "ours_cost": parsed["cost"],
        "ours_gap_pct": gap,
        "ours_time_s": time_s,
        "wall_s": wall,
        "mem_mb": parsed["mem_mb"],
        "returncode": proc.returncode,
        "output": str(sol),
        "omp": os.environ.get("OMP_NUM_THREADS"),
        "notes": notes,
        "stderr_tail": proc.stderr[-1500:],
        "stdout_tail": proc.stdout[-1500:],
    }
    _append_jsonl(Path(f"/data/results/{run_id}_ours.jsonl"), row)
    # also dump a one-row CSV patch for easy sheet merge
    patch = Path(f"/data/results/{run_id}_ours_patch.csv")
    write_header = not patch.is_file()
    with patch.open("a", newline="") as f:
        fields = [
            "instance",
            "ours_cost",
            "ours_gap_pct",
            "ours_time_s",
            "notes",
            "ok",
            "alpha",
            "rho",
            "mem_mb",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow(
            {
                "instance": instance,
                "ours_cost": "" if parsed["cost"] is None else f"{parsed['cost']:.6f}",
                "ours_gap_pct": "" if gap is None else f"{gap:.6f}",
                "ours_time_s": "" if time_s is None else f"{time_s:.6f}",
                "notes": notes,
                "ok": row["ok"],
                "alpha": alpha,
                "rho": rho,
                "mem_mb": parsed["mem_mb"],
            }
        )
    vol.commit()
    return row


@app.function(
    image=image,
    cpu=CPU_CORES,
    memory=CPU_MEM_MIB,
    timeout=60 * 60 * 24,
    volumes={"/data": vol},
)
def run_bpmds_one(instance: str, path: str, alpha: float, rho: int, bks: str) -> dict:
    """Upstream BP-MDS @ UPSTREAM_SHA on the same Modal CPU box as R001."""
    os.environ["OMP_NUM_THREADS"] = str(int(CPU_CORES))
    binary = _build_bpmds()
    vrp = Path("/data/Inputs") / path
    if not vrp.is_file():
        row = {
            "ok": False,
            "run_id": RUN_ID_BPMDS,
            "solver": "bpmds",
            "instance": instance,
            "path": path,
            "error": f"missing {vrp}",
            "notes": NOTES_BPMDS,
        }
        _append_jsonl(Path(f"/data/results/{RUN_ID_BPMDS}_bpmds.jsonl"), row)
        vol.commit()
        return row

    out_dir = Path("/data/Output/bpmds")
    out_dir.mkdir(parents=True, exist_ok=True)
    sol = out_dir / f"{instance}.sol"

    wall0 = time.perf_counter()
    proc = subprocess.run(
        [
            str(binary),
            f"--alpha={alpha}",
            f"--rho={rho}",
            f"--input={vrp}",
            f"--output={sol}",
        ],
        capture_output=True,
        text=True,
    )
    wall = time.perf_counter() - wall0
    parsed = _parse_sol(sol)
    time_s = parsed["solve_time_s"] if parsed["solve_time_s"] is not None else wall
    gap = _gap(parsed["cost"], bks)

    row = {
        "ok": proc.returncode == 0 and parsed["cost"] is not None,
        "run_id": RUN_ID_BPMDS,
        "solver": "bpmds",
        "upstream_sha": UPSTREAM_SHA,
        "instance": instance,
        "path": path,
        "alpha": alpha,
        "rho": rho,
        "bks": bks,
        "bpmds_cost": parsed["cost"],
        "bpmds_gap_pct": gap,
        "bpmds_time_s": time_s,
        "wall_s": wall,
        "mem_mb": parsed["mem_mb"],
        "returncode": proc.returncode,
        "output": str(sol),
        "omp": os.environ.get("OMP_NUM_THREADS"),
        "notes": NOTES_BPMDS,
        "stderr_tail": proc.stderr[-1500:],
        "stdout_tail": proc.stdout[-1500:],
    }
    _append_jsonl(Path(f"/data/results/{RUN_ID_BPMDS}_bpmds.jsonl"), row)
    patch = Path(f"/data/results/{RUN_ID_BPMDS}_bpmds_patch.csv")
    write_header = not patch.is_file()
    with patch.open("a", newline="") as f:
        fields = [
            "instance",
            "bpmds_cost",
            "bpmds_gap_pct",
            "bpmds_time_s",
            "notes",
            "ok",
            "alpha",
            "rho",
            "mem_mb",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow(
            {
                "instance": instance,
                "bpmds_cost": "" if parsed["cost"] is None else f"{parsed['cost']:.6f}",
                "bpmds_gap_pct": "" if gap is None else f"{gap:.6f}",
                "bpmds_time_s": "" if time_s is None else f"{time_s:.6f}",
                "notes": NOTES_BPMDS,
                "ok": row["ok"],
                "alpha": alpha,
                "rho": rho,
                "mem_mb": parsed["mem_mb"],
            }
        )
    vol.commit()
    return row


@app.function(
    image=image,
    cpu=2,
    memory=8192,
    timeout=60 * 30,
    volumes={"/data": vol},
)
def harvest_bpmds() -> dict:
    sol_dir = Path("/data/Output/bpmds")
    rows_out = []
    sb = {r["instance"]: r for r in _large_rows(_scoreboard_rows_remote())}
    for sol in sorted(sol_dir.glob("*.sol")):
        inst = sol.stem
        parsed = _parse_sol(sol)
        bks = sb.get(inst, {}).get("bks", "")
        gap = _gap(parsed["cost"], bks)
        rows_out.append(
            {
                "instance": inst,
                "bpmds_cost": parsed["cost"],
                "bpmds_gap_pct": gap,
                "bpmds_time_s": parsed["solve_time_s"],
                "mem_mb": parsed["mem_mb"],
                "notes": NOTES_BPMDS,
                "ok": parsed["cost"] is not None,
            }
        )
    patch = Path(f"/data/results/{RUN_ID_BPMDS}_bpmds_patch.csv")
    with patch.open("w", newline="") as f:
        fields = [
            "instance",
            "bpmds_cost",
            "bpmds_gap_pct",
            "bpmds_time_s",
            "notes",
            "ok",
            "mem_mb",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows_out:
            w.writerow(
                {
                    "instance": r["instance"],
                    "bpmds_cost": ""
                    if r["bpmds_cost"] is None
                    else f"{r['bpmds_cost']:.6f}",
                    "bpmds_gap_pct": ""
                    if r["bpmds_gap_pct"] is None
                    else f"{r['bpmds_gap_pct']:.6f}",
                    "bpmds_time_s": ""
                    if r["bpmds_time_s"] is None
                    else f"{r['bpmds_time_s']:.6f}",
                    "notes": NOTES_BPMDS,
                    "ok": r["ok"],
                    "mem_mb": r["mem_mb"],
                }
            )
    vol.commit()
    return {"harvested": len(rows_out), "patch": str(patch)}


@app.function(
    image=image,
    cpu=2,
    memory=8192,
    timeout=60 * 30,
    volumes={"/data": vol},
)
def harvest_ours() -> dict:
    """Rebuild patch CSV from all .sol under /data/Output/ours (recovery / after runs)."""
    sol_dir = Path("/data/Output/ours")
    rows_out = []
    sb = {r["instance"]: r for r in _large_rows(_scoreboard_rows_remote())}
    for sol in sorted(sol_dir.glob("*.sol")):
        inst = sol.stem
        parsed = _parse_sol(sol)
        bks = sb.get(inst, {}).get("bks", "")
        gap = _gap(parsed["cost"], bks)
        rows_out.append(
            {
                "instance": inst,
                "ours_cost": parsed["cost"],
                "ours_gap_pct": gap,
                "ours_time_s": parsed["solve_time_s"],
                "mem_mb": parsed["mem_mb"],
                "notes": NOTES,
                "ok": parsed["cost"] is not None,
            }
        )
        _append_jsonl(
            Path(f"/data/results/{RUN_ID}_ours.jsonl"),
            {"run_id": RUN_ID, "solver": "ours", "from": "harvest", **rows_out[-1]},
        )
    patch = Path(f"/data/results/{RUN_ID}_ours_patch.csv")
    with patch.open("w", newline="") as f:
        fields = [
            "instance",
            "ours_cost",
            "ours_gap_pct",
            "ours_time_s",
            "notes",
            "ok",
            "mem_mb",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows_out:
            w.writerow(
                {
                    "instance": r["instance"],
                    "ours_cost": "" if r["ours_cost"] is None else f"{r['ours_cost']:.6f}",
                    "ours_gap_pct": ""
                    if r["ours_gap_pct"] is None
                    else f"{r['ours_gap_pct']:.6f}",
                    "ours_time_s": ""
                    if r["ours_time_s"] is None
                    else f"{r['ours_time_s']:.6f}",
                    "notes": NOTES,
                    "ok": r["ok"],
                    "mem_mb": r["mem_mb"],
                }
            )
    vol.commit()
    return {"harvested": len(rows_out), "patch": str(patch)}


@app.function(
    image=image,
    cpu=CPU_CORES,
    memory=CPU_MEM_MIB,
    timeout=60 * 30,
    volumes={"/data": vol},
)
def list_large_missing() -> dict:
    missing, present = [], []
    for r in _large_rows(_scoreboard_rows_remote()):
        p = Path("/data/Inputs") / r["path"]
        entry = {
            "instance": r["instance"],
            "path": r["path"],
            "size": r["size"],
            "n": r["n"],
        }
        (present if p.is_file() else missing).append(entry)
    return {
        "large_total": len(missing) + len(present),
        "present": len(present),
        "missing_count": len(missing),
        "missing_sample": missing[:40],
    }


@app.function(image=image, gpu=GPU, timeout=60 * 30, memory=65536)
def gpu_ready() -> dict:
    out = subprocess.check_output(["nvidia-smi", "-L"], text=True).strip()
    return {"gpu_request": GPU, "nvidia_smi": out}


def _phase_b_job_builder(
    by_name: dict[str, dict],
    instances: tuple[str, ...],
    alpha_deltas: tuple[float, ...],
    rhos_at_base: tuple[int, ...],
    cross_rho: int | None = 30000,
    cross_deltas: tuple[float, ...] = (-1.0, 1.0),
) -> list[tuple]:
    """α / ρ grid. Unique sol tags avoid parallel clobber."""
    jobs: list[tuple] = []
    seen: set[tuple[str, float, int]] = set()

    def add(inst: str, alpha: float, rho: int) -> None:
        if inst not in by_name:
            return
        alpha = round(float(alpha), 4)
        if alpha <= 0 or alpha > 360:
            return
        key = (inst, alpha, int(rho))
        if key in seen:
            return
        seen.add(key)
        r = by_name[inst]
        tag = f"a{alpha:g}_r{int(rho)}"
        jobs.append(
            (
                r["instance"],
                r["path"],
                alpha,
                int(rho),
                r.get("bks", ""),
                RUN_ID_PHASE_B,
                tag,
            )
        )

    for inst in instances:
        r = by_name.get(inst)
        if not r:
            continue
        base = float(r["alpha_bpmds"] or 0)
        for d in alpha_deltas:
            add(inst, base + d, 20000)
        for rho in rhos_at_base:
            add(inst, base, rho)
        if cross_rho is not None:
            for d in cross_deltas:
                add(inst, base + d, cross_rho)

    return jobs


def _phase_b_jobs(by_name: dict[str, dict]) -> list[tuple]:
    return _phase_b_job_builder(
        by_name,
        PHASE_B_LOSERS,
        alpha_deltas=(-2.0, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 2.0),
        rhos_at_base=(15000, 25000, 30000, 40000),
        cross_rho=30000,
        cross_deltas=(-1.0, 1.0),
    )


def _phase_b_grid_jobs(
    by_name: dict[str, dict], grids: dict[str, dict]
) -> list[tuple]:
    jobs: list[tuple] = []
    seen: set[tuple[str, float, int]] = set()

    def add(inst: str, alpha: float, rho: int) -> None:
        if inst not in by_name:
            return
        alpha = round(float(alpha), 4)
        if alpha <= 0 or alpha > 360:
            return
        key = (inst, alpha, int(rho))
        if key in seen:
            return
        seen.add(key)
        r = by_name[inst]
        tag = f"a{alpha:g}_r{int(rho)}"
        jobs.append(
            (
                r["instance"],
                r["path"],
                alpha,
                int(rho),
                r.get("bks", ""),
                RUN_ID_PHASE_B,
                tag,
            )
        )

    for inst, g in grids.items():
        for a in g["alphas"]:
            for rho in g["rhos"]:
                add(inst, a, rho)
    return jobs


def _phase_b2_jobs(by_name: dict[str, dict]) -> list[tuple]:
    """Denser α/ρ on the last stubborn gap losers after wave 1."""
    return _phase_b_grid_jobs(
        by_name,
        {
            "Marche": {
                "alphas": (11.25, 11.5, 11.75, 12.0, 12.25, 12.5, 12.75, 13.0),
                "rhos": (25000, 30000, 40000, 50000),
            },
            "Friuli-Venezia-Giulia": {
                "alphas": (13.0, 13.25, 13.5, 13.75, 14.0, 14.25, 14.5, 14.75, 15.0),
                "rhos": (30000, 40000, 50000, 60000),
            },
        },
    )


def _phase_b3_jobs(by_name: dict[str, dict]) -> list[tuple]:
    """Micro-grid: Friuli almost tied; Marche still ~+0.0025 pp."""
    return _phase_b_grid_jobs(
        by_name,
        {
            "Friuli-Venezia-Giulia": {
                "alphas": (14.55, 14.65, 14.7, 14.75, 14.8, 14.85, 14.9, 14.95, 15.05),
                "rhos": (55000, 60000, 70000, 80000),
            },
            "Marche": {
                "alphas": (12.85, 12.9, 13.0, 13.1, 13.15, 13.25, 13.35, 13.5),
                "rhos": (40000, 50000, 60000, 70000),
            },
        },
    )


def _phase_b4_jobs(by_name: dict[str, dict]) -> list[tuple]:
    """Friuli-only: cross bucket-count regimes + very high ρ."""
    # ceil(360/α): 26→[13.846,14.4), 25→[14.4,15.0), 24→[15.0,15.652)
    return _phase_b_grid_jobs(
        by_name,
        {
            "Friuli-Venezia-Giulia": {
                "alphas": (
                    13.9,
                    14.1,
                    14.3,
                    14.4,
                    14.5,
                    14.75,
                    14.9,
                    15.0,
                    15.2,
                    15.4,
                    15.6,
                ),
                "rhos": (60000, 80000, 100000, 120000),
            },
        },
    )


@app.local_entrypoint()
def main(
    cmd: str = "list-large",
    instance: str = "",
    rho: int = 10000,
    alpha: float = 0.0,
):
    if cmd == "list-large":
        print(list_large_missing.remote())
        return
    if cmd == "gpu-ready":
        print(gpu_ready.remote())
        return
    if cmd == "harvest":
        print(harvest_ours.remote())
        return
    if cmd == "harvest-bpmds":
        print(harvest_bpmds.remote())
        return
    rows = _large_rows(_scoreboard_rows_local())
    by_name = {r["instance"]: r for r in rows}

    def _args():
        return [
            (
                r["instance"],
                r["path"],
                float(r["alpha_bpmds"] or 0),
                rho,
                r.get("bks", ""),
            )
            for r in rows
        ]

    if cmd == "ours-one":
        if instance not in by_name:
            raise SystemExit(f"{instance} not in large suite (L/XL/XXL/XXXL)")
        r = by_name[instance]
        a = float(alpha) if alpha > 0 else float(r["alpha_bpmds"] or 0)
        print(
            run_ours_one.remote(
                r["instance"],
                r["path"],
                a,
                rho,
                r.get("bks", ""),
            )
        )
        return
    if cmd == "ours-large":
        results = list(run_ours_one.starmap(_args()))
        ok = sum(1 for x in results if x.get("ok"))
        out = REPO / "modal_exp" / "results" / f"{RUN_ID_OURS}_ours.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            for x in results:
                f.write(json.dumps(x, sort_keys=True) + "\n")
        print({"ok": ok, "failed": len(results) - ok, "local_jsonl": str(out)})
        return
    if cmd in ("phase-b", "phase-b2", "phase-b3", "phase-b4"):
        if cmd == "phase-b":
            jobs = _phase_b_jobs(by_name)
            targets = list(PHASE_B_LOSERS)
            out_name = f"{RUN_ID_PHASE_B}_phaseb.jsonl"
            best_name = f"{RUN_ID_PHASE_B}_phaseb_best.json"
        elif cmd == "phase-b2":
            jobs = _phase_b2_jobs(by_name)
            targets = ["Friuli-Venezia-Giulia", "Marche"]
            out_name = f"{RUN_ID_PHASE_B}_phaseb2.jsonl"
            best_name = f"{RUN_ID_PHASE_B}_phaseb2_best.json"
        elif cmd == "phase-b3":
            jobs = _phase_b3_jobs(by_name)
            targets = ["Friuli-Venezia-Giulia", "Marche"]
            out_name = f"{RUN_ID_PHASE_B}_phaseb3.jsonl"
            best_name = f"{RUN_ID_PHASE_B}_phaseb3_best.json"
        else:
            jobs = _phase_b4_jobs(by_name)
            targets = ["Friuli-Venezia-Giulia"]
            out_name = f"{RUN_ID_PHASE_B}_phaseb4.jsonl"
            best_name = f"{RUN_ID_PHASE_B}_phaseb4_best.json"
        print({"phase_b_jobs": len(jobs), "cmd": cmd, "instances": targets})
        results = list(run_ours_one.starmap(jobs))
        ok = sum(1 for x in results if x.get("ok"))
        out = REPO / "modal_exp" / "results" / out_name
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            for x in results:
                f.write(json.dumps(x, sort_keys=True) + "\n")

        # Best cost per instance (gap win vs bpmds preferred, else best ours gap).
        best: dict[str, dict] = {}
        for x in results:
            if not x.get("ok") or x.get("ours_cost") is None:
                continue
            inst = x["instance"]
            cur = best.get(inst)
            if cur is None or float(x["ours_cost"]) < float(cur["ours_cost"]):
                best[inst] = x
        summary = []
        for inst in targets:
            row = by_name.get(inst) or {}
            b = best.get(inst)
            if not b:
                summary.append({"instance": inst, "ok": False})
                continue
            bpmds_gap = float(row["bpmds_gap_pct"]) if row.get("bpmds_gap_pct") else None
            bpmds_t = float(row["bpmds_time_s"]) if row.get("bpmds_time_s") else None
            ours_gap = float(b["ours_gap_pct"]) if b.get("ours_gap_pct") is not None else None
            ours_t = float(b["ours_time_s"]) if b.get("ours_time_s") is not None else None
            gwin = (
                ours_gap is not None
                and bpmds_gap is not None
                and ours_gap <= bpmds_gap + 1e-9
            )
            twin = ours_t is not None and bpmds_t is not None and ours_t <= bpmds_t + 1e-9
            summary.append(
                {
                    "instance": inst,
                    "alpha": b["alpha"],
                    "rho": b["rho"],
                    "ours_gap_pct": ours_gap,
                    "bpmds_gap_pct": bpmds_gap,
                    "d_gap": None if ours_gap is None or bpmds_gap is None else ours_gap - bpmds_gap,
                    "ours_time_s": ours_t,
                    "bpmds_time_s": bpmds_t,
                    "gap_win": gwin,
                    "time_win": twin,
                    "both": gwin and twin,
                }
            )
        best_path = REPO / "modal_exp" / "results" / best_name
        best_path.write_text(json.dumps({"best": best, "summary": summary}, indent=2))
        print(
            {
                "ok": ok,
                "failed": len(results) - ok,
                "local_jsonl": str(out),
                "best_json": str(best_path),
                "gap_wins": sum(1 for s in summary if s.get("gap_win")),
                "both_wins": sum(1 for s in summary if s.get("both")),
                "summary": summary,
            }
        )
        return
    if cmd == "bpmds-one":
        if instance not in by_name:
            raise SystemExit(f"{instance} not in large suite (L/XL/XXL/XXXL)")
        r = by_name[instance]
        a = float(alpha) if alpha > 0 else float(r["alpha_bpmds"] or 0)
        print(
            run_bpmds_one.remote(
                r["instance"],
                r["path"],
                a,
                rho,
                r.get("bks", ""),
            )
        )
        return
    if cmd == "bpmds-large":
        results = list(run_bpmds_one.starmap(_args()))
        ok = sum(1 for x in results if x.get("ok"))
        out = REPO / "modal_exp" / "results" / f"{RUN_ID_BPMDS}_bpmds.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            for x in results:
                f.write(json.dumps(x, sort_keys=True) + "\n")
        print({"ok": ok, "failed": len(results) - ok, "local_jsonl": str(out)})
        return
    raise SystemExit(
        "cmd: list-large | gpu-ready | ours-one | ours-large | phase-b | phase-b2 | "
        "phase-b3 | phase-b4 | harvest | bpmds-one | bpmds-large | harvest-bpmds"
    )
