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
RUN_ID_OURS = "R006"
RUN_ID_BPMDS = "R002"
RUN_ID_PHASE_B = "R007"
RUN_ID_CUOPT = "R008"
RUN_ID_OURS_GPU = "R009"
RUN_ID_OURS_CUDA = "R010"
RUN_ID_LS = "R011"
RUN_ID_GAP = "R013"
UPSTREAM_SHA = "5b9e24f"
OURS_SHA = "25b7ecb"
CUOPT_PIN = "26.10.01"  # cuopt-cu12 from pypi.nvidia.com (gpu-ready)
DEFAULT_TOPK = 2  # R006: 127/127 time wins; K≥3 still loses time on hard rows
GAP_TOPK = 6  # R013 gap-first polish
GAP_LS = 2  # Or-opt/relocate + limited 3-opt
# Dense float32 N×N cost matrix: n=25k ≈ 2.5 GB; H100 80 GB leaves solver headroom.
CUOPT_MAX_N = 25000
CUOPT_TIME_LIMIT_S = 30.0
# Host OpenMP cores on the H100 box (bucketed solver; no dense N×N / no cuOpt).
OURS_GPU_OMP = 48
CUDA_PRIM_MIN = 512
CUDA_2OPT_MIN = 128
NOTES_OURS = (
    f"run={RUN_ID_OURS} bpmds@{UPSTREAM_SHA} ours@{OURS_SHA} cuopt@na "
    f"hw=Modal-48c-256G timed=Solver::solve rho=10000 topk={DEFAULT_TOPK}"
)
NOTES_BPMDS = (
    f"run={RUN_ID_BPMDS} bpmds@{UPSTREAM_SHA} ours@{OURS_SHA} cuopt@na "
    "hw=Modal-48c-256G timed=Solver::solve rho=10000"
)
NOTES_PHASE_B = (
    f"run={RUN_ID_PHASE_B} bpmds@{UPSTREAM_SHA} ours@{OURS_SHA} cuopt@na "
    f"hw=Modal-48c-256G timed=Solver::solve phase=B topk={DEFAULT_TOPK}"
)
NOTES_CUOPT = (
    f"run={RUN_ID_CUOPT} bpmds@{UPSTREAM_SHA} ours@{OURS_SHA} cuopt@{CUOPT_PIN} "
    f"hw=Modal-H100 timed=routing.Solve time_limit={CUOPT_TIME_LIMIT_S}s "
    f"max_n={CUOPT_MAX_N}"
)
NOTES_OURS_GPU = (
    f"run={RUN_ID_OURS_GPU} bpmds@{UPSTREAM_SHA} ours@{OURS_SHA} cuopt@{CUOPT_PIN} "
    f"hw=Modal-H100 timed=Solver::solve lane=ours-gpu "
    f"algo=bucketed-euclidean omp={OURS_GPU_OMP} topk={DEFAULT_TOPK}"
)
NOTES_OURS_CUDA = (
    f"run={RUN_ID_OURS_CUDA} bpmds@{UPSTREAM_SHA} ours@{OURS_SHA} cuopt@{CUOPT_PIN} "
    f"hw=Modal-H100 timed=Solver::solve lane=ours-gpu-cuda "
    f"algo=bucketed-euclidean+cuda-prim-2opt omp={OURS_GPU_OMP} "
    f"prim_min={CUDA_PRIM_MIN} twopt_min={CUDA_2OPT_MIN} topk={DEFAULT_TOPK}"
)
NOTES_LS = (
    f"run={RUN_ID_LS} bpmds@{UPSTREAM_SHA} ours@{OURS_SHA} cuopt@{CUOPT_PIN} "
    f"hw=Modal-48c-256G timed=Solver::solve ls=2 "
    f"ops=2opt+oropt+relocate+3opt topk={DEFAULT_TOPK}"
)
NOTES_GAP = (
    f"run={RUN_ID_GAP} bpmds@{UPSTREAM_SHA} ours@{OURS_SHA} cuopt@na "
    f"hw=Modal-48c-256G timed=Solver::solve gap-first "
    f"tier=A topk={GAP_TOPK} ls={GAP_LS}"
)
RUN_ID_GAP_B = "R013b"
NOTES_GAP_B = (
    f"run={RUN_ID_GAP_B} bpmds@{UPSTREAM_SHA} ours@{OURS_SHA} cuopt@na "
    f"hw=Modal-48c-256G timed=Solver::solve dual-constraint "
    f"gap↓+time≤bpmds+gap≤bpmds"
)
RUN_ID_GAP_C = "R013c"
NOTES_GAP_C = (
    f"run={RUN_ID_GAP_C} bpmds@{UPSTREAM_SHA} ours@{OURS_SHA} cuopt@na "
    f"hw=Modal-48c-256G timed=Solver::solve dual-constraint "
    f"slack-spend gap↓+time≤bpmds+gap≤bpmds"
)
RUN_ID_LSX = "R014"
NOTES_LSX = (
    f"run={RUN_ID_LSX} bpmds@{UPSTREAM_SHA} ours@{OURS_SHA} cuopt@na "
    f"hw=Modal-48c-256G timed=Solver::solve dual-constraint "
    f"lsx=exchange+cross+span3opt gap↓+time≤bpmds+gap≤bpmds"
)
# R015: quality-first — soft time budget vs BP-MDS; hunt FILO2x gap / sweet spot.
RUN_ID_QUAL = "R015"
QUALITY_TIME_MULT = 1.5  # allow up to 1.5× bpmds_time while searching gap
NOTES_QUAL = (
    f"run={RUN_ID_QUAL} bpmds@{UPSTREAM_SHA} ours@{OURS_SHA} cuopt@na "
    f"hw=Modal-48c-256G timed=Solver::solve quality-first "
    f"soft_time≤{QUALITY_TIME_MULT}x_bpmds hunt=FILO2x-gap"
)
# Near + mid FILO2x losers for quality-first sweet-spot search.
TIER_QUAL = (
    "Toscana",
    "Basilicata",
    "Lombardia",
    "Friuli-Venezia-Giulia",
    "Campania",
    "Abruzzo",
)
# keep old name for ours path compatibility
RUN_ID = RUN_ID_OURS
NOTES = NOTES_OURS

# R013a: never-Phase-B I instances (largest unused α lever).
TIER_A = (
    "Valle-D-Aosta",
    "Umbria",
    "Trentino-Alto-Adige",
    "Calabria",
)
# R013b: close then mid FILO2x gap losers (cheap knobs).
TIER_D = ("Toscana", "Basilicata", "Lombardia")
TIER_C = (
    "Lazio",
    "Marche",
    "Veneto",
    "Abruzzo",
    "Campania",
    "Friuli-Venezia-Giulia",
)
# R013c: near-miss + fat time-slack vs BP-MDS (spend headroom).
TIER_SLACK = (
    "Toscana",  # closest to FILO2x (~+0.045 pp)
    "Valle-D-Aosta",
    "Umbria",
    "Trentino-Alto-Adige",
    "Liguria",
    "Calabria",
    "Veneto",
)

# Remaining large+BKS gap losers after R006 (topk=2, ρ=10k).
PHASE_B_LOSERS = (
    "Friuli-Venezia-Giulia",
    "Sardegna",
    "Marche",
    "Campania",
    "Lazio",
    "Basilicata",
    "Molise",
    "Abruzzo",
    "Piemonte",
    "Emilia-Romagna",
    "Liguria",
    "Puglia",
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
    "**/*.log",
    "modal_exp/results",
    "scoreboard.csv",
    "scoreboard.tsv",
    "benchmarks/scoreboard.csv",
    "benchmarks/scoreboard.tsv",
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

# GPU image for NVIDIA cuOpt (dense-matrix CVRP). Separate from CPU BP-MDS image.
gpu_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-runtime-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git", "wget", "ca-certificates")
    .pip_install("numpy")
    .pip_install(
        "cuopt-cu12",
        extra_options="--extra-index-url=https://pypi.nvidia.com",
    )
    .add_local_dir(
        str(REPO / "modal_exp"),
        remote_path="/modal_exp",
        ignore=["**/__pycache__", "**/*.log", "results/*.jsonl", "results/*.log"],
    )
)

# Devel image: nvcc + g++ for ours-gpu CUDA Prim / 2-opt (still no dense N×N).
cuda_solver_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("g++", "make", "libgomp1", "git", "wget", "ca-certificates")
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
    # Always rebuild so Solver/topk edits from add_local_dir are picked up.
    binary = Path("/tmp/bin/ours-bucket-partitioned-MDS")
    if binary.is_file():
        binary.unlink()
    return _build_solver(Path("/repo"), "ours-bucket-partitioned-MDS")


def _build_ours_cuda() -> Path:
    """Build BP-MDS with BPMDS_USE_CUDA (nvcc). H100 = sm_90."""
    src = Path("/repo")
    out = Path("/tmp/bin")
    out.mkdir(parents=True, exist_ok=True)
    binary = out / "ours-bucket-partitioned-MDS-cuda"
    if binary.is_file():
        binary.unlink()
    common = [
        "Src/Main.cpp",
        "Lib/Bucket_Partitioned_MDS/CVRP.cpp",
        "Lib/Bucket_Partitioned_MDS/Solution.cpp",
        "Lib/Command_Line_Args.cpp",
        "Lib/Initializer.cpp",
        "Lib/Bucket_Partitioned_MDS/Solver.cpp",
        "Lib/Cuda/cuda_hotpath.cu",
    ]
    utils = sorted((src / "Lib" / "Utils").rglob("*.cpp"))
    cmd = [
        "nvcc",
        "-O3",
        "-std=c++17",
        "-arch=sm_90",
        "-Xcompiler",
        "-fopenmp,-O3",
        "-DBPMDS_USE_CUDA",
        "-IInclude",
        *[str(src / c) for c in common],
        *[str(u) for u in utils],
        "-o",
        str(binary),
        "-lgomp",
        "-lcudart",
    ]
    subprocess.check_call(cmd, cwd=src)
    return binary


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
    topk: int = DEFAULT_TOPK,
    ls: int = 0,
) -> dict:
    os.environ["OMP_NUM_THREADS"] = str(int(CPU_CORES))
    binary = _build_ours()
    notes = (
        NOTES_QUAL
        if run_id == RUN_ID_QUAL
        else NOTES_LSX
        if run_id == RUN_ID_LSX
        else NOTES_GAP_C
        if run_id == RUN_ID_GAP_C
        else NOTES_GAP_B
        if run_id == RUN_ID_GAP_B
        else NOTES_GAP
        if run_id == RUN_ID_GAP
        else NOTES_LS
        if run_id == RUN_ID_LS
        else NOTES_PHASE_B
        if run_id == RUN_ID_PHASE_B
        else NOTES_OURS
    )
    notes = f"{notes} alpha={alpha} rho={rho} topk={topk} ls={ls}"
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
            "topk": topk,
            "ls": ls,
        }
        _append_jsonl(Path(f"/data/results/{run_id}_ours.jsonl"), row)
        vol.commit()
        return row

    if sol_tag:
        out_dir = Path(
            "/data/Output/ours/gap"
            if run_id in {
                RUN_ID_GAP,
                RUN_ID_GAP_B,
                RUN_ID_GAP_C,
                RUN_ID_LSX,
                RUN_ID_QUAL,
            }
            else "/data/Output/ours/phaseb"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        sol = out_dir / f"{instance}_{sol_tag}.sol"
    else:
        out_dir = Path("/data/Output/ours" + ("_ls" if run_id == RUN_ID_LS else ""))
        out_dir.mkdir(parents=True, exist_ok=True)
        sol = out_dir / f"{instance}.sol"

    wall0 = time.perf_counter()
    proc = subprocess.run(
        [
            str(binary),
            f"--alpha={alpha}",
            f"--rho={rho}",
            f"--topk={int(topk)}",
            f"--ls={int(ls)}",
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
        "topk": int(topk),
        "ls": int(ls),
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


@app.function(image=gpu_image, gpu=GPU, timeout=60 * 30, memory=65536)
def gpu_ready() -> dict:
    out = subprocess.check_output(["nvidia-smi", "-L"], text=True).strip()
    import cuopt

    return {
        "gpu_request": GPU,
        "nvidia_smi": out,
        "cuopt_version": getattr(cuopt, "__version__", "unknown"),
        "cuopt_pin": CUOPT_PIN,
    }


def _gap_pct(cost: float | None, bks: str) -> float | None:
    return _gap(cost, bks)


@app.function(
    image=gpu_image,
    gpu=GPU,
    timeout=60 * 60 * 2,
    memory=131072,  # 128 GiB host RAM for large dense matrices
    volumes={"/data": vol},
)
def run_cuopt_one(
    instance: str,
    path: str,
    bks: str,
    time_limit_s: float = CUOPT_TIME_LIMIT_S,
    max_n: int = CUOPT_MAX_N,
) -> dict:
    """NVIDIA cuOpt on one large instance (skip if n > max_n)."""
    import sys

    sys.path.insert(0, "/modal_exp")
    from cuopt_solve import solve_vrp_path  # noqa: WPS433
    import cuopt

    vrp = Path("/data/Inputs") / path
    notes = (
        f"{NOTES_CUOPT} cuopt_ver={getattr(cuopt, '__version__', '?')}"
    )
    if not vrp.is_file():
        row = {
            "ok": False,
            "run_id": RUN_ID_CUOPT,
            "solver": "cuopt",
            "instance": instance,
            "path": path,
            "error": f"missing {vrp}",
            "notes": notes,
        }
        _append_jsonl(Path(f"/data/results/{RUN_ID_CUOPT}_cuopt.jsonl"), row)
        vol.commit()
        return row

    # Quick dimension peek without full parse of huge files when possible.
    dim = None
    with vrp.open(errors="replace") as f:
        for line in f:
            if line.upper().startswith("DIMENSION"):
                dim = int(line.split(":")[-1].strip().split()[0])
                break
    if dim is not None and dim > int(max_n):
        row = {
            "ok": False,
            "run_id": RUN_ID_CUOPT,
            "solver": "cuopt",
            "instance": instance,
            "path": path,
            "n": dim,
            "skipped": True,
            "error": f"n={dim} > cuopt max_n={max_n} (dense matrix)",
            "notes": notes,
        }
        _append_jsonl(Path(f"/data/results/{RUN_ID_CUOPT}_cuopt.jsonl"), row)
        vol.commit()
        return row

    try:
        result = solve_vrp_path(str(vrp), time_limit_s=float(time_limit_s))
    except Exception as exc:  # noqa: BLE001 — record OOM/API failures
        row = {
            "ok": False,
            "run_id": RUN_ID_CUOPT,
            "solver": "cuopt",
            "instance": instance,
            "path": path,
            "error": f"{type(exc).__name__}: {exc}",
            "notes": notes,
        }
        _append_jsonl(Path(f"/data/results/{RUN_ID_CUOPT}_cuopt.jsonl"), row)
        vol.commit()
        return row

    gap = _gap_pct(result.get("cuopt_cost"), bks)
    row = {
        "ok": bool(result.get("ok")),
        "run_id": RUN_ID_CUOPT,
        "solver": "cuopt",
        "instance": instance,
        "path": path,
        "bks": bks,
        "cuopt_cost": result.get("cuopt_cost"),
        "cuopt_gap_pct": gap,
        "cuopt_time_s": result.get("cuopt_time_s"),
        "matrix_build_s": result.get("matrix_build_s"),
        "n": result.get("n"),
        "n_fleet": result.get("n_fleet"),
        "n_routes": result.get("n_routes"),
        "status": result.get("status"),
        "time_limit_s": time_limit_s,
        "cuopt_version": getattr(cuopt, "__version__", "unknown"),
        "notes": notes,
    }
    _append_jsonl(Path(f"/data/results/{RUN_ID_CUOPT}_cuopt.jsonl"), row)
    patch = Path(f"/data/results/{RUN_ID_CUOPT}_cuopt_patch.csv")
    write_header = not patch.is_file()
    with patch.open("a", newline="") as f:
        fields = [
            "instance",
            "cuopt_cost",
            "cuopt_gap_pct",
            "cuopt_time_s",
            "notes",
            "ok",
            "n",
            "status",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()

        w.writerow(
            {
                "instance": instance,
                "cuopt_cost": ""
                if result.get("cuopt_cost") is None
                else f"{float(result['cuopt_cost']):.6f}",
                "cuopt_gap_pct": "" if gap is None else f"{gap:.6f}",
                "cuopt_time_s": ""
                if result.get("cuopt_time_s") is None
                else f"{float(result['cuopt_time_s']):.6f}",
                "notes": notes,
                "ok": row["ok"],
                "n": result.get("n"),
                "status": result.get("status"),
            }
        )
    vol.commit()
    return row


def _run_ours_gpu_impl(
    instance: str,
    path: str,
    alpha: float,
    rho: int,
    bks: str,
    topk: int,
    *,
    use_cuda: bool,
) -> dict:
    """Shared ours-gpu / ours-cuda solve body."""
    omp = str(int(OURS_GPU_OMP))
    os.environ["OMP_NUM_THREADS"] = omp
    run_id = RUN_ID_OURS_CUDA if use_cuda else RUN_ID_OURS_GPU
    notes_base = NOTES_OURS_CUDA if use_cuda else NOTES_OURS_GPU
    solver = "ours_gpu_cuda" if use_cuda else "ours_gpu"
    if use_cuda:
        os.environ["BPMDS_CUDA"] = "1"
        os.environ["BPMDS_CUDA_PRIM_MIN"] = str(CUDA_PRIM_MIN)
        os.environ["BPMDS_CUDA_2OPT_MIN"] = str(CUDA_2OPT_MIN)
        binary = _build_ours_cuda()
    else:
        os.environ.pop("BPMDS_CUDA", None)
        binary = _build_ours()
    notes = f"{notes_base} alpha={alpha} rho={rho} topk={topk}"
    smi = ""
    try:
        smi = subprocess.check_output(["nvidia-smi", "-L"], text=True).strip()
    except Exception as exc:  # noqa: BLE001
        smi = f"nvidia-smi_failed:{exc}"

    vrp = Path("/data/Inputs") / path
    tag = "ours_gpu_cuda" if use_cuda else "ours_gpu"
    if not vrp.is_file():
        row = {
            "ok": False,
            "run_id": run_id,
            "solver": solver,
            "instance": instance,
            "path": path,
            "error": f"missing {vrp}",
            "notes": notes,
            "nvidia_smi": smi,
            "use_cuda": use_cuda,
        }
        _append_jsonl(Path(f"/data/results/{run_id}_{tag}.jsonl"), row)
        vol.commit()
        return row

    out_dir = Path(f"/data/Output/{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    sol = out_dir / f"{instance}.sol"

    wall0 = time.perf_counter()
    proc = subprocess.run(
        [
            str(binary),
            f"--alpha={alpha}",
            f"--rho={rho}",
            f"--topk={int(topk)}",
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
        "run_id": run_id,
        "solver": solver,
        "instance": instance,
        "path": path,
        "alpha": alpha,
        "rho": rho,
        "topk": int(topk),
        "bks": bks,
        "ours_gpu_cost": parsed["cost"],
        "ours_gpu_gap_pct": gap,
        "ours_gpu_time_s": time_s,
        "wall_s": wall,
        "mem_mb": parsed["mem_mb"],
        "returncode": proc.returncode,
        "output": str(sol),
        "omp": omp,
        "nvidia_smi": smi,
        "use_cuda": use_cuda,
        "notes": notes,
        "stderr_tail": proc.stderr[-1500:],
        "stdout_tail": proc.stdout[-1500:],
    }
    _append_jsonl(Path(f"/data/results/{run_id}_{tag}.jsonl"), row)
    patch = Path(f"/data/results/{run_id}_{tag}_patch.csv")
    write_header = not patch.is_file()
    with patch.open("a", newline="") as f:
        fields = [
            "instance",
            "ours_gpu_cost",
            "ours_gpu_gap_pct",
            "ours_gpu_time_s",
            "notes",
            "ok",
            "alpha",
            "rho",
            "topk",
            "mem_mb",
            "use_cuda",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow(
            {
                "instance": instance,
                "ours_gpu_cost": ""
                if parsed["cost"] is None
                else f"{parsed['cost']:.6f}",
                "ours_gpu_gap_pct": "" if gap is None else f"{gap:.6f}",
                "ours_gpu_time_s": "" if time_s is None else f"{time_s:.6f}",
                "notes": notes,
                "ok": row["ok"],
                "alpha": alpha,
                "rho": rho,
                "topk": topk,
                "mem_mb": parsed["mem_mb"],
                "use_cuda": use_cuda,
            }
        )
    vol.commit()
    return row


@app.function(
    image=image,
    gpu=GPU,
    cpu=OURS_GPU_OMP,
    memory=131072,  # 128 GiB host RAM — O(n) bucketed, not dense N×N
    timeout=60 * 60 * 6,
    volumes={"/data": vol},
)
def run_ours_gpu_one(
    instance: str,
    path: str,
    alpha: float,
    rho: int,
    bks: str,
    topk: int = DEFAULT_TOPK,
) -> dict:
    """GPU-lane ours: OpenMP on H100 host (no CUDA kernels)."""
    return _run_ours_gpu_impl(
        instance, path, alpha, rho, bks, topk, use_cuda=False
    )


@app.function(
    image=cuda_solver_image,
    gpu=GPU,
    cpu=OURS_GPU_OMP,
    memory=131072,
    timeout=60 * 60 * 6,
    volumes={"/data": vol},
)
def run_ours_cuda_one(
    instance: str,
    path: str,
    alpha: float,
    rho: int,
    bks: str,
    topk: int = DEFAULT_TOPK,
) -> dict:
    """GPU-lane ours with CUDA Prim / 2-opt inside buckets (still no dense N×N)."""
    return _run_ours_gpu_impl(
        instance, path, alpha, rho, bks, topk, use_cuda=True
    )


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
                DEFAULT_TOPK,
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
    # Tighter grid (aquin cancelled the 216-job map). Still topk=2 only.
    return _phase_b_job_builder(
        by_name,
        PHASE_B_LOSERS,
        alpha_deltas=(-1.0, -0.5, -0.25, 0.25, 0.5, 1.0),
        rhos_at_base=(20000, 30000, 40000, 60000),
        cross_rho=40000,
        cross_deltas=(-0.5, 0.5),
    )


def _phase_b_batch_jobs(
    by_name: dict[str, dict], batch: int
) -> tuple[list[tuple], list[str]]:
    """Split the 12 losers into two batches (1 or 2) to avoid map cancellations."""
    mid = (len(PHASE_B_LOSERS) + 1) // 2
    if batch == 1:
        targets = PHASE_B_LOSERS[:mid]
    else:
        targets = PHASE_B_LOSERS[mid:]
    jobs = _phase_b_job_builder(
        by_name,
        targets,
        alpha_deltas=(-1.0, -0.5, -0.25, 0.25, 0.5, 1.0),
        rhos_at_base=(20000, 30000, 40000, 60000),
        cross_rho=40000,
        cross_deltas=(-0.5, 0.5),
    )
    return jobs, list(targets)


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
                DEFAULT_TOPK,
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
    """Friuli-only: cross bucket-count regimes + very high ρ (topk=2)."""
    # ceil(360/α): 26→[13.846,14.4), 25→[14.4,15.0), 24→[15.0,15.652)
    return _phase_b_grid_jobs(
        by_name,
        {
            "Friuli-Venezia-Giulia": {
                "alphas": (
                    13.9,
                    14.1,
                    14.25,
                    14.4,
                    14.5,
                    14.65,
                    14.75,
                    14.85,
                    14.9,
                    15.0,
                    15.2,
                    15.4,
                ),
                "rhos": (40000, 60000, 80000, 100000),
            },
        },
    )


def _phase_b5_jobs(by_name: dict[str, dict]) -> list[tuple]:
    """Friuli micro-grid: last ~1e-5 pp at topk=2."""
    return _phase_b_grid_jobs(
        by_name,
        {
            "Friuli-Venezia-Giulia": {
                "alphas": (
                    14.68,
                    14.7,
                    14.72,
                    14.74,
                    14.75,
                    14.76,
                    14.78,
                    14.8,
                    14.82,
                ),
                "rhos": (55000, 60000, 65000, 70000, 75000, 90000),
            },
        },
    )


def _r013a_grids() -> dict[str, dict]:
    return {
        "Valle-D-Aosta": {
            "alphas": (48.0, 50.0, 52.0, 54.0, 56.0),
            "rhos": (10000, 20000, 40000, 60000),
        },
        "Umbria": {
            "alphas": (9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5),
            "rhos": (10000, 20000, 40000, 60000),
        },
        "Trentino-Alto-Adige": {
            "alphas": (30.0, 32.0, 33.0, 34.0, 35.0, 36.0, 38.0),
            "rhos": (10000, 20000, 40000, 60000),
        },
        "Calabria": {
            "alphas": (8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5),
            "rhos": (10000, 20000, 40000, 60000),
        },
    }


def _r013a_jobs(
    by_name: dict[str, dict],
    *,
    skip: set[tuple[str, float, int]] | None = None,
) -> list[tuple]:
    """Tier A never-Phase-B: α×ρ grid at gap-first topk=6 ls=2."""
    skip = skip or set()
    jobs: list[tuple] = []
    seen: set[tuple[str, float, int]] = set()
    for inst, g in _r013a_grids().items():
        r = by_name.get(inst)
        if not r:
            continue
        for a in g["alphas"]:
            alpha = round(float(a), 4)
            if alpha <= 0 or alpha > 360:
                continue
            for rho in g["rhos"]:
                key = (inst, alpha, int(rho))
                if key in seen or key in skip:
                    continue
                seen.add(key)
                tag = f"a{alpha:g}_r{int(rho)}_k{GAP_TOPK}_ls{GAP_LS}"
                jobs.append(
                    (
                        r["instance"],
                        r["path"],
                        alpha,
                        int(rho),
                        r.get("bks", ""),
                        RUN_ID_GAP,
                        tag,
                        GAP_TOPK,
                        GAP_LS,
                    )
                )
    return jobs


def _r013_done_keys(jsonl: Path) -> set[tuple[str, float, int]]:
    done: set[tuple[str, float, int]] = set()
    if not jsonl.is_file():
        return done
    for line in jsonl.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if not r.get("ok") or r.get("ours_cost") is None:
            continue
        done.add((r["instance"], float(r["alpha"]), int(r["rho"])))
    return done


def _load_knobs() -> dict[str, dict]:
    path = REPO / "modal_exp" / "results" / "R009_knobs.json"
    return json.loads(path.read_text())


def _r013b_jobs(
    by_name: dict[str, dict],
    instances: tuple[str, ...],
    *,
    topks: tuple[int, ...] = (2, 3, 4),
    lss: tuple[int, ...] = (1, 2),
    rho_mults: tuple[float, ...] = (1.0, 1.5, 2.0),
    run_id: str = RUN_ID_GAP_B,
) -> list[tuple]:
    """Dual-constraint grid: live α × ρ multiples × topk × ls."""
    knobs = _load_knobs()
    jobs: list[tuple] = []
    seen: set[tuple[str, float, int, int, int]] = set()
    for inst in instances:
        r = by_name.get(inst)
        k = knobs.get(inst)
        if not r or not k:
            continue
        alpha = round(float(k["alpha"]), 4)
        base_rho = int(k["rho"])
        for mult in rho_mults:
            rho = int(round(base_rho * mult))
            if rho <= 0:
                continue
            for topk in topks:
                for ls in lss:
                    key = (inst, alpha, rho, int(topk), int(ls))
                    if key in seen:
                        continue
                    seen.add(key)
                    tag = f"a{alpha:g}_r{rho}_k{int(topk)}_ls{int(ls)}"
                    jobs.append(
                        (
                            r["instance"],
                            r["path"],
                            alpha,
                            rho,
                            r.get("bks", ""),
                            run_id,
                            tag,
                            int(topk),
                            int(ls),
                        )
                    )
    return jobs


def _r015_summarize(
    results: list[dict],
    by_name: dict[str, dict],
    targets: list[str],
    *,
    time_mult: float = QUALITY_TIME_MULT,
) -> tuple[dict[str, dict], dict[str, dict], list[dict]]:
    """Quality-first: best gap under soft time; also best hard dual-win."""
    best_soft: dict[str, dict] = {}
    best_dual: dict[str, dict] = {}
    best_f2x: dict[str, dict] = {}
    for x in results:
        if not x.get("ok") or x.get("ours_cost") is None:
            continue
        inst = x["instance"]
        row = by_name.get(inst) or {}
        try:
            gap = float(x["ours_gap_pct"])
            time_s = float(x["ours_time_s"])
            bpmds_gap = float(row["bpmds_gap_pct"])
            bpmds_t = float(row["bpmds_time_s"])
            f2x = float(row["filo2x_gap_pct"]) if row.get("filo2x_gap_pct") else None
        except (TypeError, ValueError, KeyError):
            continue
        soft_cap = bpmds_t * time_mult
        if time_s <= soft_cap + 1e-9:
            cur = best_soft.get(inst)
            if cur is None or float(x["ours_cost"]) < float(cur["ours_cost"]):
                best_soft[inst] = x
        if (
            time_s <= bpmds_t + 1e-9
            and gap <= bpmds_gap + 1e-9
        ):
            # dual-feasible vs BP-MDS (gap vs sheet checked at apply time)
            cur = best_dual.get(inst)
            if cur is None or float(x["ours_cost"]) < float(cur["ours_cost"]):
                best_dual[inst] = x
        if f2x is not None and gap <= f2x + 1e-9 and time_s <= soft_cap + 1e-9:
            cur = best_f2x.get(inst)
            if cur is None or time_s < float(cur["ours_time_s"]):
                best_f2x[inst] = x

    summary = []
    for inst in targets:
        row = by_name.get(inst) or {}
        sheet_gap = float(row["ours_gap_pct"]) if row.get("ours_gap_pct") else None
        f2x = float(row["filo2x_gap_pct"]) if row.get("filo2x_gap_pct") else None
        bpmds_t = float(row["bpmds_time_s"]) if row.get("bpmds_time_s") else None
        soft = best_soft.get(inst)
        dual = best_dual.get(inst)
        f2 = best_f2x.get(inst)

        def pack(x: dict | None) -> dict | None:
            if not x:
                return None
            g = float(x["ours_gap_pct"])
            t = float(x["ours_time_s"])
            return {
                "gap": g,
                "time_s": t,
                "alpha": x.get("alpha"),
                "rho": x.get("rho"),
                "topk": x.get("topk"),
                "ls": x.get("ls"),
                "vs_sheet": (g - sheet_gap) if sheet_gap is not None else None,
                "vs_filo2x": (g - f2x) if f2x is not None else None,
                "time_vs_bpmds": (t / bpmds_t) if bpmds_t else None,
            }

        soft_p = pack(soft)
        dual_p = pack(dual)
        # Sweet spot: beats FILO2x and still ≤ bpmds time (and ≤ bpmds gap).
        sweet = False
        if f2 and bpmds_t is not None:
            try:
                sweet = (
                    float(f2["ours_time_s"]) <= bpmds_t + 1e-9
                    and float(f2["ours_gap_pct"])
                    <= float(row["bpmds_gap_pct"]) + 1e-9
                )
            except (TypeError, ValueError, KeyError):
                sweet = False
        summary.append(
            {
                "instance": inst,
                "sheet_gap_pct": sheet_gap,
                "filo2x_gap_pct": f2x,
                "bpmds_time_s": bpmds_t,
                "soft_cap_s": (bpmds_t * time_mult) if bpmds_t else None,
                "best_soft": soft_p,
                "best_dual": dual_p,
                "clears_filo2x_soft": pack(f2),
                "sweet_spot": sweet,
            }
        )
    return best_soft, best_dual, summary


def _r013b_summarize(
    results: list[dict],
    by_name: dict[str, dict],
    targets: list[str],
) -> tuple[dict[str, dict], list[dict]]:
    """Best dual-constraint candidate per instance (vs live sheet)."""
    best: dict[str, dict] = {}
    for x in results:
        if not x.get("ok") or x.get("ours_cost") is None:
            continue
        inst = x["instance"]
        row = by_name.get(inst) or {}
        try:
            gap = float(x["ours_gap_pct"])
            time_s = float(x["ours_time_s"])
            sheet_gap = float(row["ours_gap_pct"])
            bpmds_gap = float(row["bpmds_gap_pct"])
            bpmds_t = float(row["bpmds_time_s"])
        except (TypeError, ValueError, KeyError):
            continue
        if not (
            gap < sheet_gap - 1e-9
            and time_s <= bpmds_t + 1e-9
            and gap <= bpmds_gap + 1e-9
        ):
            continue
        cur = best.get(inst)
        if cur is None or float(x["ours_cost"]) < float(cur["ours_cost"]):
            best[inst] = x

    summary = []
    for inst in targets:
        row = by_name.get(inst) or {}
        b = best.get(inst)
        sheet_gap = float(row["ours_gap_pct"]) if row.get("ours_gap_pct") else None
        f2x = float(row["filo2x_gap_pct"]) if row.get("filo2x_gap_pct") else None
        if not b:
            # also report best gap among time-ok even if not improved
            time_ok = []
            for x in results:
                if x.get("instance") != inst or not x.get("ok"):
                    continue
                try:
                    if float(x["ours_time_s"]) <= float(row["bpmds_time_s"]) + 1e-9:
                        time_ok.append(x)
                except (TypeError, ValueError, KeyError):
                    continue
            tip = min(time_ok, key=lambda z: float(z["ours_cost"])) if time_ok else None
            summary.append(
                {
                    "instance": inst,
                    "ok": False,
                    "dual_win": False,
                    "sheet_gap_pct": sheet_gap,
                    "filo2x_gap_pct": f2x,
                    "best_time_ok_gap": float(tip["ours_gap_pct"]) if tip else None,
                    "best_time_ok_time": float(tip["ours_time_s"]) if tip else None,
                }
            )
            continue
        gap = float(b["ours_gap_pct"])
        summary.append(
            {
                "instance": inst,
                "ok": True,
                "dual_win": True,
                "alpha": b["alpha"],
                "rho": b["rho"],
                "topk": b.get("topk"),
                "ls": b.get("ls"),
                "ours_gap_pct": gap,
                "sheet_gap_pct": sheet_gap,
                "filo2x_gap_pct": f2x,
                "d_vs_sheet": gap - sheet_gap if sheet_gap is not None else None,
                "d_vs_filo2x": gap - f2x if f2x is not None else None,
                "ours_time_s": b.get("ours_time_s"),
                "gap_le_filo2x": f2x is not None and gap <= f2x + 1e-9,
            }
        )
    return best, summary


@app.local_entrypoint()
def main(
    cmd: str = "list-large",
    instance: str = "",
    rho: int = 10000,
    alpha: float = 0.0,
    topk: int = DEFAULT_TOPK,
    ls: int = 0,
    time_limit: float = CUOPT_TIME_LIMIT_S,
    max_n: int = CUOPT_MAX_N,
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

    if cmd == "cuopt-one":
        if instance not in by_name:
            raise SystemExit(f"{instance} not in large suite (L/XL/XXL/XXXL)")
        r = by_name[instance]
        print(
            run_cuopt_one.remote(
                r["instance"],
                r["path"],
                r.get("bks", ""),
                float(time_limit),
                int(max_n),
            )
        )
        return
    if cmd == "cuopt-fit":
        # All large rows with n ≤ max_n (dense-matrix feasible on H100).
        jobs = []
        for r in rows:
            try:
                n = int(r["n"])
            except (TypeError, ValueError):
                continue
            if n <= int(max_n):
                jobs.append(
                    (
                        r["instance"],
                        r["path"],
                        r.get("bks", ""),
                        float(time_limit),
                        int(max_n),
                    )
                )
        print({"cuopt_fit_jobs": len(jobs), "max_n": max_n, "time_limit": time_limit})
        results = list(run_cuopt_one.starmap(jobs))
        ok = sum(1 for x in results if x.get("ok"))
        skipped = sum(1 for x in results if x.get("skipped"))
        out = REPO / "modal_exp" / "results" / f"{RUN_ID_CUOPT}_cuopt.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            for x in results:
                f.write(json.dumps(x, sort_keys=True) + "\n")
        print(
            {
                "ok": ok,
                "skipped": skipped,
                "failed": len(results) - ok - skipped,
                "local_jsonl": str(out),
            }
        )
        return

    knobs_path = REPO / "modal_exp" / "results" / "R009_knobs.json"
    knobs: dict = {}
    if knobs_path.is_file():
        knobs = json.loads(knobs_path.read_text())

    def _gpu_knob(r: dict) -> tuple[float, int, int]:
        k = knobs.get(r["instance"], {})
        a = float(k["alpha"]) if k.get("alpha") is not None else float(r["alpha_bpmds"] or 0)
        if alpha > 0:
            a = float(alpha)
        rh = int(k["rho"]) if k.get("rho") is not None else int(rho)
        tk = int(k["topk"]) if k.get("topk") is not None else int(topk)
        return a, rh, tk

    if cmd == "ours-ls-bks":
        # Stronger LS (default ls=2) on large+BKS with R009 knobs.
        ls_v = int(ls) if ls > 0 else 2
        jobs = []
        for r in rows:
            if not (r.get("bks") or "").strip():
                continue
            a, rh, tk = _gpu_knob(r)
            jobs.append(
                (
                    r["instance"],
                    r["path"],
                    a,
                    rh,
                    r.get("bks", ""),
                    RUN_ID_LS,
                    "",
                    tk,
                    ls_v,
                )
            )
        print({"ours_ls_bks_jobs": len(jobs), "ls": ls_v})
        results = list(run_ours_one.starmap(jobs))
        ok = sum(1 for x in results if x.get("ok"))
        out = (
            REPO
            / "modal_exp"
            / "results"
            / f"{RUN_ID_LS}_ours_ls{ls_v}.jsonl"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            for x in results:
                f.write(json.dumps(x, sort_keys=True) + "\n")
        print({"ok": ok, "failed": len(results) - ok, "local_jsonl": str(out)})
        return
    if cmd == "ours-gpu-one":
        if instance not in by_name:
            raise SystemExit(f"{instance} not in large suite (L/XL/XXL/XXXL)")
        r = by_name[instance]
        a, rh, tk = _gpu_knob(r)
        print(
            run_ours_gpu_one.remote(
                r["instance"], r["path"], a, rh, r.get("bks", ""), tk
            )
        )
        return
    if cmd == "ours-cuda-one":
        if instance not in by_name:
            raise SystemExit(f"{instance} not in large suite (L/XL/XXL/XXXL)")
        r = by_name[instance]
        a, rh, tk = _gpu_knob(r)
        print(
            run_ours_cuda_one.remote(
                r["instance"], r["path"], a, rh, r.get("bks", ""), tk
            )
        )
        return
    if cmd == "ours-gpu-bks":
        # All large+BKS (25): includes n>25k where stock cuOpt is skip.
        jobs = []
        for r in rows:
            if not (r.get("bks") or "").strip():
                continue
            a, rh, tk = _gpu_knob(r)
            jobs.append(
                (r["instance"], r["path"], a, rh, r.get("bks", ""), tk)
            )
        print({"ours_gpu_bks_jobs": len(jobs), "knobs": str(knobs_path)})
        results = list(run_ours_gpu_one.starmap(jobs))
        ok = sum(1 for x in results if x.get("ok"))
        out = REPO / "modal_exp" / "results" / f"{RUN_ID_OURS_GPU}_ours_gpu.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            for x in results:
                f.write(json.dumps(x, sort_keys=True) + "\n")
        print({"ok": ok, "failed": len(results) - ok, "local_jsonl": str(out)})
        return
    if cmd == "ours-cuda-bks":
        jobs = []
        for r in rows:
            if not (r.get("bks") or "").strip():
                continue
            a, rh, tk = _gpu_knob(r)
            jobs.append(
                (r["instance"], r["path"], a, rh, r.get("bks", ""), tk)
            )
        print({"ours_cuda_bks_jobs": len(jobs), "knobs": str(knobs_path)})
        results = list(run_ours_cuda_one.starmap(jobs))
        ok = sum(1 for x in results if x.get("ok"))
        out = (
            REPO
            / "modal_exp"
            / "results"
            / f"{RUN_ID_OURS_CUDA}_ours_gpu_cuda.jsonl"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            for x in results:
                f.write(json.dumps(x, sort_keys=True) + "\n")
        print({"ok": ok, "failed": len(results) - ok, "local_jsonl": str(out)})
        return
    if cmd == "ours-gpu-fit":
        # Same 5 dense-fit rows as cuOpt — head-to-head on H100 box.
        jobs = []
        for r in rows:
            try:
                n = int(r["n"])
            except (TypeError, ValueError):
                continue
            if n > int(max_n):
                continue
            a, rh, tk = _gpu_knob(r)
            jobs.append(
                (r["instance"], r["path"], a, rh, r.get("bks", ""), tk)
            )
        print({"ours_gpu_fit_jobs": len(jobs), "max_n": max_n})
        results = list(run_ours_gpu_one.starmap(jobs))
        ok = sum(1 for x in results if x.get("ok"))
        out = (
            REPO
            / "modal_exp"
            / "results"
            / f"{RUN_ID_OURS_GPU}_ours_gpu_fit.jsonl"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            for x in results:
                f.write(json.dumps(x, sort_keys=True) + "\n")
        print({"ok": ok, "failed": len(results) - ok, "local_jsonl": str(out)})
        return

    def _args(topk_v: int = topk):
        return [
            (
                r["instance"],
                r["path"],
                float(r["alpha_bpmds"] or 0),
                rho,
                r.get("bks", ""),
                RUN_ID_OURS,
                "",
                int(topk_v),
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
                RUN_ID_OURS,
                "",
                int(topk),
                int(ls),
            )
        )
        return
    if cmd == "ours-large":
        results = list(run_ours_one.starmap(_args(topk)))
        ok = sum(1 for x in results if x.get("ok"))
        out = REPO / "modal_exp" / "results" / f"{RUN_ID_OURS}_ours.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            for x in results:
                f.write(json.dumps(x, sort_keys=True) + "\n")
        print(
            {
                "ok": ok,
                "failed": len(results) - ok,
                "topk": topk,
                "local_jsonl": str(out),
            }
        )
        return
    if cmd == "topk-sweep":
        # Probe K on R005 time-losers + current gap-losers (matched ρ=10k).
        time_losers = [
            "XML100000_2113_01",
            "XML100000_3123_01",
            "XML80000_2126_01",
            "XML100000_2163_01",
            "XML80000_3126_01",
            "XML200000_3126_01",
            "XML100000_2126_01",
            "XML100000_2121_01",
            "XML100000_2125_01",
            "XML100000_2133_01",
            "XML100000_2123_01",
            "Molise",
            "Flanders1",
            "Basilicata",
            "Umbria",
            "Piemonte",
        ]
        gap_losers = [
            "Sardegna",
            "Friuli-Venezia-Giulia",
            "Lazio",
            "Molise",
            "Basilicata",
            "Piemonte",
        ]
        # Plus a few BKS wins to guard gap regression.
        gap_guards = [
            "Ghent2",
            "Flanders2",
            "Brussels2",
            "Campania",
            "Liguria",
            "Puglia",
            "Lombardia",
            "Toscana",
            "Emilia-Romagna",
            "Abruzzo",
            "Veneto",
            "Marche",
        ]
        targets = list(dict.fromkeys(time_losers + gap_losers + gap_guards))
        ks = (1, 2, 3, 4)
        jobs = []
        for k in ks:
            for name in targets:
                r = by_name.get(name)
                if not r:
                    continue
                jobs.append(
                    (
                        r["instance"],
                        r["path"],
                        float(r["alpha_bpmds"] or 0),
                        rho,
                        r.get("bks", ""),
                        "R006sweep",
                        f"topk{k}",
                        int(k),
                    )
                )
        print({"topk_sweep_jobs": len(jobs), "targets": len(targets), "ks": list(ks)})
        results = list(run_ours_one.starmap(jobs))
        out = REPO / "modal_exp" / "results" / "R006_topk_sweep.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            for x in results:
                f.write(json.dumps(x, sort_keys=True) + "\n")

        summary = []
        for k in ks:
            subset = [x for x in results if x.get("ok") and int(x.get("topk", -1)) == k]
            time_wins = 0
            time_n = 0
            gap_wins = 0
            gap_n = 0
            for x in subset:
                row = by_name.get(x["instance"]) or {}
                if row.get("bpmds_time_s") and x.get("ours_time_s") is not None:
                    time_n += 1
                    if float(x["ours_time_s"]) <= float(row["bpmds_time_s"]) + 1e-9:
                        time_wins += 1
                if (
                    row.get("bks")
                    and row.get("bpmds_gap_pct")
                    and x.get("ours_gap_pct") is not None
                ):
                    gap_n += 1
                    if float(x["ours_gap_pct"]) <= float(row["bpmds_gap_pct"]) + 1e-9:
                        gap_wins += 1
            summary.append(
                {
                    "topk": k,
                    "ok": len(subset),
                    "time_wins": f"{time_wins}/{time_n}",
                    "gap_wins": f"{gap_wins}/{gap_n}",
                }
            )
        summary_path = REPO / "modal_exp" / "results" / "R006_topk_sweep_summary.json"
        summary_path.write_text(json.dumps({"summary": summary, "targets": targets}, indent=2))
        print({"local_jsonl": str(out), "summary": summary, "summary_json": str(summary_path)})
        return
    if cmd in (
        "phase-b",
        "phase-b-batch1",
        "phase-b-batch2",
        "phase-b2",
        "phase-b3",
        "phase-b4",
        "phase-b5",
    ):
        if cmd == "phase-b":
            jobs = _phase_b_jobs(by_name)
            targets = list(PHASE_B_LOSERS)
            out_name = f"{RUN_ID_PHASE_B}_phaseb.jsonl"
            best_name = f"{RUN_ID_PHASE_B}_phaseb_best.json"
        elif cmd == "phase-b-batch1":
            jobs, targets = _phase_b_batch_jobs(by_name, 1)
            out_name = f"{RUN_ID_PHASE_B}_phaseb_batch1.jsonl"
            best_name = f"{RUN_ID_PHASE_B}_phaseb_batch1_best.json"
        elif cmd == "phase-b-batch2":
            jobs, targets = _phase_b_batch_jobs(by_name, 2)
            out_name = f"{RUN_ID_PHASE_B}_phaseb_batch2.jsonl"
            best_name = f"{RUN_ID_PHASE_B}_phaseb_batch2_best.json"
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
        elif cmd == "phase-b4":
            jobs = _phase_b4_jobs(by_name)
            targets = ["Friuli-Venezia-Giulia"]
            out_name = f"{RUN_ID_PHASE_B}_phaseb4.jsonl"
            best_name = f"{RUN_ID_PHASE_B}_phaseb4_best.json"
        else:
            jobs = _phase_b5_jobs(by_name)
            targets = ["Friuli-Venezia-Giulia"]
            out_name = f"{RUN_ID_PHASE_B}_phaseb5.jsonl"
            best_name = f"{RUN_ID_PHASE_B}_phaseb5_best.json"
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
    if cmd in ("r013a", "r013a-resume"):
        # Gap-first Tier A: never-Phase-B I instances (resume skips done keys).
        prior = REPO / "modal_exp" / "results" / f"{RUN_ID_GAP}_ours.jsonl"
        skip = _r013_done_keys(prior) if cmd == "r013a-resume" else set()
        jobs = _r013a_jobs(by_name, skip=skip)
        targets = list(TIER_A)
        print(
            {
                "cmd": cmd,
                "r013a_jobs": len(jobs),
                "skipped": len(skip),
                "instances": targets,
                "topk": GAP_TOPK,
                "ls": GAP_LS,
            }
        )
        if not jobs:
            print({"ok": True, "msg": "nothing left to run"})
            return
        results = list(run_ours_one.starmap(jobs))
        ok = sum(1 for x in results if x.get("ok"))
        # Merge with prior ok rows for a full-grid view.
        merged: list[dict] = []
        if prior.is_file():
            for line in prior.read_text().splitlines():
                if line.strip():
                    merged.append(json.loads(line))
        merged.extend(results)
        out = REPO / "modal_exp" / "results" / f"{RUN_ID_GAP}_tierA.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            for x in merged:
                f.write(json.dumps(x, sort_keys=True) + "\n")
        # Also refresh local mine of volume appends from this wave.
        wave = REPO / "modal_exp" / "results" / f"{RUN_ID_GAP}_tierA_wave.jsonl"
        with wave.open("w") as f:
            for x in results:
                f.write(json.dumps(x, sort_keys=True) + "\n")

        best: dict[str, dict] = {}
        for x in merged:
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
            ours_gap = float(b["ours_gap_pct"]) if b.get("ours_gap_pct") is not None else None
            sheet_gap = float(row["ours_gap_pct"]) if row.get("ours_gap_pct") else None
            f2x_gap = float(row["filo2x_gap_pct"]) if row.get("filo2x_gap_pct") else None
            improved = (
                ours_gap is not None
                and sheet_gap is not None
                and ours_gap < sheet_gap - 1e-9
            )
            beat_f2x = (
                ours_gap is not None
                and f2x_gap is not None
                and ours_gap <= f2x_gap + 1e-9
            )
            summary.append(
                {
                    "instance": inst,
                    "alpha": b["alpha"],
                    "rho": b["rho"],
                    "topk": b.get("topk"),
                    "ls": b.get("ls"),
                    "ours_gap_pct": ours_gap,
                    "sheet_gap_pct": sheet_gap,
                    "filo2x_gap_pct": f2x_gap,
                    "d_vs_sheet": None
                    if ours_gap is None or sheet_gap is None
                    else ours_gap - sheet_gap,
                    "d_vs_filo2x": None
                    if ours_gap is None or f2x_gap is None
                    else ours_gap - f2x_gap,
                    "ours_time_s": b.get("ours_time_s"),
                    "improved_vs_sheet": improved,
                    "gap_le_filo2x": beat_f2x,
                }
            )
        best_path = REPO / "modal_exp" / "results" / f"{RUN_ID_GAP}_tierA_best.json"
        best_path.write_text(json.dumps({"best": best, "summary": summary}, indent=2))
        print(
            {
                "ok": ok,
                "failed": len(results) - ok,
                "wave_jsonl": str(wave),
                "local_jsonl": str(out),
                "best_json": str(best_path),
                "improved": sum(1 for s in summary if s.get("improved_vs_sheet")),
                "gap_le_filo2x": sum(1 for s in summary if s.get("gap_le_filo2x")),
                "summary": summary,
            }
        )
        return
    if cmd in ("r015", "r015-resume"):
        # Quality-first soft-time search on near/mid FILO2x losers.
        targets = list(TIER_QUAL)
        all_jobs = _r013b_jobs(
            by_name,
            tuple(targets),
            topks=(4, 6, 8, 12),
            lss=(1, 2),
            rho_mults=(1.0, 2.0, 3.0, 4.0),
            run_id=RUN_ID_QUAL,
        )
        out = REPO / "modal_exp" / "results" / f"{RUN_ID_QUAL}_qual.jsonl"
        prior_path = REPO / "modal_exp" / "results" / f"{RUN_ID_QUAL}_ours.jsonl"
        prior: list[dict] = []
        for p in (out, prior_path):
            if not p.is_file():
                continue
            for line in p.read_text().splitlines():
                if line.strip():
                    prior.append(json.loads(line))
        # Dedup prior by (inst, alpha, rho, topk, ls), prefer ok.
        prior_by_key: dict[tuple, dict] = {}
        for x in prior:
            if not x.get("ok"):
                continue
            try:
                key = (
                    x["instance"],
                    float(x["alpha"]),
                    int(x["rho"]),
                    int(x.get("topk") or 0),
                    int(x.get("ls") or 0),
                )
            except (TypeError, ValueError, KeyError):
                continue
            prior_by_key[key] = x
        jobs = []
        for j in all_jobs:
            key = (j[0], float(j[2]), int(j[3]), int(j[7]), int(j[8]))
            if key not in prior_by_key:
                jobs.append(j)
        print(
            {
                "cmd": cmd,
                "tier": "qual",
                "r015_jobs_total": len(all_jobs),
                "r015_jobs_todo": len(jobs),
                "r015_prior_ok": len(prior_by_key),
                "instances": targets,
                "topks": [4, 6, 8, 12],
                "ls": [1, 2],
                "rho_mults": [1.0, 2.0, 3.0, 4.0],
                "soft_time_mult": QUALITY_TIME_MULT,
            }
        )
        new_results: list[dict] = []
        if jobs:
            new_results = list(run_ours_one.starmap(jobs))
        for x in new_results:
            if not x.get("ok"):
                continue
            try:
                key = (
                    x["instance"],
                    float(x["alpha"]),
                    int(x["rho"]),
                    int(x.get("topk") or 0),
                    int(x.get("ls") or 0),
                )
            except (TypeError, ValueError, KeyError):
                continue
            prior_by_key[key] = x
        results = list(prior_by_key.values())
        ok = sum(1 for x in results if x.get("ok"))
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            for x in results:
                f.write(json.dumps(x, sort_keys=True) + "\n")
        best_soft, best_dual, summary = _r015_summarize(
            results, by_name, targets, time_mult=QUALITY_TIME_MULT
        )
        best_path = REPO / "modal_exp" / "results" / f"{RUN_ID_QUAL}_qual_best.json"
        best_path.write_text(
            json.dumps(
                {
                    "soft_time_mult": QUALITY_TIME_MULT,
                    "best_soft": best_soft,
                    "best_dual": best_dual,
                    "summary": summary,
                },
                indent=2,
            )
        )
        # Sheet apply: only hard dual-wins that also improve vs sheet.
        apply_rows = []
        for inst, b in best_dual.items():
            row = by_name.get(inst) or {}
            try:
                if float(b["ours_gap_pct"]) < float(row["ours_gap_pct"]) - 1e-9:
                    apply_rows.append(
                        {
                            "instance": inst,
                            "cost": b["ours_cost"],
                            "time_s": b["ours_time_s"],
                            "gap_pct": b["ours_gap_pct"],
                            "alpha": b["alpha"],
                            "rho": b["rho"],
                            "topk": b.get("topk"),
                            "ls": b.get("ls"),
                        }
                    )
            except (TypeError, ValueError, KeyError):
                continue
        apply_path = (
            REPO / "modal_exp" / "results" / f"{RUN_ID_QUAL}_qual_apply.jsonl"
        )
        with apply_path.open("w") as f:
            for r in apply_rows:
                f.write(json.dumps(r) + "\n")
        print(
            {
                "ok": ok,
                "failed": len(new_results) - sum(1 for x in new_results if x.get("ok")),
                "run_id": RUN_ID_QUAL,
                "local_jsonl": str(out),
                "best_json": str(best_path),
                "apply_jsonl": str(apply_path),
                "dual_wins_vs_sheet": len(apply_rows),
                "clears_filo2x_soft": sum(
                    1 for s in summary if s.get("clears_filo2x_soft")
                ),
                "sweet_spots": sum(1 for s in summary if s.get("sweet_spot")),
                "summary": summary,
            }
        )
        return
    if cmd in ("r013b", "r013b-c", "r013c", "r014"):
        if cmd == "r014":
            # Stronger/cheaper LS — closest FILO2x losers; denser ρ/topk, ls=1 first.
            targets = list(TIER_D)
            tier = "D"
            run_id = RUN_ID_LSX
            jobs = _r013b_jobs(
                by_name,
                tuple(targets),
                topks=(2, 4, 6),
                lss=(1, 2),
                rho_mults=(1.0, 2.0, 3.0),
                run_id=RUN_ID_LSX,
            )
            out_prefix = RUN_ID_LSX
            print(
                {
                    "cmd": cmd,
                    "tier": tier,
                    "r014_jobs": len(jobs),
                    "instances": targets,
                    "topks": [2, 4, 6],
                    "ls": [1, 2],
                    "rho_mults": [1.0, 2.0, 3.0],
                    "ops": "exchange+cross+cheap-span3opt",
                }
            )
        elif cmd == "r013c":
            targets = list(TIER_SLACK)
            tier = "slack"
            run_id = RUN_ID_GAP_C
            jobs = _r013b_jobs(
                by_name,
                tuple(targets),
                topks=(4, 6, 8),
                lss=(1, 2),
                rho_mults=(1.0, 2.0, 3.0),
                run_id=RUN_ID_GAP_C,
            )
            out_prefix = RUN_ID_GAP_C
            print(
                {
                    "cmd": cmd,
                    "tier": tier,
                    "r013c_jobs": len(jobs),
                    "instances": targets,
                    "topks": [4, 6, 8],
                    "ls": [1, 2],
                    "rho_mults": [1.0, 2.0, 3.0],
                }
            )
        else:
            targets = list(TIER_C if cmd == "r013b-c" else TIER_D)
            tier = "C" if cmd == "r013b-c" else "D"
            run_id = RUN_ID_GAP_B
            jobs = _r013b_jobs(by_name, tuple(targets))
            out_prefix = RUN_ID_GAP_B
            print(
                {
                    "cmd": cmd,
                    "tier": tier,
                    "r013b_jobs": len(jobs),
                    "instances": targets,
                    "topks": [2, 3, 4],
                    "ls": [1, 2],
                    "rho_mults": [1.0, 1.5, 2.0],
                }
            )
        results = list(run_ours_one.starmap(jobs))
        ok = sum(1 for x in results if x.get("ok"))
        out = REPO / "modal_exp" / "results" / f"{out_prefix}_tier{tier}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            for x in results:
                f.write(json.dumps(x, sort_keys=True) + "\n")
        best, summary = _r013b_summarize(results, by_name, targets)
        best_path = (
            REPO / "modal_exp" / "results" / f"{out_prefix}_tier{tier}_best.json"
        )
        best_path.write_text(json.dumps({"best": best, "summary": summary}, indent=2))
        apply_rows = []
        for inst, b in best.items():
            apply_rows.append(
                {
                    "instance": inst,
                    "cost": b["ours_cost"],
                    "time_s": b["ours_time_s"],
                    "gap_pct": b["ours_gap_pct"],
                    "alpha": b["alpha"],
                    "rho": b["rho"],
                    "topk": b.get("topk"),
                    "ls": b.get("ls"),
                }
            )
        apply_path = (
            REPO / "modal_exp" / "results" / f"{out_prefix}_tier{tier}_apply.jsonl"
        )
        with apply_path.open("w") as f:
            for r in apply_rows:
                f.write(json.dumps(r) + "\n")
        print(
            {
                "ok": ok,
                "failed": len(results) - ok,
                "run_id": run_id,
                "local_jsonl": str(out),
                "best_json": str(best_path),
                "apply_jsonl": str(apply_path),
                "dual_wins": len(best),
                "gap_le_filo2x": sum(1 for s in summary if s.get("gap_le_filo2x")),
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
        "cmd: list-large | gpu-ready | cuopt-one | cuopt-fit | "
        "ours-gpu-one | ours-gpu-fit | ours-gpu-bks | "
        "ours-cuda-one | ours-cuda-bks | ours-ls-bks | "
        "ours-one | ours-large | topk-sweep | "
        "phase-b | phase-b-batch1 | phase-b-batch2 | phase-b2 | phase-b3 | "
        "phase-b4 | phase-b5 | r013a | r013a-resume | r013b | r013b-c | r013c | r014 | r015 | r015-resume | "
        "harvest | bpmds-one | bpmds-large | harvest-bpmds"
    )
