#!/usr/bin/env python3
"""Generate report figures as high-DPI PNGs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Arc, Wedge
import numpy as np

OUT = Path(__file__).resolve().parent / "figs"
OUT.mkdir(exist_ok=True)

# palette — clay vs teal; figure/axes stay transparent for Overleaf paper
INK = "#1a1a1a"
MUTED = "#5c5c5c"
LINE = "#d0d0d0"
BASE = "#c45c26"
MOD = "#0f7a5a"
BASE_SOFT = "#f0d5c4"
MOD_SOFT = "#d4ebe3"

plt.rcParams.update(
    {
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "axes.edgecolor": LINE,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.18,
        "savefig.facecolor": "none",
        "savefig.transparent": True,
    }
)


def save(fig: plt.Figure, name: str) -> None:
    path = OUT / name
    fig.patch.set_alpha(0.0)
    for ax in fig.axes:
        ax.patch.set_alpha(0.0)
    fig.savefig(path, dpi=220, transparent=True, facecolor="none", edgecolor="none")
    plt.close(fig)
    print(f"wrote {path}")


def fig_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 2.55))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis("off")

    stages = [
        (0.35, "Load\n.vrp", False),
        (2.25, "Partition\nα sectors", True),
        (4.15, "MST\nper sector", True),
        (6.05, "ρ × DFS\n+ early abort", True),
        (7.95, "2-opt\nΔ edges", True),
        (9.85, "Merge\n& .sol", False),
    ]

    for x, label, hot in stages:
        face = (*plt.matplotlib.colors.to_rgba(MOD_SOFT)[:3], 0.55) if hot else "none"
        edge = MOD if hot else "#b8b8b8"
        box = FancyBboxPatch(
            (x, 1.05),
            1.55,
            1.25,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            linewidth=1.6,
            edgecolor=edge,
            facecolor=face,
        )
        ax.add_patch(box)
        ax.text(
            x + 0.775,
            1.68,
            label,
            ha="center",
            va="center",
            fontsize=10,
            color=INK,
            linespacing=1.25,
        )

    for i in range(len(stages) - 1):
        x0 = stages[i][0] + 1.55
        x1 = stages[i + 1][0]
        ax.annotate(
            "",
            xy=(x1 - 0.02, 1.68),
            xytext=(x0 + 0.02, 1.68),
            arrowprops=dict(
                arrowstyle="-|>",
                color=MUTED,
                lw=1.5,
                mutation_scale=12,
            ),
        )

    ax.text(
        6.0,
        0.35,
        "teal = rewritten hot path · same combinatorial neighborhood",
        ha="center",
        va="center",
        fontsize=9,
        color=MUTED,
    )
    save(fig, "pipeline.png")


def _scatter_instance(rng: np.random.Generator, n: int = 420):
    # clustered customers around depot — looks like a real CVRP cut
    centers = rng.uniform(-0.85, 0.85, size=(7, 2))
    pts = []
    for c in centers:
        k = n // len(centers)
        cloud = c + rng.normal(0, 0.11, size=(k, 2))
        pts.append(cloud)
    pts = np.vstack(pts)
    pts = pts[(np.abs(pts) < 1.05).all(axis=1)]
    return pts


def fig_partition() -> None:
    rng = np.random.default_rng(7)
    pts = _scatter_instance(rng)
    angles = np.arctan2(pts[:, 1], pts[:, 0])
    n_sec = 8
    edges = np.linspace(-np.pi, np.pi, n_sec + 1)
    colors = plt.cm.Spectral(np.linspace(0.12, 0.88, n_sec))

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8))
    for ax, title, mode in zip(
        axes,
        ["Before · wedge tests", "After · atan2 → bucket"],
        ["before", "after"],
    ):
        ax.set_aspect("equal")
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.25, 1.25)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(LINE)
        ax.set_title(title, pad=10)

        # sector wedges
        for i in range(n_sec):
            w = Wedge(
                (0, 0),
                1.18,
                np.degrees(edges[i]),
                np.degrees(edges[i + 1]),
                facecolor=(*colors[i][:3], 0.18 if mode == "after" else 0.08),
                edgecolor="#9a9a9a",
                linewidth=0.7,
            )
            ax.add_patch(w)

        if mode == "before":
            ax.scatter(pts[:, 0], pts[:, 1], s=8, c="#3a3a3a", alpha=0.55, linewidths=0, zorder=3)
            # draw a few comparison rays to every wedge edge (busy = slow)
            for a in edges[:-1]:
                ax.plot([0, 1.05 * np.cos(a)], [0, 1.05 * np.sin(a)], color=BASE, lw=0.9, alpha=0.55)
            ax.text(
                0,
                -1.42,
                "test each customer against every sector",
                ha="center",
                fontsize=9,
                color=MUTED,
            )
        else:
            # color by bucket
            buckets = np.digitize(angles, edges[1:-1], right=False)
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                s=10,
                c=colors[buckets],
                alpha=0.85,
                linewidths=0,
                zorder=3,
            )
            # one atan2 ray on a sample point
            i = int(np.argmax(pts[:, 0] ** 2 + pts[:, 1] ** 2))
            ax.annotate(
                "",
                xy=(pts[i, 0], pts[i, 1]),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=MOD, lw=2.0, mutation_scale=14),
                zorder=4,
            )
            ax.text(
                pts[i, 0] * 0.55 + 0.08,
                pts[i, 1] * 0.55 + 0.08,
                "atan2",
                color=MOD,
                fontsize=10,
                fontweight="bold",
            )
            ax.text(
                0,
                -1.42,
                "one angle → sector index",
                ha="center",
                fontsize=9,
                color=MUTED,
            )

        ax.add_patch(Circle((0, 0), 0.045, color=MOD if mode == "after" else BASE, zorder=5))
        ax.text(0.06, 0.08, "depot", fontsize=8, color=MUTED)

    fig.tight_layout()
    save(fig, "partition.png")


def fig_algo() -> None:
    """MST tree on a sector + 2-opt delta geometry."""
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.5))

    # --- left: MST in one angular sector ---
    ax = axes[0]
    ax.set_aspect("equal")
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.15, 1.15)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Sector MST (dense Prim on d²)", pad=10)
    for spine in ax.spines.values():
        spine.set_color(LINE)

    # points in a 45° wedge
    n = 28
    r = rng.uniform(0.2, 1.0, n)
    th = rng.uniform(0.15, 1.05, n)
    pts = np.column_stack([r * np.cos(th), r * np.sin(th)])
    depot = np.array([[0.0, 0.0]])
    all_pts = np.vstack([depot, pts])

    # Prim on squared distance
    N = len(all_pts)
    in_tree = np.zeros(N, dtype=bool)
    in_tree[0] = True
    parent = np.full(N, -1)
    best = np.full(N, np.inf)
    best[0] = 0.0
    for j in range(1, N):
        d = np.sum((all_pts[j] - all_pts[0]) ** 2)
        best[j] = d
        parent[j] = 0
    for _ in range(N - 1):
        cand = np.where(~in_tree)[0]
        v = cand[np.argmin(best[cand])]
        in_tree[v] = True
        for w in cand:
            if w == v:
                continue
            dw = np.sum((all_pts[w] - all_pts[v]) ** 2)
            if dw < best[w]:
                best[w] = dw
                parent[w] = v

    for j in range(1, N):
        a, b = all_pts[j], all_pts[parent[j]]
        ax.plot([a[0], b[0]], [a[1], b[1]], color=MOD, lw=1.4, alpha=0.85, zorder=2)
    ax.scatter(pts[:, 0], pts[:, 1], s=28, c=INK, zorder=3, linewidths=0)
    ax.scatter([0], [0], s=90, c=MOD, zorder=4, edgecolors="white", linewidths=1.2)
    ax.text(0.02, -0.08, "depot", fontsize=8, color=MUTED)
    ax.text(
        0.5,
        -0.12,
        "no heap · skip √ · same tree",
        ha="center",
        fontsize=9,
        color=MUTED,
        transform=ax.get_xaxis_transform(),
        clip_on=False,
    )

    # --- right: 2-opt delta ---
    ax = axes[1]
    ax.set_aspect("equal")
    ax.set_xlim(-0.2, 3.4)
    ax.set_ylim(-0.3, 2.4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("2-opt scored with O(1) edge Δ", pad=10)
    for spine in ax.spines.values():
        spine.set_color(LINE)

    A = np.array([0.35, 0.55])
    B = np.array([1.05, 2.05])
    C = np.array([2.35, 1.85])
    D = np.array([3.05, 0.45])
    # route spine
    ax.plot([A[0], B[0], C[0], D[0]], [A[1], B[1], C[1], D[1]], color="#c8c8c8", lw=2.2, zorder=1)
    # removed edges
    ax.plot([A[0], B[0]], [A[1], B[1]], color=BASE, lw=2.4, ls="--", zorder=2, label="removed")
    ax.plot([C[0], D[0]], [C[1], D[1]], color=BASE, lw=2.4, ls="--", zorder=2)
    # added edges
    ax.plot([A[0], C[0]], [A[1], C[1]], color=MOD, lw=2.6, zorder=3, label="added")
    ax.plot([B[0], D[0]], [B[1], D[1]], color=MOD, lw=2.6, zorder=3)
    for p, name in zip([A, B, C, D], ["a−", "a", "b", "b+"]):
        ax.scatter([p[0]], [p[1]], s=55, c=INK, zorder=4, linewidths=0)
        ax.text(p[0], p[1] + 0.14, name, ha="center", fontsize=9, color=MUTED)

    ax.text(
        1.7,
        -0.12,
        "Δ = d(a−,b) + d(a,b+) − d(a−,a) − d(b,b+)",
        ha="center",
        fontsize=9.5,
        color=INK,
        family="DejaVu Sans Mono",
    )
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    fig.tight_layout()
    save(fig, "algo.png")


def fig_omp() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6), sharey=True)
    rng = np.random.default_rng(1)

    for ax, title, mode in zip(
        axes,
        ["Before · lock every improve", "After · thread-local best"],
        ["before", "after"],
    ):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6.2)
        ax.axis("off")
        ax.set_title(title, pad=8)

        # timeline spine
        ax.plot([0.6, 9.4], [1.2, 1.2], color=LINE, lw=1.2)
        ax.text(0.3, 1.2, "t", color=MUTED, fontsize=9, va="center")

        for thr, y in enumerate([4.6, 3.5, 2.4]):
            ax.text(0.15, y + 0.35, f"T{thr}", fontsize=9, color=MUTED, ha="left")
            # trial blocks
            starts = rng.uniform(0.8, 1.4) + thr * 0.05
            for k in range(5):
                x = starts + k * 1.55
                ax.add_patch(
                    FancyBboxPatch(
                        (x, y),
                        1.15,
                        0.7,
                        boxstyle="round,pad=0.02,rounding_size=0.08",
                        facecolor="none",
                        edgecolor="#b0b0b0",
                        linewidth=1.1,
                    )
                )
                # critical hits
                if mode == "before" and k in (1, 3) or (mode == "before" and thr == 1 and k == 2):
                    ax.add_patch(
                        FancyBboxPatch(
                            (x + 0.25, 0.85),
                            0.65,
                            0.35,
                            boxstyle="round,pad=0.01,rounding_size=0.05",
                            facecolor=BASE_SOFT,
                            edgecolor=BASE,
                            linewidth=1.1,
                        )
                    )
                    ax.plot([x + 0.55, x + 0.55], [y, 1.2], color=BASE, lw=0.9, alpha=0.7)

        if mode == "after":
            ax.add_patch(
                FancyBboxPatch(
                    (2.2, 0.75),
                    5.6,
                    0.55,
                    boxstyle="round,pad=0.02,rounding_size=0.08",
                    facecolor=MOD_SOFT,
                    edgecolor=MOD,
                    linewidth=1.3,
                )
            )
            ax.text(5.0, 1.02, "one merge of thread-local bests", ha="center", va="center", fontsize=9, color=INK)
        else:
            ax.text(5.0, 0.45, "critical section on every improving trial", ha="center", fontsize=9, color=MUTED)

    fig.tight_layout()
    save(fig, "omp.png")


def fig_timing() -> None:
    sizes = ["10⁶ customers", "10⁷ customers"]
    base = np.array([57.57, 421.98])
    mod = np.array([13.49, 229.60])
    x = np.arange(len(sizes))
    w = 0.34

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    b1 = ax.bar(x - w / 2, base, w, color=BASE, label="baseline", zorder=3, edgecolor="none")
    b2 = ax.bar(x + w / 2, mod, w, color=MOD, label="modified", zorder=3, edgecolor="none")

    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                h + 8,
                f"{h:.1f}s",
                ha="center",
                va="bottom",
                fontsize=10,
                color=INK,
            )

    # percent labels
    for i, (bv, mv) in enumerate(zip(base, mod)):
        pct = 100 * (mv - bv) / bv
        ax.text(
            i,
            max(bv, mv) * 0.55,
            f"{pct:.1f}%",
            ha="center",
            fontsize=12,
            fontweight="bold",
            color=MOD,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="none", edgecolor=MOD, linewidth=1.2),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_ylabel("Solver::solve wall time (s)")
    ax.set_title("Matched Main runs · α=2 · ρ=10⁴ · 48 threads", pad=12)
    ax.legend(frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(LINE)
    ax.spines["bottom"].set_color(LINE)
    ax.yaxis.grid(True, color=LINE, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 500)
    fig.tight_layout()
    save(fig, "timing.png")


def fig_stress() -> None:
    fig, ax = plt.subplots(figsize=(9.2, 2.9))
    labels = ["modified (finished)", "baseline (aborted)"]
    hours = [4.53, 26.0]
    colors = [MOD, BASE]
    y = np.arange(len(labels))

    ax.barh(y, hours, color=colors, height=0.55, edgecolor="none", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("wall time (hours)")
    ax.set_title("≈4.7×10⁷ customers · α=5 stress", pad=10)
    ax.set_xlim(0, 30)
    ax.axvline(4.53, color=MOD, ls=":", lw=1.0, alpha=0.7)
    ax.text(4.53, 1.45, "4.53 h", color=MOD, fontsize=9, ha="center")
    ax.text(26.0, -0.08, ">26 h · no .sol written", color=BASE, fontsize=9, ha="right", va="top")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(LINE)
    ax.spines["bottom"].set_color(LINE)
    ax.xaxis.grid(True, color=LINE, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "stress.png")


def main() -> None:
    fig_pipeline()
    fig_partition()
    fig_algo()
    fig_omp()
    fig_timing()
    fig_stress()


if __name__ == "__main__":
    main()
