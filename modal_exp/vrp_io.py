"""Parse CVRPLIB-style EUC_2D .vrp files for cuOpt / GPU runners."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class VrpInstance:
    name: str
    capacity: int
    coords: np.ndarray  # (n, 2) float64, row 0 = depot
    demand: np.ndarray  # (n,) int32, demand[0] == 0
    path: str = ""

    @property
    def n(self) -> int:
        return int(self.coords.shape[0])

    @property
    def n_customers(self) -> int:
        return self.n - 1

    def min_vehicles(self) -> int:
        total = int(self.demand[1:].sum())
        return max(1, (total + self.capacity - 1) // self.capacity)

    def euclidean_cost_matrix(self, dtype=np.float32, chunk: int = 512) -> np.ndarray:
        """Dense pairwise Euclidean distances (float32 by default).

        Chunked to avoid an (n,n,2) temporary at large n.
        """
        xy = self.coords.astype(np.float64, copy=False)
        n = xy.shape[0]
        mat = np.empty((n, n), dtype=dtype)
        for i0 in range(0, n, chunk):
            i1 = min(n, i0 + chunk)
            d = xy[i0:i1, None, :] - xy[None, :, :]
            mat[i0:i1] = np.sqrt((d * d).sum(axis=-1)).astype(dtype, copy=False)
        np.fill_diagonal(mat, 0.0)
        return mat


def parse_vrp(path: Path | str) -> VrpInstance:
    path = Path(path)
    text = path.read_text(errors="replace")
    name = path.stem
    capacity = None
    coords: dict[int, tuple[float, float]] = {}
    demand: dict[int, int] = {}
    section = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("NAME"):
            name = line.split(":", 1)[-1].strip() or name
            continue
        if upper.startswith("CAPACITY"):
            capacity = int(line.split(":")[-1].strip().split()[0])
            continue
        if upper.startswith("NODE_COORD_SECTION"):
            section = "coord"
            continue
        if upper.startswith("DEMAND_SECTION"):
            section = "demand"
            continue
        if upper.startswith("DEPOT_SECTION") or upper == "EOF":
            section = None
            continue
        if section == "coord":
            parts = line.split()
            if len(parts) >= 3 and parts[0].lstrip("-").isdigit():
                idx = int(parts[0])
                coords[idx] = (float(parts[1]), float(parts[2]))
        elif section == "demand":
            parts = line.split()
            if len(parts) >= 2 and parts[0].lstrip("-").isdigit():
                idx = int(parts[0])
                demand[idx] = int(parts[1])

    if capacity is None:
        raise ValueError(f"no CAPACITY in {path}")
    if not coords:
        raise ValueError(f"no NODE_COORD_SECTION in {path}")

    # CVRPLIB ids are 1-based; pack densely with depot at index 0.
    ids = sorted(coords.keys())
    if ids[0] != 1:
        # still ok if contiguous
        pass
    n = len(ids)
    id_to_row = {i: k for k, i in enumerate(ids)}
    xy = np.zeros((n, 2), dtype=np.float64)
    dem = np.zeros(n, dtype=np.int32)
    for i, node_id in enumerate(ids):
        xy[i] = coords[node_id]
        dem[i] = int(demand.get(node_id, 0))

    # Ensure depot (first id, usually 1) has zero demand.
    dem[0] = 0
    return VrpInstance(
        name=name, capacity=capacity, coords=xy, demand=dem, path=str(path)
    )
