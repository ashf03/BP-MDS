"""Solve EUC_2D CVRP with NVIDIA cuOpt (GPU). Timed region = routing.Solve only."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from vrp_io import VrpInstance, parse_vrp


def estimate_fleet(inst: VrpInstance, slack: float = 1.25) -> int:
    """Enough identical capacity-Q vehicles to cover demand (+slack)."""
    base = inst.min_vehicles()
    return max(base, int(np.ceil(base * slack)))


def solve_cvrp_cuopt(
    inst: VrpInstance,
    *,
    time_limit_s: float = 30.0,
    fleet_slack: float = 1.25,
    fleet_override: int | None = None,
) -> dict[str, Any]:
    """
    Build dense Euclidean cost matrix and call cuopt.routing.Solve.

    Returns cost/gap/time fields plus diagnostics. Does not include matrix
    build time in cuopt_time_s.
    """
    import cudf
    from cuopt import routing

    n = inst.n
    n_orders = inst.n_customers
    n_fleet = int(fleet_override) if fleet_override else estimate_fleet(inst, fleet_slack)

    t_mat0 = time.perf_counter()
    cost_np = inst.euclidean_cost_matrix(dtype=np.float32)
    t_mat = time.perf_counter() - t_mat0

    cost_mat = cudf.DataFrame(cost_np)
    order_locations = cudf.Series(np.arange(1, n, dtype=np.int32))
    demand = cudf.Series(inst.demand[1:].astype(np.int32))
    capacity = cudf.Series(
        np.full(n_fleet, int(inst.capacity), dtype=np.int32)
    )

    dm = routing.DataModel(n, n_fleet, n_orders)
    dm.add_cost_matrix(cost_mat)
    dm.set_order_locations(order_locations)
    dm.add_capacity_dimension("demand", demand, capacity)

    ss = routing.SolverSettings()
    if time_limit_s and time_limit_s > 0:
        ss.set_time_limit(float(time_limit_s))

    t0 = time.perf_counter()
    sol = routing.Solve(dm, ss)
    solve_s = time.perf_counter() - t0

    status = int(sol.get_status()) if hasattr(sol, "get_status") else -1
    objective = None
    try:
        objective = float(sol.get_total_objective())
    except Exception:
        objective = None

    n_routes = None
    try:
        route_df = sol.get_route()
        if route_df is not None and hasattr(route_df, "truck_id"):
            n_routes = int(route_df["truck_id"].nunique())
    except Exception:
        n_routes = None

    # status 0 = Success; 1 = often time-limit with a usable incumbent.
    # Scoreboard records any finite objective; status is kept for diagnostics.
    usable = objective is not None and np.isfinite(objective)
    return {
        "ok": usable,
        "status": status,
        "cuopt_cost": objective,
        "cuopt_time_s": solve_s,
        "matrix_build_s": t_mat,
        "n": n,
        "n_fleet": n_fleet,
        "n_routes": n_routes,
        "time_limit_s": time_limit_s,
        "solver": "cuopt",
    }


def solve_vrp_path(
    path: str,
    *,
    time_limit_s: float = 30.0,
    fleet_slack: float = 1.25,
) -> dict[str, Any]:
    inst = parse_vrp(path)
    out = solve_cvrp_cuopt(
        inst, time_limit_s=time_limit_s, fleet_slack=fleet_slack
    )
    out["instance"] = inst.name
    out["path"] = path
    out["capacity"] = inst.capacity
    return out
