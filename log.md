# Experiment log

Pinned versions, hardware, timed region, and run batches.
Scoreboard `notes` should cite a **Run ID** from here (e.g. `run=R001`).

Sheet: https://docs.google.com/spreadsheets/d/18Z0JchFiihrfq31Alr3_XLvNnxVLcktAKo5pzkanE6s/edit?usp=sharing

---

## Defaults (current pins)

| Field | Value |
|:------|:------|
| **ours** (fork) | `ashf03/BP-MDS` @ `8f80169` · branch `mod/cleanup-and-speedups` |
| **BP-MDS** (upstream) | `CodeCraftsmanSandeep/BP-MDS` @ `5b9e24f` · `master` |
| **cuOpt** | _not pinned yet_ |
| **timed region (CPU)** | `Solver::solve` wall time (seconds) |
| **timed region (cuOpt)** | _TBD — cuOpt solve call only; exclude I/O setup unless noted_ |
| **gap** | `100 * (cost - BKS) / BKS` |
| **local hardware (dev)** | Apple M5 · 10 logical CPUs · 16 GB · arm64 (not the fair cluster box) |

**Notes string template** (paste into scoreboard `notes`):

```text
run=R00X bpmds@<sha> ours@<sha> cuopt@<tag|na> hw=<spec> timed=<region>
```

---

## Run log

Newest first. Add a block per batch.

### R000 — pins established (no new solves)

- **date:** 2026-11-11
- **what:** created this log; recorded current SHAs + local Mac hardware
- **bpmds:** `5b9e24f`
- **ours:** `8f80169`
- **cuopt:** —
- **hardware:** Apple M5 / 10 CPU / 16 GB (dev only)
- **timed:** `Solver::solve`
- **suite:** —
- **scoreboard:** no row updates
- **notes paste:** `run=R000 bpmds@5b9e24f ours@8f80169 cuopt@na hw=M5-10c-16G timed=Solver::solve`

<!--
### R00X — template

- **date:** YYYY-MM-DD
- **what:** short description
- **bpmds:** `<sha>`
- **ours:** `<sha>`
- **cuopt:** `<version/tag>` or —
- **hardware:** e.g. `48xCPU cluster` / `1xA100-80GB`
- **timed:** `Solver::solve` / `cuopt-solve`
- **suite:** CVRPLIB | FILO2 | Synthetic | subset…
- **alpha / rho / other:** …
- **scoreboard:** which columns filled (`ours_*`, `bpmds_time_s`, …)
- **notes paste:** `run=R00X bpmds@… ours@… cuopt@… hw=… timed=…`
-->
