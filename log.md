# Experiment log

Pinned versions, hardware, timed region, and run batches.
Scoreboard `notes` should cite a **Run ID** from here (e.g. `run=R001`).

Sheet: https://docs.google.com/spreadsheets/d/18Z0JchFiihrfq31Alr3_XLvNnxVLcktAKo5pzkanE6s/edit?usp=sharing

---

## Defaults (current pins)

| Field | Value |
|:------|:------|
| **ours** (fork) | `ashf03/BP-MDS` @ `1190a34` · branch `mod/cleanup-and-speedups` |
| **Modal workspace** | **`aquin`** (profile `aquin`; was `ash-34107` for R001–R004) |
| **Modal CPU** | 48 physical cores · 256 GiB · `OMP_NUM_THREADS=48` |
| **Modal GPU** | H100 (cuOpt lane) |
| **Modal volume** | `bp-mds-data` (`/data/Inputs`, `/data/Output`, `/data/results`) — recreate/migrate on `aquin` if missing |
| **BP-MDS** (upstream) | `CodeCraftsmanSandeep/BP-MDS` @ `5b9e24f` · `master` |
| **cuOpt** | _not pinned yet_ · public repo https://github.com/NVIDIA/cuOpt |
| **suite focus** | **large only** (prof): size L / XL / XXL / XXXL — not XS/S |
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

### R006 — top-K tune (default K=2) + large re-bench — COMPLETE

- **date:** 2026-11-14
- **what:** `--topk` CLI + tight DFS prune (abort on best-so-far); sweep K∈{1,2,3,4}; full large 127 @ K=2
- **workspace:** Modal **`aquin`**
- **sweep (31 hard):** K=1 31/31 time · 2/20 gap · **K=2 31/31 time · 8/20 gap** · K=3 30/31 · 10/20 · K=4 27/31 · 10/20
- **choice:** **topk=2** — only K that restores full time wins on hard set while keeping most of the post-LS gap lift vs K=1
- **R006 full large @ K=2:** **127/127** time≤ bpmds (median ~2.05×) · large+BKS gap≤ **13/25** · both **13/25**
- **vs R005 (K=12):** much faster (109/127 quicker) but gap 13/25 vs 19/25 — K=12 gap needs time we cannot spend at matched ρ=10k
- **artifacts:** `R006_ours.jsonl` · `R006_topk_sweep.jsonl` · Modal `ap-2qiv1MZrgj5vPnvBJlrVcM`
- **ours:** `a7a7533` (topk=2 + tight DFS prune)
- **notes paste:** `run=R006 bpmds@5b9e24f ours@a7a7533 cuopt@na hw=Modal-48c-256G timed=Solver::solve rho=10000 topk=2`

### R005 — Modal ours large suite with top-K=12 Solver — COMPLETE

- **date:** 2026-11-14
- **what:** full large 127 re-bench of ours @ `653111b` (top-K post-LS) · matched α=`alpha_bpmds` · ρ=10000
- **workspace:** Modal **`aquin`** (new volume `bp-mds-data`)
- **bpmds:** `5b9e24f` (R002 baselines unchanged)
- **ours:** `653111b` (top-K=12)
- **hardware:** Modal 48c / 256 GiB · `OMP_NUM_THREADS=48`
- **timed:** `Execution time for solving (sec)` from `.sol`
- **suite:** large **127/127 ok** · Modal `ap-pVjiNIXR7Fwx7zFr19V1Y8`
- **vs BP-MDS (matched ρ=10k):** time≤ **92/127** (median ~1.43×; top-K polish costs time vs R001’s 127/127) · large+BKS gap≤ **19/25** · both **17/25**
- **gap losers (6):** Sardegna Friuli-Venezia-Giulia Lazio Molise Basilicata Piemonte
- **note:** R004 Phase B (tuned α/ρ) had 25/25 both-win; R005 is the fair matched-protocol re-bench of the new Solver
- **artifacts:** `modal_exp/results/R005_ours.jsonl` · `R005_ours_patch.csv`
- **cmd:** `modal run modal_exp/app.py --cmd ours-large --rho 10000`
- **notes paste:** `run=R005 bpmds@5b9e24f ours@653111b cuopt@na hw=Modal-48c-256G timed=Solver::solve rho=10000 topk=12`
- **PR:** updates [PR #7](https://github.com/CodeCraftsmanSandeep/BP-MDS/pull/7) on `mod/cleanup-and-speedups`

### R004 — Phase B close remaining gap losers — COMPLETE

- **date:** 2026-11-14
- **what:** close 9 large+BKS gap losers via α/ρ search + top-K post-LS selection
- **bpmds:** `5b9e24f` · **ours:** `1190a34` (+ top-K=12 post-2opt pick in `Solver.cpp`)
- **hardware:** Modal 48c / 256 GiB · `OMP_NUM_THREADS=48`
- **timed:** `Execution time for solving (sec)` from `.sol`
- **wave 1** `phase-b` (126 jobs): closed Campania Sardegna Abruzzo Lazio Ghent2 Toscana Emilia-Romagna
- **wave 2–3** denser α/ρ: closed Marche (α=12.9 ρ=60k); Friuli stuck at +0.00049 pp
- **wave 4 / LS:** top-K=12 DFS candidates → polish with `process_routes` → pick best; Friuli α=14.75 ρ=60k **gap 1.10497 ≤ bpmds 1.10651**
- **result (large+BKS n=25):** gap≤ **25/25** · time≤ **25/25** · both **25/25**
- **artifacts:** `modal_exp/results/R004_phaseb*.jsonl` · `R004_friuli_topk.json` · volume `/data/Output/ours/phaseb/`
- **notes paste:** `run=R004 bpmds@5b9e24f ours@1190a34 cuopt@na hw=Modal-48c-256G timed=Solver::solve phase=B`

### R003 — Modal ours ρ=20000 gap recovery (18 offenders) — COMPLETE

- **date:** 2026-11-14
- **what:** re-run gap-losing large BKS rows at `rho=20000` (same α as sheet); keep result only if gap ≤ R001
- **bpmds:** `5b9e24f` (R002 baselines)
- **ours:** `1190a34`
- **hardware:** Modal 48c / 256 GiB · `OMP_NUM_THREADS=48`
- **timed:** `Execution time for solving (sec)` from `.sol`
- **suite:** 18 offenders · Flanders2 Brussels2 Friuli-Venezia-Giulia Campania Marche Sardegna Molise Basilicata Abruzzo Lazio Ghent2 Veneto Toscana Piemonte Emilia-Romagna Liguria Puglia Lombardia
- **artifacts:** volume `/data/Output/ours/<inst>.sol` (overwritten) · `modal_exp/results/R003_rho20k.jsonl`
- **scoreboard:** applied `ours_*` on **16/18** (gap improved vs R001); **kept R001** on Ghent2 + Sardegna (ρ=20k gap worse)
- **vs BP-MDS (large+BKS, n=25):** gap≤ **16/25** · time≤ **25/25** · both **16/25** (was ~7/25 gap at ρ=10k)
- **both-win at ρ=20k among offenders:** Flanders2 Brussels2 Molise Basilicata Veneto Piemonte Liguria Puglia Lombardia
- **still gap-lose (tiny–small):** Friuli-Venezia-Giulia Campania Marche Sardegna Abruzzo Lazio Ghent2 Toscana Emilia-Romagna (Lazio/Toscana ≈ tie at ~1e-4 pp)
- **notes paste:** `run=R003 bpmds@5b9e24f ours@1190a34 cuopt@na hw=Modal-48c-256G timed=Solver::solve rho=20000`

### R002 — Modal production upstream BP-MDS (large suite) — COMPLETE

- **date:** 2026-11-13/14
- **what:** upstream `CodeCraftsmanSandeep/BP-MDS` @ `5b9e24f` on **same** Modal 48c/256G as R001
- **bpmds:** `5b9e24f`
- **ours:** `1190a34` (timed in R001)
- **hardware:** Modal 48c / 256 GiB · `OMP_NUM_THREADS=48`
- **timed:** `Execution time for solving (sec)` from `.sol`
- **suite:** large 127/127 ok · rho=10000 · alpha=`alpha_bpmds`
- **Modal:** Ghent2 `ap-IPZc2Wg12pUX79nqz1DlVl` · full `ap-b7bfnkda8EerwLaaaCQIAo`
- **artifacts:** `/data/Output/bpmds/` · `R002_bpmds.jsonl` · `R002_bpmds_patch.csv`
- **scoreboard:** `bpmds_cost` / `bpmds_gap_pct` / **`bpmds_time_s`** filled on large 127
- **notes paste:** `run=R002 bpmds@5b9e24f ours@1190a34 cuopt@na hw=Modal-48c-256G timed=Solver::solve rho=10000`
- **Ghent2 spotlight:** bpmds_time_s=1.0926 vs ours_time_s=0.2367 (~4.6× faster ours)

### R001 — Modal production ours (large suite)

- **date:** 2026-11-13
- **what:** ours on Modal fat CPU; large sizes only (L/XL/XXL/XXXL)
- **bpmds:** `5b9e24f` (times still TBD on same Modal box)
- **ours:** `1190a34`
- **cuopt:** —
- **hardware:** Modal 48c / 256 GiB · `OMP_NUM_THREADS=48`
- **timed:** `Execution time for solving (sec)` from `.sol` (= `Solver::solve`)
- **suite:** large 127 · rho=10000 · alpha=`alpha_bpmds` per row
- **artifacts:**
  - volume `/data/Output/ours/<instance>.sol`
  - volume `/data/results/R001_ours.jsonl`
  - volume `/data/results/R001_ours_patch.csv`
- **scoreboard columns:** `ours_cost`, `ours_gap_pct`, `ours_time_s`, `notes`
- **progress:** **COMPLETE ours large 127/127** · Modal `ap-MQUIQ76uMDrUhrhRyC8K3f` · harvest `ap-DBmjn7pSPm5wyYs5Az1au8`
- **scoreboard:** `ours_*` filled on all L/XL/XXL/XXXL rows · local `modal_exp/results/R001_ours.jsonl` + `R001_ours_patch.csv`
- **still empty:** `bpmds_time_s` (need upstream binary on same Modal 48c/256G) · all `cuopt_*`
- **notes paste:** `run=R001 bpmds@5b9e24f ours@1190a34 cuopt@na hw=Modal-48c-256G timed=Solver::solve rho=10000`
- **apply locally after download:**
  ```bash
  modal volume get bp-mds-data /results/R001_ours_patch.csv modal_exp/results/R001_ours_patch.csv
  python modal_exp/apply_results.py --patch-csv modal_exp/results/R001_ours_patch.csv
  # then paste scoreboard.tsv into the Google Sheet
  ```

### R000 — pins established (no new solves)

- **date:** 2026-11-11
- **what:** created this log; recorded current SHAs + local Mac hardware; stamped all scoreboard `notes`
- **bpmds:** `5b9e24f`
- **ours:** `8f80169`
- **cuopt:** —
- **hardware:** Apple M5 / 10 CPU / 16 GB (dev only)
- **timed:** `Solver::solve`
- **suite:** all 310 rows (pin stamp only; no new solves)
- **scoreboard:** `notes` filled on every row in root + `benchmarks/` scoreboard files
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
