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
| **Modal GPU** | H100 (cuOpt + ours-gpu lane; ours-gpu = bucketed BP-MDS on H100 host OpenMP) |
| **Modal volume** | `bp-mds-data` (`/data/Inputs`, `/data/Output`, `/data/results`) — recreate/migrate on `aquin` if missing |
| **BP-MDS** (upstream) | `CodeCraftsmanSandeep/BP-MDS` @ `5b9e24f` · `master` |
| **cuOpt** | `cuopt-cu12` **26.10.01** (pypi.nvidia.com) · https://github.com/NVIDIA/cuOpt |
| **suite focus** | **large only** (prof): size L / XL / XXL / XXXL — not XS/S |
| **timed region (CPU)** | `Solver::solve` wall time (seconds) |
| **timed region (cuOpt)** | `cuopt.routing.Solve` only (exclude VRP parse + dense matrix build) |
| **gap** | `100 * (cost - BKS) / BKS` |
| **local hardware (dev)** | Apple M5 · 10 logical CPUs · 16 GB · arm64 (not the fair cluster box) |

**Notes string template** (paste into scoreboard `notes`):

```text
run=R00X bpmds@<sha> ours@<sha> cuopt@<tag|na> hw=<spec> timed=<region>
```

---

## Run log

Newest first. Add a block per batch.

### R011 — stronger LS (Or-opt / relocate / 3-opt) — COMPLETE

- **date:** 2026-11-14
- **what:** `--ls=0|1|2` — after 2-opt: **Or-opt** (L=1..3, L=1≡relocate) + **inter-route relocate**; ls=2 adds **limited 3-opt** (segment swap/reverse, n≤64)
- **code:** `Solver::tsp_or_opt` / `tsp_3opt` / `inter_route_relocate` · default `--ls=0` keeps prior time protocol
- **workspace:** Modal **`aquin`** · 48c/256G · knobs=`R009_knobs.json` · topk=2
- **raw:** ls=2 → gap↓ on **25/25** but time broke both-win on 11/25 · ls=1 recovered those with smaller time hit
- **apply:** per instance pick best **both-win** among {prior, ls=1, ls=2} → large+BKS still **25/25** both-win vs BP-MDS · **14** rows stamped R011 (better gap, time≤bpmds)
- **artifacts:** `R011_ours_ls1.jsonl` · `R011_ours_ls2.jsonl` · Modal `ap-Pi3OhP4bOSvP3Ir4LwLCMh` (ls2) · ls1 follow-up
- **notes paste:** `run=R011 … ls=1|2 ops=2opt+oropt+relocate[+3opt] topk=2`

### R010 — CUDA Prim / 2-opt inside buckets (ours-gpu-cuda) — COMPLETE

- **date:** 2026-11-14
- **what:** optional CUDA accelerate **Prim MST** + **2-opt** per bucket (device mem O(bucket) / O(route) — **still no dense N×N**)
- **code:** `Lib/Cuda/cuda_hotpath.cu` · `#ifdef BPMDS_USE_CUDA` in `Solver.cpp` · `make cuda` · Modal `ours-cuda-*`
- **knobs:** `BPMDS_CUDA=1` · `BPMDS_CUDA_PRIM_MIN=512` · `BPMDS_CUDA_2OPT_MIN=128` · OpenMP 48 + H100 mutex around kernels
- **workspace:** Modal **`aquin`** · H100 · timed=`Solver::solve`
- **suite:** large+BKS **25/25 ok**
- **vs R009 (host OpenMP only):** faster/eq **16/25** (big wins on XXL: Puglia −160s, Emilia −114s, Sicilia −79s) · gap≤ **10/25** (2-opt is best-improvement on GPU → slight gap churn)
- **vs cuOpt (5 fit):** still **5/5** both-win
- **artifacts:** `R010_ours_gpu_cuda.jsonl` · Modal `ap-7Rrd4cAvl8HUDhFeIGqtjk`
- **scoreboard:** `ours_gpu_*` updated to R010
- **notes paste:** `run=R010 … lane=ours-gpu-cuda algo=bucketed-euclidean+cuda-prim-2opt omp=48 prim_min=512 twopt_min=128 topk=2`

### R009 — ours-gpu (bucketed Euclidean on H100) vs cuOpt — COMPLETE

- **date:** 2026-11-14
- **what:** GPU-lane ours — same BP-MDS fork (angular α buckets, O(n) memory, **no dense N×N**) on Modal **H100** host OpenMP (`OMP_NUM_THREADS=48`)
- **workspace:** Modal **`aquin`** · GPU **H100 80GB** present (nvidia-smi logged) · timed=`Solver::solve`
- **knobs:** `modal_exp/results/R009_knobs.json` (R007 Phase B winners + `alpha_bpmds`/ρ=10k elsewhere) · topk=2
- **suite:** all **large+BKS 25/25 ok** (includes 20 with n>25k where stock cuOpt is skip)
- **vs cuOpt (5 dense-fit):** gap≤ **5/5** · time≤ **5/5** · both **5/5**
  - e.g. Ghent2: ours-gpu 13.26% / 0.28s · cuOpt 17.62% / 31.6s
- **n>25k:** **20/20** solved (cuOpt skip); e.g. Molise n=50k 1.93% / 1.3s · Lazio n=1M 1.39% / 57.7s
- **artifacts:** `R009_ours_gpu.jsonl` · `R009_knobs.json` · Modal `ap-vVhoXbaR2pXxC6mU9oxuzD`
- **scoreboard:** new cols `ours_gpu_cost` / `ours_gpu_gap_pct` / `ours_gpu_time_s`
- **notes paste:** `run=R009 bpmds@5b9e24f ours@a7a7533 cuopt@26.10.01 hw=Modal-H100 timed=Solver::solve lane=ours-gpu algo=bucketed-euclidean omp=48 topk=2`
- **next:** (done in R010) CUDA Prim/2-opt; optional stronger LS beyond 2-opt

### R008 — NVIDIA cuOpt H100 baseline on dense-fit large rows — COMPLETE

- **date:** 2026-11-14
- **what:** pin cuOpt + fill `cuopt_*` on large suite where dense N×N fits H100
- **workspace:** Modal **`aquin`** · GPU **H100 80GB**
- **cuOpt:** `cuopt-cu12` **26.10.01** · timed=`routing.Solve` · `time_limit=30s` · `max_n=25000`
- **fit set (n≤25k):** Ghent2 · Brussels1 · Brussels2 · Valle-D-Aosta · Flanders1 (**5/127**)
- **oversized:** **122/127** stamped `cuopt=skip:n>25000` (stock cuOpt needs dense cost matrix; XXL/XXXL need ours-gpu / sparse later)
- **results (5 fit):** all returned finite costs (Valle/Flanders status=1 time-limit incumbent)
  - vs ours (CPU 48c): **ours wins time 5/5** and **gap 5/5** at this protocol
  - e.g. Ghent2: ours gap 13.26% / 0.20s · cuOpt 17.62% / 31.6s
- **artifacts:** `modal_exp/results/R008_cuopt.jsonl` · `R008_gpu_ready.log` · Modal `ap-UaaSjLWlJD1Yjddr5xI72d` (smoke) · fit run in `R008_cuopt_fit.log`
- **notes paste:** `run=R008 bpmds@5b9e24f ours@a7a7533 cuopt@26.10.01 hw=Modal-H100 timed=routing.Solve time_limit=30s max_n=25000`
- **next:** ours-gpu (bucketed Euclidean, no dense N×N) to compete on n>25k; optional stronger LS beyond 2-opt

### R007 — Phase B α/ρ at topk=2 on 12 R006 gap losers — COMPLETE

- **date:** 2026-11-14
- **what:** close remaining large+BKS gap losers with α/ρ search at **topk=2** (no higher K)
- **targets:** Friuli-Venezia-Giulia Sardegna Marche Campania Lazio Basilicata Molise Abruzzo Piemonte Emilia-Romagna Liguria Puglia
- **workspace:** Modal **`aquin`** · ours `@a7a7533`
- **batches:** `phase-b-batch1/2` (72+72) · Friuli follow-ups `phase-b4` + `phase-b5`
- **apply rule:** gap ≤ bpmds **and** time ≤ bpmds (reject gap-only wins that blow the time budget)
- **result (large+BKS n=25):** gap≤ **25/25** · time≤ **25/25** · both **25/25** · full large time still **127/127**
- **Friuli close:** α=14.75 ρ=75000 topk=2 → gap 1.10635 ≤ bpmds 1.10651
- **artifacts:** `R007_phaseb_batch{1,2}.jsonl` · `R007_phaseb4/5.jsonl` · `R007_phaseb_best.json`
- **notes paste:** `run=R007 bpmds@5b9e24f ours@a7a7533 cuopt@na hw=Modal-48c-256G timed=Solver::solve phase=B topk=2`

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
