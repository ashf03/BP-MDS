# Scoreboard handoff (local — for agents)

**Send this file + `scoreboard.tsv` to an agent.** One Google Sheet tab. Keep the TSV and the live sheet in sync.

## Live sheet

| | |
|:--|:--|
| URL | https://docs.google.com/spreadsheets/d/18Z0JchFiihrfq31Alr3_XLvNnxVLcktAKo5pzkanE6s/edit?usp=sharing |
| Spreadsheet ID | `18Z0JchFiihrfq31Alr3_XLvNnxVLcktAKo5pzkanE6s` |
| Tabs | **one tab only** (do not add extra sheets) |
| Local paste file | `scoreboard.tsv` (repo root) · also `scoreboard.csv` |

## What the sheet is

Single scoreboard: **BP-MDS · ours · cuOpt** on all **310** instances (CVRPLIB 130 · FILO2 20 · Synthetic 160).  
**Primary fight (prof):** large sizes only — `L` / `XL` / `XXL` / `XXXL`. cuOpt source: https://github.com/NVIDIA/cuOpt

Metrics per solver: **cost · gap% · time_s**. Shared: **BKS**. Pins/runs: root `log.md`.

```text
gap% = 100 * (cost - BKS) / BKS
```

## Header (row 1) — do not rename casually

```text
sno	size	source	set	instance	path	customers	n	q	bks	alpha_bpmds	bpmds_cost	bpmds_gap_pct	bpmds_time_s	ours_cost	ours_gap_pct	ours_time_s	cuopt_cost	cuopt_gap_pct	cuopt_time_s	ours_gpu_cost	ours_gpu_gap_pct	ours_gpu_time_s	notes
```

## Agent: update Google Sheet from `scoreboard.tsv`

1. Auth Google Sheets MCP if needed (`mcp_auth` on `plugin-google-sheets-google-sheets`).
2. `get_spreadsheet` on ID `18Z0JchFiihrfq31Alr3_XLvNnxVLcktAKo5pzkanE6s` — note the **single** tab title + sheetId.
3. Read root `scoreboard.tsv` (tab-separated; first row = header; 310 data rows).
4. Replace the tab contents with that table (header at A1). Prefer a full rewrite of the used range over appending duplicates.
5. Keep **one tab**. Freeze header row.
6. After runs: edit `scoreboard.tsv` first (source of truth), then push to Sheets again the same way.

### Manual paste (no MCP)

1. Open `scoreboard.tsv` → select all → copy.
2. Open the Sheet URL → click A1 → paste.
3. Data → Create a filter (optional).

## Recording (Modal R001+)

Timed region = **`Execution time for solving (sec)`** inside each `.sol` (not wall clock around the process).

After a Modal batch:

```bash
modal volume get bp-mds-data /results/R001_ours_patch.csv modal_exp/results/R001_ours_patch.csv
# or harvest if patch incomplete:
# modal run modal_exp/app.py --cmd harvest
# modal volume get bp-mds-data /results/R001_ours_patch.csv modal_exp/results/R001_ours_patch.csv

python modal_exp/apply_results.py --patch-csv modal_exp/results/R001_ours_patch.csv
# then full-replace the Google Sheet tab from scoreboard.tsv
```

Fill `ours_*` from ours runs; `bpmds_time_s` only from upstream binary on the **same** Modal CPU shape; `cuopt_*` from H100 runs. Always cite `log.md` run id in `notes`.

## Agent: update `scoreboard.tsv` after a run

When given new results, fill the matching `instance` row:

| Columns | Meaning |
|:--------|:--------|
| `bpmds_cost` `bpmds_gap_pct` `bpmds_time_s` | Upstream BP-MDS |
| `ours_cost` `ours_gap_pct` `ours_time_s` | This fork (CPU lane, Modal 48c) |
| `cuopt_cost` `cuopt_gap_pct` `cuopt_time_s` | NVIDIA cuOpt (H100; dense N×N) |
| `ours_gpu_cost` `ours_gpu_gap_pct` `ours_gpu_time_s` | Ours GPU lane — bucketed Euclidean on H100 host (no dense N×N) |
| `notes` | cite `log.md` run id, e.g. `run=R001 bpmds@… ours@… cuopt@… hw=… timed=…` |

Pins, hardware, and timed region live in root **`log.md`** (append a run block every batch).  
Then sync the Google Sheet (steps above).

## Rebuild TSV from repo catalogs

```bash
python3 benchmarks/build_scoreboard.py
# writes root scoreboard.tsv / scoreboard.csv and benchmarks/ mirrors
```

## Win rule (per instance)

- `ours_gap_pct` ≤ contender gap
- `ours_time_s` ≤ contender time
- same instance + knobs recorded in `notes`
