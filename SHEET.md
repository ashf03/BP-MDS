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

Metrics per solver: **cost · gap% · time_s**. Shared: **BKS**.

```text
gap% = 100 * (cost - BKS) / BKS
```

## Header (row 1) — do not rename casually

```text
sno	size	source	set	instance	path	customers	n	q	bks	alpha_bpmds	bpmds_cost	bpmds_gap_pct	bpmds_time_s	ours_cost	ours_gap_pct	ours_time_s	cuopt_cost	cuopt_gap_pct	cuopt_time_s	notes
```

## Agent: update Google Sheet from `scoreboard.tsv`

1. Auth Google Sheets MCP if needed (`mcp_auth` on `plugin-google-sheets-google-sheets`).
2. `get_spreadsheet` on ID `18Z0JchFiihrfq31Alr3_XLvNnxVLcktAKo5pzkanE6s` — note the **single** tab title + sheetId.
3. Read root `scoreboard.tsv` (tab-separated; first row = header; 310 data rows).
4. Replace the tab contents with that table (header at A1). Prefer a full rewrite of the used range over appending duplicates.
5. Keep **one tab**. Freeze header row. Optional: filter on `source` for CVRPLIB.
6. After runs: edit `scoreboard.tsv` first (source of truth), then push to Sheets again the same way.

### Manual paste (no MCP)

1. Open `scoreboard.tsv` → select all → copy.
2. Open the Sheet URL → click A1 → paste.
3. Data → Create a filter (optional).

## Agent: update `scoreboard.tsv` after a run

When given new results, fill the matching `instance` row:

| Columns | Meaning |
|:--------|:--------|
| `bpmds_cost` `bpmds_gap_pct` `bpmds_time_s` | Upstream BP-MDS |
| `ours_cost` `ours_gap_pct` `ours_time_s` | This fork |
| `cuopt_cost` `cuopt_gap_pct` `cuopt_time_s` | NVIDIA cuOpt |
| `notes` | hardware, commits, cuOpt version, timed region |

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
