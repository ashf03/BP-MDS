# Scoreboard

**One Google Sheet** for everything. Agent paste + brief at repo root (pushable):

| | |
|:--|:--|
| Live sheet | [Google Sheet](https://docs.google.com/spreadsheets/d/18Z0JchFiihrfq31Alr3_XLvNnxVLcktAKo5pzkanE6s/edit?usp=sharing) |
| Agent handoff | [`../SHEET.md`](../SHEET.md) + [`../scoreboard.tsv`](../scoreboard.tsv) |
| Mirror here | [`scoreboard.tsv`](scoreboard.tsv) · [`scoreboard.csv`](scoreboard.csv) |

Rebuild (writes root + `benchmarks/`):

```bash
python3 benchmarks/build_scoreboard.py
```

Record solves into the scoreboard (`.sol` Cost + Solver::solve time):

```bash
python3 modal_exp/record_sols.py --solver ours --sol-dir Output/ours --dry-run
python3 modal_exp/record_sols.py --solver ours --sol-dir Output/ours --notes 'run=R0XX …'
```
