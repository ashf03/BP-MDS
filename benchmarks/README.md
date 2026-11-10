# Scoreboard

**One sheet** for everything: BP-MDS · ours · cuOpt — **BKS · cost · gap% · time**.

| | |
|:--|:--|
| Live sheet | [Google Sheet](https://docs.google.com/spreadsheets/d/18Z0JchFiihrfq31Alr3_XLvNnxVLcktAKo5pzkanE6s/edit?usp=sharing) |
| Repo mirror (paste) | [`scoreboard.tsv`](scoreboard.tsv) |
| Repo mirror (CSV) | [`scoreboard.csv`](scoreboard.csv) |

All **310** instances (CVRPLIB 130 · FILO2 20 · Synthetic 160). Filter the `source` column in Sheets when you only want CVRPLIB.

**Prefilled:** `bks`, `alpha_bpmds`, `bpmds_cost`, `bpmds_gap_pct`  
**Fill as we run:** `*_time_s`, all `ours_*`, all `cuopt_*`

$$
\mathrm{gap}_{\%} = 100 \times \frac{\mathrm{cost} - \mathrm{BKS}}{\mathrm{BKS}}
$$

Rebuild from catalogs:

```bash
python3 benchmarks/build_scoreboard.py
```
