# Modal — production only

**No Sample. No toy. No stubs.** Large suite only: `L` / `XL` / `XXL` / `XXXL`.

| | |
|:--|:--|
| CPU | 48 cores · 256 GiB |
| GPU | H100 |
| Volume | `bp-mds-data` → `/data/Inputs/...` |
| Profile | **`aquin`** (`modal profile activate aquin`) |

```bash
modal profile activate aquin
source .venv-modal/bin/activate

# volume + inputs (once per workspace)
modal volume create bp-mds-data 2>/dev/null || true
# Inputs from release inputs-v1 unpacked under /tmp/bp-mds-inputs/Inputs
modal volume put bp-mds-data /tmp/bp-mds-inputs/Inputs /Inputs

modal run modal_exp/app.py --cmd list-large
modal run modal_exp/app.py --cmd ours-large --rho 10000
modal run modal_exp/app.py --cmd harvest
```
