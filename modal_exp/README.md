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
modal run modal_exp/app.py --cmd ours-large --rho 10000 --topk 2
modal run modal_exp/app.py --cmd harvest

# GPU / cuOpt (dense N×N — only n ≤ CUOPT_MAX_N ≈ 25k fits H100)
modal run modal_exp/app.py --cmd gpu-ready
modal run modal_exp/app.py --cmd cuopt-one --instance=Ghent2 --time-limit 30
modal run modal_exp/app.py --cmd cuopt-fit --time-limit 30 --max-n 25000

# GPU lane ours (bucketed Euclidean on H100 host — no dense N×N; knobs in R009_knobs.json)
modal run modal_exp/app.py --cmd ours-gpu-one --instance=Ghent2
modal run modal_exp/app.py --cmd ours-gpu-fit --max-n 25000
modal run modal_exp/app.py --cmd ours-gpu-bks
python modal_exp/apply_results.py --solver ours_gpu --jsonl modal_exp/results/R009_ours_gpu.jsonl

# CUDA Prim / 2-opt inside buckets (still O(bucket), no full N×N)
modal run modal_exp/app.py --cmd ours-cuda-one --instance=Ghent2
modal run modal_exp/app.py --cmd ours-cuda-bks
python modal_exp/apply_results.py --solver ours_gpu --jsonl modal_exp/results/R010_ours_gpu_cuda.jsonl

# Stronger LS: --ls=1 Or-opt/relocate, --ls=2 + limited 3-opt
modal run modal_exp/app.py --cmd ours-ls-bks --ls 2
python modal_exp/apply_results.py --solver ours --jsonl modal_exp/results/R011_ours_ls.jsonl
```

**Scale note:** stock cuOpt needs a dense cost matrix (`n > max_n` → skip). **ours-gpu** / **ours-cuda** use angular buckets (O(n) memory) on H100 — fills `ours_gpu_*`. CUDA path: `BPMDS_USE_CUDA`, thresholds `BPMDS_CUDA_PRIM_MIN` (default 512) / `BPMDS_CUDA_2OPT_MIN` (128).

