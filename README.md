# Master Thesis: 2D Electronic Spectroscopy

This repository contains the full simulation, plotting, and thesis sources for 1D/2D electronic spectroscopy, built on top of QuTiP. It now follows a single monorepo model (Option A) with BOTH internal packages developed in-place (editable installs) for frictionless local ↔ HPC synchronization.

## Monorepo components

- `code/python/qspectro2d` – simulation engine (package: `qspectro2d`)
- `code/python/plotstyle` – figure styling utilities (package: `plotstyle`)
- `code/python/scripts` – CLI scripts (HPC + local)
- `latex/` – thesis LaTeX sources
- `figures/` – generated figures
- `environment.yml` – single consistent Conda environment (name: `thesis`)

## Environment setup (local & HPC)

Prerequisites: Miniconda or Anaconda.

```bash
# From repo root
conda env create -f environment.yml   # first time
conda activate thesis
# Later updates (after editing environment.yml):
conda env update -f environment.yml --prune

# Quick verification
python - <<'PY'
import qutip, qspectro2d, plotstyle
from qutip import basis
print('OK:', qutip.__version__)
print('OK:', getattr(qspectro2d, '__version__', ''))
psi = basis(2,0)
print('qutip basis dim:', psi.shape)
PY
```

## Project structure

```
Master_thesis/
├── code/
│   └── python/
│       ├── qspectro2d/              # Core spectroscopy simulations (package)
│       ├── plotstyle/               # Plot styling (package)
│       ├── scripts/                 # HPC + local CLI scripts
│       └── paths.py (legacy helpers, will be folded in later)
├── figures/
│   └── figures_from_python/
├── latex/
├── environment.yml
├── Makefile
└── README.md
```

# To only clone code, figures, environment.yml, README.md (no latex): Do a sparse checkout
```bash
git sparse-checkout init --cone
 git sparse-checkout set --skip-checks code figures environment.yml README.md
 ```


## Scripts and workflow

All scripts live in `code/python/scripts` and are designed to compose:

1) Configure inputs
- Put a YAML in `scripts/simulation_configs/` (use the provided templates).
- Mark the active config by including a `*` in its filename (auto-picked).

2) Run simulations
- Local 1D or 2D-style sweep:
    - `python calc_datas.py --sim_type 1d`
    - `python calc_datas.py --sim_type 2d` (iterates over t_det as t_coh)
- HPC batching via SLURM (auto-generates and submits jobs):
    - `python hpc_calc_datas.py --n_batches 10 --sim_type 2d`
    - `python hpc_calc_datas.py --n_batches 5 --sim_type 1d`

3) Stack outputs
- Stack multiple t_coh results into one 2D dataset:
    - `python stack_times.py --abs_path "/abs/path/to/one/_data.npz" --skip_if_exists`
- Average inhomogeneous configurations at fixed t_coh:
    - `python stack_inhomogenity.py --abs_path "/abs/path/to/one/_data.npz"`

4) Plot
- Local plotting of 1D/2D with time and frequency domains:
    - `python plot_datas.py --abs_path "/abs/path/to/*_data.npz" --extend 10`
- HPC plotting helper (stacks then plots):
    - `python hpc_plot_datas.py --abs_path "/abs/path/to/1d_dir_or_file"`

Notes
- Scripts auto-detect data/figure locations via `project_config.paths`.
- Saved files include metadata needed by stacking and plotting.

## HPC workflow (simplified)

1. Local: develop, then `git add -u && git commit -m "msg" && git push`.
2. HPC first time: clone repo, create env (see above).
3. Subsequent runs:
    ```bash
    git pull --ff-only
    conda activate thesis
    conda env update -f environment.yml --prune  # only if deps changed
    ```
4. (Optional) For reproducibility, record commit hash manually:
    ```bash
    git rev-parse HEAD > COMMIT_HASH.txt
    ```

## Troubleshooting
- To re-sync environment after dependency edits: `conda env update -f environment.yml --prune`.
- For cluster usage, make sure `sbatch` is available on PATH before running the HPC helpers.
