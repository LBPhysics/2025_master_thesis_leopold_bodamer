# Master Thesis: 2D Electronic Spectroscopy

This repository contains the full simulation and plotting pipeline for 1D/2D electronic spectroscopy, built on top of QuTiP. It is organized as a single repo with local Python utilities and two local submodules used in editable mode.

## What this project uses

- Conda environment defined in `environment.yml` (single source of truth)
- Local editable packages installed via pip:
    - `code/python` → installs `project_config` (paths, small helpers)
    - `external/qspectro2d` → installs `qspectro2d` (simulation core, data I/O, plotting API)
    - `external/plotstyle` → installs `plotstyle` (Matplotlib style + save helpers)
- Core scientific stack: Python 3.11, NumPy, SciPy, Matplotlib, QuTiP, tqdm
- CLI scripts in `code/python/scripts` to run/stack/plot and HPC helpers
- Data written under `data/1d_spectroscopy` and `data/2d_spectroscopy`
- Figures written under `figures/figures_from_python`

## Environment setup

Prerequisites: Miniconda or Anaconda.

```bash
# From repo root
conda env create -f environment.yml
conda activate master_env

# (Editable installs are done automatically by conda; if needed, re-run)
pip install -e code/python -e external/qspectro2d -e external/plotstyle

# Quick verification
python - <<'PY'
import qutip, qspectro2d, project_config, plotstyle
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
│       ├── project_config/         # Local utils (paths etc.)
│       └── scripts/                # CLI entry points (see below)
├── external/
│   ├── qspectro2d/                 # Simulation package (installed editable)
│   └── plotstyle/                  # Plotting styles (installed editable)
├── data/
│   ├── 1d_spectroscopy/            # Saved 1D results (per t_coh or per inhom config)
│   └── 2d_spectroscopy/            # Stacked 2D results
├── figures/
│   └── figures_from_python/        # Saved figures
├── latex/                          # Thesis sources
├── environment.yml                 # Conda env spec
└── README.md
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

## Troubleshooting

- If imports fail, ensure the environment is active: `conda activate master_env`.
- If local packages are not importable, re-run editable installs:
    `pip install -e code/python -e external/qspectro2d -e external/plotstyle`.
- For cluster usage, make sure `sbatch` is available on PATH before running the HPC helpers.
