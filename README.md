# Master Thesis – 2D Electronic Spectroscopy

Central monorepo for the thesis: simulation code, plotting utilities, generated figures, and LaTeX manuscript sources. Everything lives in-place for smooth syncing between local development and HPC runs.

## Contents at a glance
- `code/python/qspectro2d` – spectroscopy simulation engine (see its README for detailed docs)
- `code/python/plotstyle` – Matplotlib/LaTeX styling helpers
- `code/python/README.md` – quick-start for path constants and CLI entry points
- `latex/` – thesis manuscript
- `figures/` – generated figures (Python + manual)
- `environment.yml` – shared Conda env (`master_env`)

## Getting started
```bash
git clone git@github.com:LBPhysics/Master_thesis.git
cd Master_thesis
conda env create -f environment.yml
conda activate master_env
```

Verify the stack:
```bash
python - <<'PY'
import qutip, qspectro2d, plotstyle
print("QuTiP:", qutip.__version__)
print("qspectro2d:", getattr(qspectro2d, "__version__", "dev"))
PY
```

## Repository layout
```
Master_thesis/
├── code/
│   └── python/
│       ├── qspectro2d/        # simulation package
│       ├── plotstyle/         # plotting style package
│       ├── scripts/           # CLI + SLURM helpers + YAML configs
│       └── README.md          # usage guide for thesis_paths & scripts
├── data/                      # simulation outputs (generated)
├── figures/                   # exported figures (generated)
├── latex/                     # thesis text
├── environment.yml            # Conda spec (env name: thesis)
└── README.md                  # this overview
```

Sparse checkout (code + figures only):
```bash
git sparse-checkout init --cone
git sparse-checkout set --skip-checks code figures environment.yml README.md
```

## Workflow summary
1. **Configure** – Copy/edit a YAML under `code/python/scripts/simulation_configs/`.
2. **Simulate** – Run `calc_datas --sim_type {1d,2d}` locally or via SLURM (`hpc_calc_datas.py` with n_batches).
3. **Aggregate** – Stack inhomogeneous runs (`stack_inhomogenity.py`) and build 2D datasets (`stack_times`).
4. **Plot** – Use `plot_datas.py` (or `hpc_plot_datas.py`) to create time/frequency figures; outputs land in `figures/figures_from_python/`.
5. **Document** – Update LaTeX sources under `latex/` and reference generated figures.

## HPC checklist
```bash
git pull --ff-only
conda activate master_env
conda env update -f environment.yml --prune   # only when dependencies changed
python code/python/scripts/hpc_calc_datas.py --n_batches N --sim_type 2d
```

Generated SLURM scripts store logs in `code/python/scripts/batch_jobs/`.

## References
- `code/python/README.md` – top-level CLI + path instructions
- `code/python/qspectro2d/README.md` – detailed simulation docs and YAML schema
- `code/python/plotstyle/README.md` – plotting style usage
- `latex/` – thesis structure, chapter templates, bibliography