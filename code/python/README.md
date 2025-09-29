# thesis_paths

Utility layer for the spectroscopy thesis project: discovers repository paths, exports reusable directory constants, and bundles the CLI scripts that drive simulations, stacking, and plotting.

## Install
```bash
pip install -e ./code/python              # thesis_paths + console scripts
pip install -e ./code/python/qspectro2d   # simulation engine
pip install -e ./code/python/plotstyle    # plotting style (optional but recommended)
```

## Key exports
All the paths relevant for the thesis project:
```python
from thesis_paths import (
    PROJECT_ROOT,
    DATA_DIR,
    FIGURES_DIR,
    FIGURES_PYTHON_DIR,
    FIGURES_TESTS_DIR,
    PYTHON_CODE_DIR,
    SCRIPTS_DIR,
    NOTEBOOKS_DIR,
    LATEX_DIR,
    SIM_CONFIGS_DIR,
)
```

## Workflow snapshot
1. Edit a YAML in `scripts/simulation_configs/` (templates for monomer/dimer setups).
2. Run `python calc_datas --sim_type {1d,2d}` (or `python scripts/calc_datas.py`) to generate 1D traces in `data/1d_spectroscopy/...`.
3. For inhomogeneous runs, average with `python scripts/stack_inhomogenity.py --abs_path <*_data.npz>`.
4. Stack coherence scans via `stack_times --abs_path <*_data.npz>` to create 2D datasets.
5. Plot with `python scripts/plot_datas.py --abs_path <data_or_info>`; figures land in `figures/figures_from_python/...`.

## CLI essentials
- `calc_datas` – simulation runner with batching (`--n_batches`, `--batch_idx`).
- `stack_times` – convert per-`t_coh` files into a single 2D bundle (`--skip_if_exists`).
- `stack_inhomogenity.py` – average inhomogeneous 1D configs.
- `plot_datas.py` – time/frequency plots with zero-padding control (`--extend`).

## HPC scripts
- `python scripts/hpc_calc_datas.py --n_batches N --sim_type {1d,2d}` creates/submits SLURM jobs per batch.
- `python scripts/hpc_plot_datas.py --abs_path <1d_dir_or_file>` stacks to 2D, writes a plotting SLURM script, and submits unless `--no_submit` is set.

## Related packages
- [`qspectro2d`](./qspectro2d/README.md)
- [`plotstyle`](./plotstyle/README.md)