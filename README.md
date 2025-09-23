# Master Thesis: 2D Electronic Spectroscopy

## Environment Setup

This project uses conda for environment management. Follow these steps to set up the development environment:

### Prerequisites
- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed

### Quick Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd master_thesis

# Create and activate the conda environment
conda env create -f environment.yml
conda activate master_env

# Install local packages in editable mode (if not auto-installed)
pip install -e code/python

# Verify installation
python -c "import qspectro2d; print('Package installed successfully!')"



## Project Structure

```
master_thesis/
├── code/python/         # Main Python package
├── notebooks/           # Jupyter notebooks
├── latex/               # LaTeX thesis files
├── figures/             # Generated figures
├── environment.yml      # Conda environment specification (single source of dependencies)
├── code/python/pyproject.toml  # Packaging config for editable install
└── README.md           # This file -> For package‑specific docs, see `code/python/README.md`.
```

## Simulation Pipeline (High-Level)

This section explains the core end‑to‑end flow used by both 1D and 2D spectroscopy simulations. It clarifies how the modular pieces you refactored interact.

1. Configure Inputs
    - create a *your_sim.yaml based on the template.yaml file to get desired simulation parameters.
    This is an example file in the SCRIPTS / simulation_configs folder which explains all possible parameters. The * indicates that this file will be used to run the simulation.

2. Run the Simulation (with suitable cli params)
    - Either on a HPC -> run the hpc_calc_datas.py file
    - Or locally -> run the calc_datas.py file.
    calc_datas.py base functionality: calc the 1d radiated electric field for one combination of t_coh, t_wait over t_det
3. If needed stack the results into a single file using 
    - Either stack_times.py -> stacks the different 1d datas for many t_coh 
    - Or stack_inhom.py
4. Plot the results using
    - Either hpc_plot_datas.py (with suitable cli params)
    -> also does the stacking of times
    - Or plot_datas.py