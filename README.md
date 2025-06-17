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

# Verify installation
python -c "import qspectro2d; print('Package installed successfully!')"
```

### Environment Management

#### Update environment (when environment.yml changes):
```bash
conda env update -f environment.yml
```

#### Export current environment (if you add new packages):
```bash
conda env export > environment.yml
```

#### Remove environment (if needed):
```bash
conda env remove -n master_env
```

### Using Black Code Formatter with Notebooks

This project includes Black with Jupyter support for consistent code formatting:

```bash
# Format Python files
black code/python/

# Format Jupyter notebooks
black notebooks/

# Check what would be formatted (dry run)
black --diff code/python/
```

### VS Code Integration

For VS Code users, the environment should be automatically detected. You can also:
1. Open Command Palette (Ctrl+Shift+P)
2. Select "Python: Select Interpreter"
3. Choose the conda environment: `master_env`

## Project Structure

```
master_thesis/
├── code/python/         # Main Python package
├── notebooks/           # Jupyter notebooks
├── latex/               # LaTeX thesis files
├── figures/             # Generated figures
├── environment.yml      # Conda environment specification
├── requirements.txt     # Pip requirements (backup)
└── README.md           # This file
```