# config/paths.py
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
CODE_DIR = PROJECT_ROOT / "code" / "python"
DATA_DIR = CODE_DIR / "data"
FIGURES_DIR = PROJECT_ROOT / "figures" / "figures_from_python"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
