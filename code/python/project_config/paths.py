"""Project path definitions (no side effects).

This module exposes lazily-resolved paths relative to the detected project root.
It intentionally performs no filesystem mutations at import time (no directory
creation) to keep imports pure and predictable.

Call :func:`ensure_dirs` explicitly in entry‑point scripts before writing data
or figures.

Env override: set QSPECTRO2D_ROOT to the repository root to bypass discovery.
"""

from pathlib import Path
import os


def find_project_root() -> Path:
    """Locate the repository root.

    Strategy (in order):
    1. Environment variable override: ``QSPECTRO2D_ROOT`` if it points to an existing dir.
    2. Ascend parents from this file looking for a marker set that characterises the repo
       (``.git`` directory OR simultaneous presence of ``environment.yml`` and ``requirements.txt``).
    3. Fallback: first parent whose name contains ``master_thesis`` (case‑insensitive).
    """

    override = os.environ.get("QSPECTRO2D_ROOT")
    if override:
        p = Path(override).expanduser().resolve()
        if p.is_dir():
            return p

    current_path = Path(__file__).resolve()
    marker_files = {"environment.yml", "requirements.txt"}

    for parent in current_path.parents:
        if (parent / ".git").is_dir():
            return parent
        if all((parent / m).exists() for m in marker_files):
            return parent

    for parent in current_path.parents:
        if "master_thesis" in parent.name.lower():
            return parent

    raise RuntimeError(
        "Could not determine project root. Set 'QSPECTRO2D_ROOT' to the repository path"
    )


PROJECT_ROOT = find_project_root()

# Base directories (pure)
DATA_DIR = (PROJECT_ROOT / "data").resolve()
FIGURES_DIR = (PROJECT_ROOT / "figures").resolve()

# Additional useful paths
PYTHON_CODE_DIR = PROJECT_ROOT / "code" / "python"
SCRIPTS_DIR = PYTHON_CODE_DIR / "scripts"
NOTEBOOKS_DIR = PYTHON_CODE_DIR / "notebooks"
LATEX_DIR = PROJECT_ROOT / "latex"

# Figure subdirectories (lazy)
FIGURES_PYTHON_DIR = FIGURES_DIR / "figures_from_python"
FIGURES_TESTS_DIR = FIGURES_PYTHON_DIR / "tests"


def ensure_dirs() -> None:
    """Create canonical output directory hierarchy if missing."""
    for path in [
        DATA_DIR,
        FIGURES_DIR,
        FIGURES_PYTHON_DIR,
        FIGURES_TESTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print("Project root:", PROJECT_ROOT)
    print("Data directory:", DATA_DIR)
    print("Figures directory:", FIGURES_DIR)
    print("(Call ensure_dirs() to create directories)")
