"""Central project path definitions (single-file version, no package).

Import directly::

    from paths import DATA_DIR, FIGURES_PYTHON_DIR, ensure_dirs

Pure on import: no directories are created until :func:`ensure_dirs` is called.

Environment override:
    Set ``QSPECTRO2D_ROOT`` to the repository root to bypass automatic discovery.
"""

from __future__ import annotations

from pathlib import Path


def find_project_root() -> Path:
    """Locate the repository root directory.
    Walk parents of this file looking for a ``.git`` directory; if found, use it.
    """

    current_path = Path(__file__).resolve()

    for parent in current_path.parents:
        if (parent / ".git").is_dir():  # primary reliable marker
            return parent


PROJECT_ROOT = find_project_root()

# Base directory handles (pure objects, no side effects)
DATA_DIR = (PROJECT_ROOT / "data").resolve()
FIGURES_DIR = (PROJECT_ROOT / "figures").resolve()
PYTHON_CODE_DIR = PROJECT_ROOT / "code" / "python"
SCRIPTS_DIR = PYTHON_CODE_DIR / "scripts"
NOTEBOOKS_DIR = PYTHON_CODE_DIR / "notebooks"
LATEX_DIR = PROJECT_ROOT / "latex"

# Figure subdirectories (lazily created)
FIGURES_PYTHON_DIR = FIGURES_DIR / "figures_from_python"
FIGURES_TESTS_DIR = FIGURES_PYTHON_DIR / "tests"


def ensure_dirs() -> None:
    """Create canonical output directory hierarchy if missing.

    Safe to call many times (idempotent)."""
    for path in [
        DATA_DIR,
        FIGURES_DIR,
        FIGURES_PYTHON_DIR,
        FIGURES_TESTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "FIGURES_DIR",
    "FIGURES_PYTHON_DIR",
    "FIGURES_TESTS_DIR",
    "PYTHON_CODE_DIR",
    "SCRIPTS_DIR",
    "NOTEBOOKS_DIR",
    "LATEX_DIR",
    "ensure_dirs",
]


if __name__ == "__main__":  # simple probe
    print("Project root:", PROJECT_ROOT)
    print("Data directory:", DATA_DIR)
    print("Figures directory:", FIGURES_DIR)
    print("(Call ensure_dirs() to create directories)")
