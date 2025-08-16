"""Project path definitions (no side effects).

This module exposes lazily-resolved paths relative to the detected project root.
It intentionally performs **no filesystem mutations at import time** (no directory
creation) to keep imports pure and to avoid surprising side effects in contexts like
unit tests, packaging, or documentation builds.

Call :func:`ensure_dirs` explicitly in entry‑point scripts (e.g. simulation runners,
CLI tools, notebooks) before writing data / figures.

Rationale for removing side effects:
    * Deterministic imports (import does not touch disk)
    * Easier mocking of filesystem in tests
    * Clearer responsibility boundary (callers decide when to create output dirs)
    * Avoids accidental creation of large directory trees during static analysis

Example::

    from qspectro2d.config.paths import DATA_DIR, FIGURES_DIR, ensure_dirs
    ensure_dirs()  # creates the base output directory structure if missing
    (DATA_DIR / "1d_spectroscopy").mkdir(exist_ok=True)

Future extension: accept a ``base=Path`` override or environment variable to allow
redirecting outputs (e.g. for scratch / temporary runs) without modifying code.
"""

from pathlib import Path
import os


def find_project_root() -> Path:
    """Robustly locate the repository root.

    Strategy (in order):
    1. Environment variable override: ``QSPECTRO2D_ROOT`` if it points to an existing dir.
    2. Ascend parents from this file looking for a *marker set* that characterises the repo
       (``.git`` directory OR simultaneous presence of ``environment.yml`` and ``requirements.txt``).
    3. Fallback: first parent whose name contains ``master_thesis`` (case‑insensitive) to
       stay compatible with older naming schemes (e.g. "2025_master_thesis_*...").
    4. If still not found raise a descriptive error with guidance.
    """

    # 1. Explicit override
    override = os.environ.get("QSPECTRO2D_ROOT")
    if override:
        p = Path(override).expanduser().resolve()
        if p.is_dir():
            return p

    current_path = Path(__file__).resolve()

    marker_files = {"environment.yml", "requirements.txt"}

    for parent in current_path.parents:
        # 2a. .git directory is a strong indicator
        if (parent / ".git").is_dir():
            return parent
        # 2b. Both marker files present
        if all((parent / m).exists() for m in marker_files):
            return parent

    # 3. Name heuristic (substring master_thesis)
    for parent in current_path.parents:
        if "master_thesis" in parent.name.lower():
            return parent

    # 4. Failure – construct guidance
    raise RuntimeError(
        "Could not determine project root. Set environment variable 'QSPECTRO2D_ROOT' "
        "to the repository path or ensure a .git directory / marker files (environment.yml, requirements.txt) exist. "
        f"Start path: {current_path}"
    )


PROJECT_ROOT = find_project_root()

#########################
# Base directories (pure)
#########################
DATA_DIR = (PROJECT_ROOT / "data").resolve()
FIGURES_DIR = (PROJECT_ROOT / "figures").resolve()

# Additional useful paths (NOT used)
PYTHON_CODE_DIR = PROJECT_ROOT / "code" / "python"
SCRIPTS_DIR = PYTHON_CODE_DIR / "scripts"
NOTEBOOKS_DIR = PYTHON_CODE_DIR / "notebooks"
LATEX_DIR = PROJECT_ROOT / "latex"

##############################
# Figure subdirectories (lazy)
##############################
FIGURES_PYTHON_DIR = FIGURES_DIR / "figures_from_python"
FIGURES_TESTS_DIR = FIGURES_PYTHON_DIR / "tests"  # For test figures


def ensure_dirs() -> None:
    """Create the canonical output directory hierarchy if missing.

    Safe to call multiple times. Kept separate from import to avoid side effects.
    """

    for path in [
        DATA_DIR,
        FIGURES_DIR,
        FIGURES_PYTHON_DIR,
        FIGURES_TESTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


# Print paths for debugging (only when directly executed)
if __name__ == "__main__":  # manual debug helper
    print("Project root:", PROJECT_ROOT)
    print("Data directory:", DATA_DIR)
    print("Figures directory:", FIGURES_DIR)
    print("(Call ensure_dirs() to create directories)")
