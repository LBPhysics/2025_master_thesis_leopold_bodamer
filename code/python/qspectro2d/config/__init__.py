"""Configuration package (new structured API only).

All former flat simulation parameter constants have been removed in favor of a
single structured configuration object ``CONFIG`` (instance of ``MasterConfig``).

Access pattern examples:
    from qspectro2d.config import CONFIG
    n_atoms = CONFIG.atomic.n_atoms
    phases = CONFIG.signal.phase_cycling_phases
    solver_name = CONFIG.solver.solver

Call ``CONFIG.validate()`` if you need explicit validation.
"""

# Plotting utilities are no longer imported eagerly to avoid side effects during
# parallel worker initialization. Use `from plotstyle import init_style, ...`
# where plotting is actually needed.

from .paths import (
    # Paths (pure; call ensure_dirs() explicitly when needed)
    DATA_DIR,
    FIGURES_DIR,
    FIGURES_PYTHON_DIR,
    SCRIPTS_DIR,
    FIGURES_BATH_DIR,
    FIGURES_PULSES_DIR,
    FIGURES_TESTS_DIR,
    ensure_dirs,
)

from .default_simulation_params import validate_defaults  # kept for physics warnings
from qspectro2d.constants import HBAR, BOLTZMANN  # ensure re-export

# Structured config API
from .models import (
    MasterConfig,
    AtomicConfig,
    LaserConfig,
    BathConfig,
    SignalProcessingConfig,
    SolverConfig,
    SimulationWindowConfig,
)
from .loader import load_config

# Instantiate single global configuration object (defaults only unless caller provides a path)
CONFIG: MasterConfig = load_config()

# Optionally run validation at import (can be commented out if too strict)
try:
    CONFIG.validate()
except Exception as _e:
    # Defer hard failures to explicit user validation; just emit message.
    print(f"[qspectro2d.config] Validation warning during import: {_e}")

# Export all important symbols for import *
__all__ = [
    # constants
    # (plotting symbols intentionally not re-exported; import from plotstyle)
    # paths
    "DATA_DIR",
    "FIGURES_DIR",
    "FIGURES_PYTHON_DIR",
    "SCRIPTS_DIR",
    "FIGURES_BATH_DIR",
    "FIGURES_PULSES_DIR",
    "FIGURES_TESTS_DIR",
    "ensure_dirs",
    # default simulation parameters
    "HBAR",
    "BOLTZMANN",
    # validation
    "validate_defaults",
    # structured config API
    "MasterConfig",
    "AtomicConfig",
    "LaserConfig",
    "BathConfig",
    "SignalProcessingConfig",
    "SolverConfig",
    "SimulationWindowConfig",
    "load_config",
    "CONFIG",
]
