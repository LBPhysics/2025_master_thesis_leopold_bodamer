"""Simulation subpackage.

Structured components formerly contained in a single monolithic
`simulation_class.py` (now removed). Import directly from this
subpackage instead of the old legacy path.

Modules
-------
config              : SimulationConfig dataclass & validation
builders            : Core helper functions (interaction Hamiltonians)
liouvillian_paper   : Paper specific time–dependent Liouvillian builders
redfield            : Redfield tensor construction helpers
"""

from .config import SimulationConfig
from .builders import SimulationModuleOQS, H_int_
from .liouvillian_paper import matrix_ODE_paper
from .redfield import R_paper

__all__ = [
    "SimulationConfig",
    "SimulationModuleOQS",
    "H_int_",
    "matrix_ODE_paper",
    "R_paper",
]
