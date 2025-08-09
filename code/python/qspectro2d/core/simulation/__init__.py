"""Simulation subpackage.

Provides structured components split out of the original monolithic
`simulation_class.py` for better maintainability.

Modules
-------
config              : SimulationConfig dataclass & enums
builders            : Core helper functions (interaction Hamiltonians)
liouvillian_paper   : Paper specific time–dependent Liouvillian builders
redfield            : Redfield tensor construction helpers

Backward compatibility:
The legacy `simulation_class.py` will import from here so external code using
`from qspectro2d.core.simulation_class import SimulationConfig` keeps working.
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
