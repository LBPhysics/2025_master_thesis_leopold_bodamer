"""Simulation subpackage.

Modular simulation components. Legacy `simulation_class.py` has been
fully removed; import from here or concrete submodules directly.

Modules
-------
sim_config        : SimulationConfig dataclass & validation
builders          : Core helper functions (interaction Hamiltonians)
liouvillian_paper : Paper specific time–dependent Liouvillian builders
redfield          : Redfield tensor construction helpers
"""

from .sim_config import SimulationConfig
from .simulation_class import SimulationModuleOQS, H_int_
from .liouvillian_paper import matrix_ODE_paper
from .redfield_paper import redfield_paper

__all__ = [
    "SimulationConfig",
    "SimulationModuleOQS",
    "H_int_",
    "matrix_ODE_paper",
    "redfield_paper",
]
