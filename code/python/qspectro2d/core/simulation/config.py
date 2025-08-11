"""Simulation configuration data structures.

Separated from the former monolithic simulation_class module.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Tuple
import warnings

# Strict import of supported solvers (no fallback)
from qspectro2d.config.default_simulation_params import SUPPORTED_SOLVERS  # type: ignore


@dataclass
class SimulationConfig:
    """Primary configuration object for simulations.

    Focused immutable configuration object; no legacy compatibility paths.
    """

    ode_solver: str = "Paper_BR"
    rwa_sl: bool = True
    keep_track: str = "eigenstates"  # or "basis"

    dt: float = 0.1
    t_coh: float = 100.0
    t_wait: float = 0.0
    t_det_max: float = 100.0

    n_phases: int = 4
    n_freqs: int = 1

    max_workers: int = 1
    simulation_type: str = "1d"
    IFT_component: Tuple[int, int, int] = (1, -1, 0)

    def __post_init__(self) -> None:
        # Validate solver
        if self.ode_solver not in SUPPORTED_SOLVERS:
            raise ValueError(
                f"Invalid ode_solver '{self.ode_solver}'. Supported: {sorted(SUPPORTED_SOLVERS)}"
            )

        if self.ode_solver == "Paper_eqs" and not self.rwa_sl:
            warnings.warn(
                "rwa_sl forced True for Paper_eqs solver.",
                category=UserWarning,
                stacklevel=2,
            )
            self.rwa_sl = True

        if self.dt <= 0:
            raise ValueError("dt must be > 0")
        if self.t_coh < 0:
            raise ValueError("t_coh must be >= 0")
        if self.t_wait < 0:
            raise ValueError("t_wait must be >= 0")
        if self.t_det_max <= 0:
            raise ValueError("t_det_max must be > 0")
        if self.n_phases <= 0:
            raise ValueError("n_phases must be > 0")
        if self.n_freqs <= 0:
            raise ValueError("n_freqs must be > 0")

        # Derived total window (consistent with original logic)
        if self.t_coh < self.t_det_max:
            self.t_max = self.t_wait + 2 * self.t_det_max
        else:
            self.t_max = self.t_coh + self.t_wait + self.t_det_max

    @property
    def combinations(self) -> int:
        return self.n_phases * self.n_phases * self.n_freqs

    def summary(self) -> str:
        return (
            "SimulationConfig Summary:\n"
            "-------------------------------\n"
            f"{self.simulation_type} ELECTRONIC SPECTROSCOPY SIMULATION\n"
            "Time Parameters:\n"
            f"Coherence Time     : {self.t_coh} fs\n"
            f"Wait Time          : {self.t_wait} fs\n"
            f"Max Det. Time      : {self.t_det_max} fs\n\n"
            f"Total Time (t_max) : {self.t_max} fs\n"
            f"Time Step (dt)     : {self.dt} fs\n"
            "-------------------------------\n"
            f"Solver Type        : {self.ode_solver}\n"
            f"Use rwa_sl         : {self.rwa_sl}\n\n"
            "-------------------------------\n"
            f"Phase Cycles       : {self.n_phases}\n"
            f"Inhom. Points      : {self.n_freqs}\n"
            f"Total Combinations : {self.combinations}\n"
            f"Max Workers        : {self.max_workers}\n"
            "-------------------------------\n"
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SimulationConfig":  # type: ignore[name-defined]
        return cls(**data)

    def __str__(self) -> str:  # pragma: no cover simple repr
        return self.summary()
