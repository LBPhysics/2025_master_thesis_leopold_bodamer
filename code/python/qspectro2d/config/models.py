"""Structured configuration models (experimental).

Introduces typed, namespaced dataclass-based configuration objects that wrap
legacy module-level constants (single source of truth remains
``default_simulation_params``). This is an *additive* API; no existing imports
break.

Goals:
1. Semantic grouping (atomic, laser, bath, signal, solver, window)
2. Type hints & discoverability (``cfg.atomic.freqs_cm`` vs flat names)
3. Foundation for layered overrides (file/env/runtime) later
4. No import-time side effects; explicit validation

Usage:
    from qspectro2d.config import load_config
    cfg = load_config()
    cfg.validate()  # optional strict validation

Future (not implemented here): file/env override merge & aggregated reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from . import default_simulation_params as _defs


@dataclass(slots=True)
class AtomicConfig:
    """Atomic system parameters (frequencies in cm^-1)."""

    n_atoms: int = _defs.N_ATOMS
    freqs_cm: Sequence[float] = tuple(_defs.FREQS_CM)
    dip_moments: Sequence[float] = tuple(_defs.DIP_MOMENTS)
    at_coupling_cm: float = _defs.AT_COUPLING_CM
    delta_cm: float = _defs.DELTA_CM

    def validate(self) -> None:
        if len(self.freqs_cm) != self.n_atoms:
            raise ValueError("AtomicConfig: freqs_cm length != n_atoms")
        if len(self.dip_moments) != self.n_atoms:
            raise ValueError("AtomicConfig: dip_moments length != n_atoms")


@dataclass(slots=True)
class LaserConfig:
    """Laser / pulse parameters."""

    pulse_fwhm_fs: float = _defs.PULSE_FWHM
    base_amplitude: float = _defs.BASE_AMPLITUDE
    envelope_type: str = _defs.ENVELOPE_TYPE
    carrier_freq_cm: float = _defs.CARRIER_FREQ_CM

    def validate(self) -> None:
        if self.pulse_fwhm_fs <= 0:
            raise ValueError("LaserConfig: pulse_fwhm_fs must be positive")
        if self.base_amplitude <= 0:
            raise ValueError("LaserConfig: base_amplitude must be positive")


@dataclass(slots=True)
class BathConfig:
    """Bath / environment parameters."""

    bath_type: str = _defs.BATH_TYPE
    temperature: float = _defs.BATH_TEMP
    cutoff: float = _defs.BATH_CUTOFF
    coupling: float = _defs.BATH_COUPLING

    def validate(self) -> None:
        if self.temperature <= 0:
            raise ValueError("BathConfig: temperature must be positive")
        if self.cutoff <= 0:
            raise ValueError("BathConfig: cutoff must be positive")
        if self.coupling <= 0:
            raise ValueError("BathConfig: coupling must be positive")


@dataclass(slots=True)
class SignalProcessingConfig:
    """Phase cycling / detection parameters."""

    phase_cycling_phases: Sequence[float] = tuple(_defs.PHASE_CYCLING_PHASES)
    detection_phase: float = _defs.DETECTION_PHASE
    ift_component: Sequence[int] = tuple(_defs.IFT_COMPONENT)
    relative_e0s: Sequence[float] = tuple(_defs.RELATIVE_E0S)
    n_phases: int = _defs.N_PHASES

    def validate(self) -> None:
        if self.n_phases <= 0:
            raise ValueError("SignalProcessingConfig: n_phases must be positive")
        if len(self.relative_e0s) != 3:
            raise ValueError(
                "SignalProcessingConfig: relative_e0s must have 3 elements"
            )


@dataclass(slots=True)
class SolverConfig:
    """Solver selection & numerical tolerances."""

    solver: str = _defs.ODE_SOLVER
    solver_options: dict = field(default_factory=lambda: dict(_defs.SOLVER_OPTIONS))
    supported_solvers: Sequence[str] = tuple(_defs.SUPPORTED_SOLVERS)
    negative_eigval_threshold: float = _defs.NEGATIVE_EIGVAL_THRESHOLD
    trace_tolerance: float = _defs.TRACE_TOLERANCE

    def validate(self) -> None:
        if self.solver not in self.supported_solvers:
            raise ValueError(
                f"SolverConfig: solver '{self.solver}' not in {self.supported_solvers}"
            )


@dataclass(slots=True)
class SimulationWindowConfig:
    """Time / discretization controls for spectroscopy runs."""

    t_det_max_fs: float = _defs.T_DET_MAX
    dt_fs: float = _defs.DT
    batches: int = _defs.BATCHES
    n_freqs: int = _defs.N_FREQS
    rwa_sl: bool = _defs.RWA_SL

    def validate(self) -> None:
        if self.dt_fs <= 0:
            raise ValueError("SimulationWindowConfig: dt_fs must be positive")
        if self.t_det_max_fs <= 0:
            raise ValueError("SimulationWindowConfig: t_det_max_fs must be positive")
        if self.batches <= 0:
            raise ValueError("SimulationWindowConfig: batches must be positive")
        if self.n_freqs <= 0:
            raise ValueError("SimulationWindowConfig: n_freqs must be positive")


@dataclass(slots=True)
class MasterConfig:
    """Aggregate structured configuration (experimental API)."""

    atomic: AtomicConfig = field(default_factory=AtomicConfig)
    laser: LaserConfig = field(default_factory=LaserConfig)
    bath: BathConfig = field(default_factory=BathConfig)
    signal: SignalProcessingConfig = field(default_factory=SignalProcessingConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    window: SimulationWindowConfig = field(default_factory=SimulationWindowConfig)

    def validate(self) -> None:
        self.atomic.validate()
        self.laser.validate()
        self.bath.validate()
        self.signal.validate()
        self.solver.validate()
        self.window.validate()
        # leverage legacy global validation (includes RWA detuning warning)
        _defs.validate_defaults()

    @classmethod
    def from_defaults(cls) -> "MasterConfig":
        return cls()


__all__ = [
    "AtomicConfig",
    "LaserConfig",
    "BathConfig",
    "SignalProcessingConfig",
    "SolverConfig",
    "SimulationWindowConfig",
    "MasterConfig",
]
