"""Structured configuration models (experimental).

Introduces typed, namespaced dataclass-based configuration objects that wrap
legacy module-level constants (single source of truth remains
``default_simulation_params``). This is an *additive* API; no existing imports
break.

Goals:
1. Semantic grouping (atomic, laser, bath, signal, solver, window)
2. Type hints & discoverability (``cfg.atomic.frequencies_cm`` vs flat names)
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
from typing import Sequence, Mapping, Any

from .default_simulation_params import (
    N_ATOMS,
    N_CHAINS,
    FREQUENCIES_CM,
    DIP_MOMENTS,
    COUPLING_CM,
    DELTA_CM,
    MAX_EXCITATION,
    N_FREQS,
    PULSE_FWHM,
    BASE_AMPLITUDE,
    ENVELOPE_TYPE,
    CARRIER_FREQ_CM,
    RWA_SL,
    BATH_TYPE,
    BATH_TEMP,
    BATH_CUTOFF,
    BATH_COUPLING,
    PHASE_CYCLING_PHASES,
    DETECTION_PHASE,
    IFT_COMPONENT,
    RELATIVE_E0S,
    N_PHASES,
    ODE_SOLVER,
    SOLVER_OPTIONS,
    SUPPORTED_SOLVERS,
    NEGATIVE_EIGVAL_THRESHOLD,
    TRACE_TOLERANCE,
    T_DET_MAX,
    DT,
    N_BATCHES,
    validate as validate_defaults_fn,
)


@dataclass(slots=True)
class AtomicConfig:
    """Atomic system parameters (frequencies in cm^-1)."""

    n_atoms: int = N_ATOMS
    # Optional geometry spec for multi-atom systems (see AtomicSystem); if n_atoms>2 and None -> linear chain
    n_chains: int = N_CHAINS
    frequencies_cm: Sequence[float] = field(
        default_factory=lambda: list(FREQUENCIES_CM)
    )
    dip_moments: Sequence[float] = field(default_factory=lambda: list(DIP_MOMENTS))
    coupling_cm: float = COUPLING_CM
    delta_cm: float = DELTA_CM
    # Excitation manifold truncation (1: ground+single, 2: add double manifold)
    max_excitation: int = MAX_EXCITATION
    n_freqs: int = N_FREQS  # moved from window

    def validate(self) -> None:
        if len(self.frequencies_cm) != self.n_atoms:
            raise ValueError("AtomicConfig: frequencies_cm length != n_atoms")
        if len(self.dip_moments) != self.n_atoms:
            raise ValueError("AtomicConfig: dip_moments length != n_atoms")
        if self.max_excitation not in (1, 2):
            raise ValueError("AtomicConfig: max_excitation must be 1 or 2")
        # Geometry divisibility checks (mirrors default validation)
        if self.n_chains is not None and self.n_atoms > 2:
            if self.n_chains < 1:
                raise ValueError("AtomicConfig: n_chains must be >=1 when specified")
            if self.n_atoms % self.n_chains != 0:
                raise ValueError(
                    f"AtomicConfig: n_chains ({self.n_chains}) does not divide n_atoms ({self.n_atoms})"
                )


@dataclass(slots=True)
class LaserConfig:
    """Laser / pulse parameters."""

    pulse_fwhm_fs: float = PULSE_FWHM
    base_amplitude: float = BASE_AMPLITUDE
    envelope_type: str = ENVELOPE_TYPE
    carrier_freq_cm: float = CARRIER_FREQ_CM
    rwa_sl: bool = RWA_SL  # moved from window

    def validate(self) -> None:
        if self.pulse_fwhm_fs <= 0:
            raise ValueError("LaserConfig: pulse_fwhm_fs must be positive")
        if self.base_amplitude <= 0:
            raise ValueError("LaserConfig: base_amplitude must be positive")


@dataclass(slots=True)
class BathConfig:
    """Bath / environment parameters."""

    bath_type: str = BATH_TYPE
    temperature: float = BATH_TEMP
    cutoff: float = BATH_CUTOFF
    coupling: float = BATH_COUPLING

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

    phase_cycling_phases: Sequence[float] = field(
        default_factory=lambda: list(PHASE_CYCLING_PHASES)
    )
    detection_phase: float = DETECTION_PHASE
    ift_component: Sequence[int] = field(default_factory=lambda: list(IFT_COMPONENT))
    relative_e0s: Sequence[float] = field(default_factory=lambda: list(RELATIVE_E0S))
    n_phases: int = N_PHASES

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

    solver: str = ODE_SOLVER
    solver_options: Mapping[str, Any] = field(
        default_factory=lambda: dict(SOLVER_OPTIONS)
    )
    supported_solvers: Sequence[str] = field(
        default_factory=lambda: list(SUPPORTED_SOLVERS)
    )
    negative_eigval_threshold: float = NEGATIVE_EIGVAL_THRESHOLD
    trace_tolerance: float = TRACE_TOLERANCE

    def validate(self) -> None:
        if self.solver not in self.supported_solvers:
            raise ValueError(
                f"SolverConfig: solver '{self.solver}' not in {self.supported_solvers}"
            )


@dataclass(slots=True)
class SimulationWindowConfig:
    """Time / discretization controls for spectroscopy runs."""

    t_det_max: float = T_DET_MAX
    dt: float = DT
    n_batches: int = N_BATCHES
    # new: allow YAML to define runtime defaults that CLI can override
    t_wait: float = 0.0
    t_coh: float = 0.0
    batch_idx: int = 0

    def validate(self) -> None:
        if self.dt <= 0:
            raise ValueError("SimulationWindowConfig: dt must be positive")
        if self.t_det_max <= 0:
            raise ValueError("SimulationWindowConfig: t_det_max must be positive")
        if self.n_batches <= 0:
            raise ValueError("SimulationWindowConfig: n_batches must be positive")
        if self.batch_idx < 0:
            raise ValueError("SimulationWindowConfig: batch_idx must be non-negative")


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
        # Keep section-specific checks (unique to these models)
        self.laser.validate()
        self.window.validate()

        # Unified cross-section validation using defaults module
        params = {
            "solver": self.solver.solver,
            "bath_type": self.bath.bath_type,
            "frequencies_cm": list(self.atomic.frequencies_cm),
            "n_atoms": self.atomic.n_atoms,
            "dip_moments": list(self.atomic.dip_moments),
            "temperature": self.bath.temperature,
            "cutoff": self.bath.cutoff,
            "coupling": self.bath.coupling,
            "n_phases": self.signal.n_phases,
            "max_excitation": self.atomic.max_excitation,
            "n_chains": self.atomic.n_chains,
            "relative_e0s": list(self.signal.relative_e0s),
            "rwa_sl": self.laser.rwa_sl,  # moved from window
            "carrier_freq_cm": self.laser.carrier_freq_cm,
        }
        validate_defaults_fn(params)

    @classmethod
    def from_defaults(cls) -> "MasterConfig":
        return cls(
            atomic=AtomicConfig(
                n_atoms=N_ATOMS,
                n_chains=N_CHAINS,
                frequencies_cm=list(FREQUENCIES_CM),
                dip_moments=list(DIP_MOMENTS),
                coupling_cm=COUPLING_CM,
                delta_cm=DELTA_CM,
                max_excitation=MAX_EXCITATION,
                n_freqs=N_FREQS,  # new
            ),
            laser=LaserConfig(
                pulse_fwhm_fs=PULSE_FWHM,
                base_amplitude=BASE_AMPLITUDE,
                envelope_type=ENVELOPE_TYPE,
                carrier_freq_cm=CARRIER_FREQ_CM,
                rwa_sl=RWA_SL,  # new
            ),
            bath=BathConfig(),
            signal=SignalProcessingConfig(),
            solver=SolverConfig(),
            window=SimulationWindowConfig(),
        )


__all__ = [
    "AtomicConfig",
    "LaserConfig",
    "BathConfig",
    "SignalProcessingConfig",
    "SolverConfig",
    "SimulationWindowConfig",
    "MasterConfig",
]
