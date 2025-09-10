"""Simplified configuration → simulation factory.

Purpose:
    Load a YAML config (or fall back to defaults) and directly build a
    `SimulationModuleOQS` instance with the core objects:
        - AtomicSystem
        - LaserPulseSequence
        - BosonicEnvironment (qutip)
        - SimulationConfig

Usage:
    from qspectro2d.config.create_sim_obj import load_simulation
    sim = load_simulation("scripts/config.yaml")  # or None for defaults

YAML Schema (minimal example):
    atomic:
      n_atoms: 2
      frequencies_cm: [15900.0, 16000.0]
      dip_moments: [1.0, 1.0]
      coupling_cm: 0.0
      delta_cm: 0.0
      max_excitation: 2
      n_freqs: 1
    laser:
      pulse_fwhm_fs: 5.0
      base_amplitude: 0.5
      envelope_type: gaussian
      carrier_freq_cm: 16000.0
      rwa_sl: true
    bath:
      bath_type: ohmic   # currently informational only
      temperature: 0.001
      cutoff: 100.0
      coupling: 0.0001
    solver:
      solver: BR
    window:
      dt: 0.1
      t_coh: 100.0
      t_wait: 0.0
      t_det_max: 200.0

Optional pulse control:
    pulses:
      # Delays BETWEEN pulses (n delays => n+1 pulses)
      delays: [t12, t23]
      relative_e0s: [1.0, 1.0, 0.1]
      phases: [0.0, 0.0, 0.0]

If pulses section omitted, a 3‑pulse sequence is synthesized using:
    delays = [window.t_coh, window.t_wait]
    relative_e0s = [1.0, 1.0, 0.1]
    phases = [0.0, 0.0, 0.0]

Notes:
    - This bypasses the hierarchical dataclass layer in `models.py`.
    - Keeps validation light; relies on existing class constructors raising errors.
    - Extra / unknown keys are ignored.
"""

from __future__ import annotations

import os
import numpy as np
import psutil
from pathlib import Path
from typing import Any, Mapping, Optional, TYPE_CHECKING
from qutip import OhmicEnvironment
import yaml

if TYPE_CHECKING:
    from qspectro2d.core.simulation.simulation_class import SimulationModuleOQS
    from qspectro2d.core.simulation.sim_config import SimulationConfig

from qspectro2d.config import default_simulation_params as dflt

__all__ = ["load_simulation", "create_base_sim_oqs", "get_max_workers"]


# =============================
# HELPERS
# =============================


def _read_yaml(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    if yaml is None:
        raise RuntimeError("PyYAML not installed. Add 'PyYAML' to requirements.")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, Mapping):
        raise TypeError("Top-level YAML must be a mapping/dict")
    return data


def _get_section(cfg: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    sec = cfg.get(name, {})
    return sec if isinstance(sec, Mapping) else {}


def load_simulation(
    path: Optional[str | Path] = None, validate: bool = True
) -> SimulationModuleOQS:
    """Create a `SimulationModuleOQS` directly from a YAML file or defaults.

    Parameters
    ----------
    path: str | Path | None
        YAML configuration file. If None, defaults from
        `default_simulation_params` are used.
    validate: bool
        If True (default) run physics validation via `default_simulation_params.validate`.
    """
    # Import here to avoid circular import
    from qspectro2d.core.atomic_system.system_class import AtomicSystem
    from qspectro2d.core.laser_system.laser_class import LaserPulseSequence
    from qspectro2d.core.simulation.simulation_class import SimulationModuleOQS
    from qspectro2d.core.simulation.sim_config import SimulationConfig

    # -----------------
    # LOAD / FALLBACK
    # -----------------
    if path is None:
        cfg_root: Mapping[str, Any] = {}
    else:
        cfg_root = _read_yaml(Path(path))

    # -----------------
    # ATOMIC SYSTEM
    # -----------------
    atomic_cfg = _get_section(cfg_root, "atomic")
    n_atoms = int(atomic_cfg.get("n_atoms", dflt.N_ATOMS))
    n_chains = int(atomic_cfg.get("n_chains", dflt.N_CHAINS))
    freqs_cm = list(atomic_cfg.get("frequencies_cm", dflt.FREQUENCIES_CM))
    dip_moments = list(atomic_cfg.get("dip_moments", dflt.DIP_MOMENTS))
    coupling_cm = float(atomic_cfg.get("coupling_cm", dflt.COUPLING_CM))
    delta_cm = float(atomic_cfg.get("delta_cm", dflt.DELTA_CM))
    max_excitation = int(atomic_cfg.get("max_excitation", dflt.MAX_EXCITATION))
    n_freqs = int(atomic_cfg.get("n_freqs", dflt.N_FREQS))

    atomic_system = AtomicSystem(
        n_atoms=n_atoms,
        n_chains=n_chains,
        frequencies_cm=freqs_cm,
        dip_moments=dip_moments,
        coupling_cm=coupling_cm,
        delta_cm=delta_cm,
        max_excitation=max_excitation,
    )

    # -----------------
    # LASER / PULSES
    # -----------------
    laser_cfg = _get_section(cfg_root, "laser")
    pulse_fwhm = float(laser_cfg.get("pulse_fwhm_fs", dflt.PULSE_FWHM))
    base_amp = float(laser_cfg.get("base_amplitude", dflt.BASE_AMPLITUDE))
    envelope = str(laser_cfg.get("envelope_type", dflt.ENVELOPE_TYPE))
    carrier_cm = float(laser_cfg.get("carrier_freq_cm", dflt.CARRIER_FREQ_CM))
    rwa_sl = bool(laser_cfg.get("rwa_sl", dflt.RWA_SL))

    pulses_cfg = _get_section(cfg_root, "pulses")
    relative_e0s = list(pulses_cfg.get("relative_e0s", dflt.RELATIVE_E0S))

    # synthesize 3-pulse sequence from time window (typical 2D: coherence, wait)
    window_cfg = _get_section(cfg_root, "window")
    t_coh = float(window_cfg.get("t_coh", dflt.T_COH))
    t_wait = float(window_cfg.get("t_wait", dflt.T_WAIT))
    delays = [t_coh, t_wait]  # -> 3 pulses
    phases = [0.0, 0.0, 0.0]

    laser_sequence = LaserPulseSequence.from_delays(
        delays=delays,
        base_amplitude=base_amp,
        pulse_fwhm=pulse_fwhm,
        carrier_freq_cm=carrier_cm,
        envelope_type=envelope,
        relative_E0s=relative_e0s,
        phases=phases,
    )

    # -----------------
    # BATH (qutip BosonicEnvironment stub + tag extras)
    # -----------------
    bath_cfg = _get_section(cfg_root, "bath")
    temperature = float(bath_cfg.get("temperature", dflt.BATH_TEMP))
    cutoff = float(bath_cfg.get("cutoff", dflt.BATH_CUTOFF))
    coupling = float(bath_cfg.get("coupling", dflt.BATH_COUPLING))
    bath_type = str(bath_cfg.get("bath_type", dflt.BATH_TYPE))

    # TODO extend to BsosonicEnvironment
    bath_env = OhmicEnvironment(
        T=temperature,
        alpha=coupling / cutoff,  # NOTE this is now exactly the paper implementation
        wc=cutoff,
        s=1.0,
        tag=bath_type,
    )
    # -----------------
    # SIMULATION CONFIG (flat)
    # -----------------
    window_cfg = _get_section(cfg_root, "window")
    solver_cfg = _get_section(cfg_root, "solver")

    dt = float(window_cfg.get("dt", dflt.DT))
    t_coh = float(window_cfg.get("t_coh", 0.0))
    t_wait = float(window_cfg.get("t_wait", 0.0))
    t_det_max = float(window_cfg.get("t_det_max", dflt.T_DET_MAX))
    n_phases = int(solver_cfg.get("n_phases", dflt.N_PHASES))  # allow override
    ode_solver = str(solver_cfg.get("solver", dflt.ODE_SOLVER))
    signal_types = list(solver_cfg.get("signal_types", dflt.SIGNAL_TYPES))

    # -----------------
    # VALIDATION (physics-level) BEFORE FINAL ASSEMBLY
    # -----------------
    if validate:
        params = {
            "solver": ode_solver,
            "bath_type": bath_type,
            "frequencies_cm": freqs_cm,
            "n_atoms": n_atoms,
            "dip_moments": dip_moments,
            "temperature": temperature,
            "cutoff": cutoff,
            "coupling": coupling,
            "n_phases": n_phases,
            "max_excitation": max_excitation,
            "n_chains": n_chains,
            "relative_e0s": relative_e0s,
            "rwa_sl": rwa_sl,
            "carrier_freq_cm": carrier_cm,
            "signal_types": signal_types,
        }
        dflt.validate(params)

    sim_config = SimulationConfig(
        ode_solver=ode_solver,
        rwa_sl=rwa_sl,
        dt=dt,
        t_coh=t_coh,
        t_wait=t_wait,
        t_det_max=t_det_max,
        n_phases=n_phases,
        n_freqs=n_freqs,
        signal_types=signal_types,
    )

    # -----------------
    # ASSEMBLE
    # -----------------
    simulation = SimulationModuleOQS(
        simulation_config=sim_config,
        system=atomic_system,
        laser=laser_sequence,
        bath=bath_env,
    )

    return simulation


def create_base_sim_oqs(
    args,
    config_path: str | None = None,
) -> tuple[SimulationModuleOQS, float]:
    """Create base simulation instance and perform solver validation once.

    Parameters:
        args: Parsed command line arguments (may contain overrides)
        config_path: Optional path to YAML config (None -> defaults)

    Returns:
        tuple: (SimulationModuleOQS instance, time_cut from solver validation)
    """
    # -----------------
    # LOAD BASE SIMULATION (validated physics params inside loader)
    # -----------------
    sim = load_simulation(config_path, validate=True)

    print("🔧 Base simulation created from config.")

    # -----------------
    # APPLY CLI OVERRIDES (times / dt / solver) IF PROVIDED
    # -----------------
    cfg = sim.simulation_config
    override = False

    t_coh = getattr(args, "t_coh", None)
    t_wait = getattr(args, "t_wait", None)
    t_det_max = getattr(args, "t_det_max", None)
    dt = getattr(args, "dt", None)
    solver = getattr(args, "ode_solver", None)

    if t_coh is not None:
        override = True
    if t_wait is not None:
        override = True
    if t_det_max is not None:
        override = True
    if dt is not None:
        override = True
    if solver is not None:
        override = True

    if override:
        new_cfg = SimulationConfig(
            ode_solver=solver if solver is not None else cfg.ode_solver,
            rwa_sl=cfg.rwa_sl,
            dt=dt if dt is not None else cfg.dt,
            t_coh=t_coh if t_coh is not None else cfg.t_coh,
            t_wait=t_wait if t_wait is not None else cfg.t_wait,
            t_det_max=t_det_max if t_det_max is not None else cfg.t_det_max,
            n_phases=cfg.n_phases,
            n_freqs=cfg.n_freqs,
            signal_types=cfg.signal_types,
        )
        # Re-wrap in fresh SimulationModuleOQS so __post_init__ recalculates evo objects
        sim = SimulationModuleOQS(
            simulation_config=new_cfg,
            system=sim.system,
            laser=sim.laser,
            bath=sim.bath,
        )
        print("⚙️  Applied CLI overrides to simulation configuration.")

    # -----------------
    # SOLVER VALIDATION
    # -----------------
    time_cut = -np.inf
    t_max = sim.simulation_config.t_max
    print("🔍 Validating solver...")
    try:
        from qspectro2d.spectroscopy.calculations import check_the_solver

        _, time_cut = check_the_solver(sim)
        print("#" * 60)
        print(
            f"✅ Solver validation worked: Evolution becomes unphysical at "
            f"({time_cut / t_max:.2f} × t_max)"
        )
    except Exception as e:  # pragma: no cover
        print(f"⚠️  WARNING: Solver validation failed: {e}")

    if time_cut < t_max:
        print(
            f"⚠️  WARNING: Time cut {time_cut} is less than the last time point "
            f"{t_max}. This may affect the simulation results.",
            flush=True,
        )

    return sim, time_cut


def get_max_workers() -> int:
    """Get the maximum number of workers for parallel processing."""
    # Use SLURM environment variable if available, otherwise detect automatically
    try:
        slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 0))
    except ValueError:
        slurm_cpus = 0

    local_cpus = psutil.cpu_count(logical=True) or 1
    return slurm_cpus if slurm_cpus > 0 else local_cpus
