"""Minimal configuration loader (defaults + optional explicit file).

This simplified loader returns a ``MasterConfig`` constructed from:
    1. Built-in defaults (dataclass defaults from ``default_simulation_params``)
    2. Optional explicit config file passed via ``path=`` (YAML/TOML/JSON)

Usage examples::

    from qspectro2d.config.loader import load_config

    # Defaults only
    cfg = load_config()

    # Merge explicit file over defaults
    cfg = load_config(path="config.yaml")

Call ``cfg.validate()`` explicitly for strict validation if desired.
"""

from __future__ import annotations
from .models import MasterConfig
from pathlib import Path
import json
from typing import Any, Mapping, MutableMapping

try:  # YAML optional
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None  # fallback; error only if YAML actually requested

try:  # TOML via stdlib (3.11+)
    import tomllib  # type: ignore
except Exception:  # pragma: no cover
    tomllib = None


def _deep_merge(base: MutableMapping[str, Any], upd: Mapping[str, Any]) -> None:
    """Deep in-place merge of ``upd`` into ``base`` (section-wise)."""
    for k, v in upd.items():
        if isinstance(v, Mapping) and isinstance(base.get(k), Mapping):
            _deep_merge(base[k], v)  # type: ignore[index]
        else:
            base[k] = v  # type: ignore[index]


def _as_dict(cfg: MasterConfig) -> dict:
    """Convert MasterConfig dataclass tree to nested dict (shallow for sequences)."""
    return {
        "atomic": dict(
            n_atoms=cfg.atomic.n_atoms,
            n_chains=cfg.atomic.n_chains,
            frequencies_cm=list(cfg.atomic.frequencies_cm),
            dip_moments=list(cfg.atomic.dip_moments),
            coupling_cm=cfg.atomic.coupling_cm,
            delta_cm=cfg.atomic.delta_cm,
            max_excitation=cfg.atomic.max_excitation,
            n_freqs=cfg.atomic.n_freqs,
        ),
        "laser": dict(
            pulse_fwhm_fs=cfg.laser.pulse_fwhm_fs,
            base_amplitude=cfg.laser.base_amplitude,
            envelope_type=cfg.laser.envelope_type,
            carrier_freq_cm=cfg.laser.carrier_freq_cm,
            rwa_sl=cfg.laser.rwa_sl,
        ),
        "bath": dict(
            bath_type=cfg.bath.bath_type,
            temperature=cfg.bath.temperature,
            cutoff=cfg.bath.cutoff,
            coupling=cfg.bath.coupling,
        ),
        # signal intentionally omitted -> always use defaults
        "solver": dict(
            solver=cfg.solver.solver,
            solver_options=dict(cfg.solver.solver_options),
            supported_solvers=list(cfg.solver.supported_solvers),
            negative_eigval_threshold=cfg.solver.negative_eigval_threshold,
            trace_tolerance=cfg.solver.trace_tolerance,
        ),
        "window": dict(
            t_det_max=cfg.window.t_det_max,
            dt=cfg.window.dt,
            n_batches=cfg.window.n_batches,
            t_wait=cfg.window.t_wait,     # new
            t_coh=cfg.window.t_coh,       # new
            batch_idx=cfg.window.batch_idx,  # new
        ),
    }


def _dict_to_config(cfg_dict: Mapping[str, Any]) -> MasterConfig:
    """Instantiate MasterConfig from nested dict (expects same shape as _as_dict)."""
    base = MasterConfig.from_defaults()
    # atomic
    a = cfg_dict.get("atomic", {})
    base.atomic.n_atoms = a.get("n_atoms", base.atomic.n_atoms)
    base.atomic.n_chains = a.get("n_chains", base.atomic.n_chains)
    base.atomic.frequencies_cm = a.get("frequencies_cm", base.atomic.frequencies_cm)
    base.atomic.dip_moments = a.get("dip_moments", base.atomic.dip_moments)
    base.atomic.coupling_cm = a.get("coupling_cm", base.atomic.coupling_cm)
    base.atomic.delta_cm = a.get("delta_cm", base.atomic.delta_cm)
    base.atomic.max_excitation = a.get("max_excitation", base.atomic.max_excitation)
    base.atomic.n_freqs = a.get("n_freqs", base.atomic.n_freqs)
    # laser
    l = cfg_dict.get("laser", {})
    base.laser.pulse_fwhm_fs = l.get("pulse_fwhm_fs", base.laser.pulse_fwhm_fs)
    base.laser.base_amplitude = l.get("base_amplitude", base.laser.base_amplitude)
    base.laser.envelope_type = l.get("envelope_type", base.laser.envelope_type)
    base.laser.carrier_freq_cm = l.get("carrier_freq_cm", base.laser.carrier_freq_cm)
    base.laser.rwa_sl = l.get("rwa_sl", base.laser.rwa_sl)
    # bath
    b = cfg_dict.get("bath", {})
    base.bath.bath_type = b.get("bath_type", base.bath.bath_type)
    base.bath.temperature = b.get("temperature", base.bath.temperature)
    base.bath.cutoff = b.get("cutoff", base.bath.cutoff)
    base.bath.coupling = b.get("coupling", base.bath.coupling)
    # signal: omitted -> keep defaults
    # solver
    so = cfg_dict.get("solver", {})
    base.solver.solver = so.get("solver", base.solver.solver)
    base.solver.solver_options.update(so.get("solver_options", {}))
    base.solver.supported_solvers = so.get(
        "supported_solvers", base.solver.supported_solvers
    )
    base.solver.negative_eigval_threshold = so.get(
        "negative_eigval_threshold", base.solver.negative_eigval_threshold
    )
    base.solver.trace_tolerance = so.get("trace_tolerance", base.solver.trace_tolerance)
    # window
    w = cfg_dict.get("window", {})
    base.window.t_det_max = w.get("t_det_max", base.window.t_det_max)
    base.window.dt = w.get("dt", base.window.dt)
    base.window.n_batches = w.get("n_batches", base.window.n_batches)
    base.window.t_wait = w.get("t_wait", base.window.t_wait)       # new
    base.window.t_coh = w.get("t_coh", base.window.t_coh)          # new
    base.window.batch_idx = w.get("batch_idx", base.window.batch_idx)  # new
    return base


def _load_file(path: Path) -> dict:
    if not path.exists():  # pragma: no cover - defensive
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix in {".yml", ".yaml"}:
        if yaml is None:
            raise RuntimeError(
                "PyYAML not installed. Add 'PyYAML' to requirements to load YAML configs."
            )
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    if suffix == ".toml":
        if tomllib is None:
            raise RuntimeError("tomllib unavailable (Python <3.11?) cannot read TOML")
        with path.open("rb") as fh:
            return tomllib.load(fh) or {}
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    raise ValueError(f"Unsupported config file extension: {suffix}")


def load_config(path: str | Path | None = None) -> MasterConfig:
    """Load configuration from defaults and optionally merge an explicit file.

    Parameters
    ----------
    path:
        Optional explicit config file (YAML / TOML / JSON). If provided, values
        from the file are deep-merged over defaults. If None, pure defaults are
        returned.
    """

    base_cfg = MasterConfig.from_defaults()
    cfg_dict = _as_dict(base_cfg)

    if path is not None:
        _deep_merge(cfg_dict, _load_file(Path(path)))

    return _dict_to_config(cfg_dict)


__all__ = ["load_config"]
