"""Configuration loader with layered overrides.

Provides ``load_config`` returning a ``MasterConfig`` constructed from:
    1. Built-in defaults (dataclass defaults sourced from ``default_simulation_params``)
    2. Optional project config file (``qspectro2d.config.{yml,toml}``) at repo root
    3. Optional user-specified file passed via ``path=`` (YAML or TOML)
    4. Optional environment variables (``QSPEC_<SECTION>_<FIELD>=value``)
    5. Optional runtime overrides dict supplied via ``overrides=`` argument

Precedence increases downward (later layers override earlier). Only provided
keys are merged; unspecified keys inherit earlier-layer values.

Environment variable mapping rules:
    * Prefix ``QSPEC_`` (configurable via ``env_prefix`` argument)
    * Section & field names joined by underscore, case-insensitive
      Example: QSPEC_WINDOW_DT_FS=2.0  ->  CONFIG.window.dt_fs = 2.0
    * Values are parsed with a small heuristic: int -> float -> bool -> literal

TOML support is basic and uses the stdlib ``tomllib`` (Python 3.11+). YAML
requires ``PyYAML``; if missing and a YAML file is requested an informative
error is raised.

Usage examples::

    from qspectro2d.config.loader import load_config

    # Pure defaults
    cfg = load_config()

    # With user file + env layer
    cfg = load_config(path="run.yml", use_env=True)

    # Inline runtime overrides
    cfg = load_config(overrides={"window": {"dt_fs": 1.0}})

Call ``cfg.validate()`` explicitly for strict validation.
"""

from __future__ import annotations

from .models import MasterConfig
from pathlib import Path
import os
import json
from copy import deepcopy
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
            freqs_cm=list(cfg.atomic.freqs_cm),
            dip_moments=list(cfg.atomic.dip_moments),
            at_coupling_cm=cfg.atomic.at_coupling_cm,
            delta_cm=cfg.atomic.delta_cm,
        ),
        "laser": dict(
            pulse_fwhm_fs=cfg.laser.pulse_fwhm_fs,
            base_amplitude=cfg.laser.base_amplitude,
            envelope_type=cfg.laser.envelope_type,
            carrier_freq_cm=cfg.laser.carrier_freq_cm,
        ),
        "bath": dict(
            bath_type=cfg.bath.bath_type,
            temperature=cfg.bath.temperature,
            cutoff=cfg.bath.cutoff,
            coupling=cfg.bath.coupling,
        ),
        "signal": dict(
            phase_cycling_phases=list(cfg.signal.phase_cycling_phases),
            detection_phase=cfg.signal.detection_phase,
            ift_component=list(cfg.signal.ift_component),
            relative_e0s=list(cfg.signal.relative_e0s),
            n_phases=cfg.signal.n_phases,
        ),
        "solver": dict(
            solver=cfg.solver.solver,
            solver_options=dict(cfg.solver.solver_options),
            supported_solvers=list(cfg.solver.supported_solvers),
            negative_eigval_threshold=cfg.solver.negative_eigval_threshold,
            trace_tolerance=cfg.solver.trace_tolerance,
        ),
        "window": dict(
            t_det_max_fs=cfg.window.t_det_max_fs,
            dt_fs=cfg.window.dt_fs,
            batches=cfg.window.batches,
            n_freqs=cfg.window.n_freqs,
            rwa_sl=cfg.window.rwa_sl,
        ),
    }


def _apply_env(cfg_dict: MutableMapping[str, Any], prefix: str) -> None:
    plen = len(prefix)
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        tail = key[plen:]
        parts = tail.lower().split("_")
        if len(parts) < 2:
            continue
        section, field = parts[0], "_".join(parts[1:])
        sect = cfg_dict.get(section)
        if not isinstance(sect, MutableMapping):
            continue
        # heuristic parse order
        parsed: Any = value
        for caster in (int, float):
            try:
                parsed = caster(value)
                break
            except Exception:
                pass
        if value.lower() in {"true", "false"}:
            parsed = value.lower() == "true"
        # JSON objects/arrays
        if isinstance(parsed, str) and (value.startswith("[") or value.startswith("{")):
            try:
                parsed = json.loads(value)
            except Exception:
                pass
        sect[field] = parsed  # type: ignore[index]


def _dict_to_config(cfg_dict: Mapping[str, Any]) -> MasterConfig:
    """Instantiate MasterConfig from nested dict (expects same shape as _as_dict)."""
    base = MasterConfig.from_defaults()
    # atomic
    a = cfg_dict.get("atomic", {})
    base.atomic.n_atoms = a.get("n_atoms", base.atomic.n_atoms)
    base.atomic.freqs_cm = a.get("freqs_cm", base.atomic.freqs_cm)
    base.atomic.dip_moments = a.get("dip_moments", base.atomic.dip_moments)
    base.atomic.at_coupling_cm = a.get("at_coupling_cm", base.atomic.at_coupling_cm)
    base.atomic.delta_cm = a.get("delta_cm", base.atomic.delta_cm)
    # laser
    l = cfg_dict.get("laser", {})
    base.laser.pulse_fwhm_fs = l.get("pulse_fwhm_fs", base.laser.pulse_fwhm_fs)
    base.laser.base_amplitude = l.get("base_amplitude", base.laser.base_amplitude)
    base.laser.envelope_type = l.get("envelope_type", base.laser.envelope_type)
    base.laser.carrier_freq_cm = l.get("carrier_freq_cm", base.laser.carrier_freq_cm)
    # bath
    b = cfg_dict.get("bath", {})
    base.bath.bath_type = b.get("bath_type", base.bath.bath_type)
    base.bath.temperature = b.get("temperature", base.bath.temperature)
    base.bath.cutoff = b.get("cutoff", base.bath.cutoff)
    base.bath.coupling = b.get("coupling", base.bath.coupling)
    # signal
    s = cfg_dict.get("signal", {})
    base.signal.phase_cycling_phases = s.get(
        "phase_cycling_phases", base.signal.phase_cycling_phases
    )
    base.signal.detection_phase = s.get("detection_phase", base.signal.detection_phase)
    base.signal.ift_component = s.get("ift_component", base.signal.ift_component)
    base.signal.relative_e0s = s.get("relative_e0s", base.signal.relative_e0s)
    base.signal.n_phases = s.get("n_phases", base.signal.n_phases)
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
    base.window.t_det_max_fs = w.get("t_det_max_fs", base.window.t_det_max_fs)
    base.window.dt_fs = w.get("dt_fs", base.window.dt_fs)
    base.window.batches = w.get("batches", base.window.batches)
    base.window.n_freqs = w.get("n_freqs", base.window.n_freqs)
    base.window.rwa_sl = w.get("rwa_sl", base.window.rwa_sl)
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


def load_config(
    path: str | os.PathLike | None = None,
    *,
    overrides: Mapping[str, Any] | None = None,
    use_env: bool = False,
    env_prefix: str = "QSPEC_",
    project_file: bool = True,
) -> MasterConfig:
    """Load configuration applying layered overrides.

    Parameters
    ----------
    path:
        Optional explicit config file (YAML / TOML / JSON). Applied after project file.
    overrides:
        Final in-memory mapping to deep-merge last (highest precedence).
    use_env:
        Include environment variable layer.
    env_prefix:
        Prefix for environment variables (default 'QSPEC_'). Case-insensitive.
    project_file:
        If True, attempt to load `qspectro2d.config.yml` or `.toml` from repository root.
    """

    base_cfg = MasterConfig.from_defaults()
    cfg_dict = _as_dict(base_cfg)

    # project file search (same directory traversal as paths.find_project_root could offer; simplified here)
    if project_file:
        root = (
            Path(__file__).resolve().parents[2]
        )  # .../code/python/qspectro2d -> project root
        for candidate in [
            root / "qspectro2d.config.yml",
            root / "qspectro2d.config.yaml",
            root / "qspectro2d.config.toml",
            root / "qspectro2d.config.json",
        ]:
            if candidate.exists():
                _deep_merge(cfg_dict, _load_file(candidate))
                break

    if path is not None:
        _deep_merge(cfg_dict, _load_file(Path(path)))

    if use_env:
        _apply_env(cfg_dict, env_prefix)

    if overrides:
        _deep_merge(cfg_dict, overrides)

    return _dict_to_config(cfg_dict)


__all__ = ["load_config"]
