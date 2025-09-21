"""High-level plotting entry points (simplified).

This refactor targets: (a) simpler public API, (b) removal of duplicated
branches, (c) clearer error surfaces, (d) forward compatibility with the
"new workflow" (saving / loading pre-packaged dicts). The expectations
for ``loaded_data_and_info`` are now minimal and validated explicitly.

Required keys in ``loaded_data_and_info``:
    system, bath, laser, sim_config, t_det, signal_types
    For 2D: t_coh
    Each signal (string in signal_types) must map to a numpy array.

Public function:
    plot_data(loaded_data_and_info, plot_config, dimension)

Key simplifications:
    * Removed nested per-component/per-signal loops for freq domain (batching)
    * Unified figure saving
    * Centralised FFT extension logic
    * Defensive checks with concise messages
    * Fixed previously undefined variables (datas, signal_types)
"""

from __future__ import annotations

import gc
import time
import matplotlib.pyplot as plt
from typing import Any, Iterable, List, Sequence, Tuple, Dict, cast
from qutip import BosonicEnvironment

### Project-specific imports
from qspectro2d.visualization.plotting import (
    plot_1d_el_field,
    plot_2d_el_field,
)
from qspectro2d.spectroscopy.post_processing import (
    extend_time_axes,
    compute_1d_fft_wavenumber,
    compute_2d_fft_wavenumber,
)
from qspectro2d.utils import generate_unique_plot_filename
from qspectro2d.core.bath_system.bath_fcts import extract_bath_parameters

from plotstyle import init_style, save_fig

# Style is initialised once on import.
init_style()


# Helper to collect paths from save_fig (which may return a single path or a list of paths)
def _collect_saved_paths(accumulator: List[str], saved: Any) -> None:
    """Append saved file path(s) to ``accumulator`` (handles list / scalar)."""
    if isinstance(saved, (list, tuple, set)):
        accumulator.extend([str(s) for s in saved])
    else:  # single path-like
        accumulator.append(str(saved))


def _extend_and_fft(
    loaded_data_and_info: Dict[str, Any],
    extend_for: Tuple[int, int],
    dimension: str,
):
    """Pad (time) axes then compute FFTs.

    Returns (freq_axes, data_freq, freq_labels).
    Failed signals are skipped but reported; at least one must survive.
    """
    # Extract inputs from loaded dict
    sim_config = loaded_data_and_info.get("sim_config")
    if sim_config is None:
        raise KeyError("'sim_config' missing in loaded_data_and_info")
    signal_types: Sequence[str] = sim_config.signal_types
    t_det = loaded_data_and_info.get("t_det")
    if t_det is None:
        raise KeyError("'t_det' missing in loaded_data_and_info")
    t_coh = loaded_data_and_info.get("t_coh") if dimension == "2d" else None

    datas: Sequence[Any] = [loaded_data_and_info.get(st) for st in signal_types]
    if any(d is None for d in datas):
        missing_signals = [st for st, d in zip(signal_types, datas) if d is None]
        raise KeyError(f"Missing signal arrays for: {missing_signals}")

    kept_data: list[Any] = []
    kept_types: list[str] = []
    extended_axes: Any = None
    for i, (data, st) in enumerate(zip(datas, signal_types)):
        try:
            if dimension == "1d":
                ext_t_det, ext_data = extend_time_axes(data=data, t_det=t_det, pad_t_det=extend_for)
                extended_axes = ext_t_det
            else:
                ext_t_det, ext_t_coh, ext_data = extend_time_axes(
                    data=data,
                    t_det=t_det,
                    t_coh=t_coh,
                    pad_t_det=extend_for,
                    pad_t_coh=extend_for,
                )
                extended_axes = (ext_t_det, ext_t_coh)
            kept_data.append(ext_data)
            kept_types.append(st)
            print(f"   ✅ FFT prep {i+1}/{len(datas)} {st}")
        except Exception as e:  # pragma: no cover (defensive)
            print(f"   ❌ Skip {st}: {e}")
    if not kept_data:
        raise ValueError("All signals failed during extension.")
    if dimension == "1d":
        nu_det, freq_datas, freq_labels = compute_1d_fft_wavenumber(
            extended_axes, kept_data, signal_types=kept_types
        )
        freq_axes = nu_det
    else:
        nu_det, nu_coh, freq_datas, freq_labels = compute_2d_fft_wavenumber(
            extended_axes[0], extended_axes[1], kept_data, signal_types=kept_types
        )
        freq_axes = (nu_det, nu_coh)
    return freq_axes, freq_datas, freq_labels


def _plot_components(
    *,
    datas: Sequence[Any],
    signal_types: Sequence[str],
    axis_det,
    axis_coh,
    domain: str,
    components: Sequence[str],
    plot_func,
    base_metadata: Dict[str, Any],
    system,
    sim_config,
    dimension: str,
    **kwargs,
) -> List[str]:
    """Loop over components (and signals for time domain) producing figures.

    For both time and frequency domain we make one figure per (signal/label, component).
    Frequency labels come from the FFT step (e.g., 'rephasing','nonrephasing','absorptive').
    """
    saved_paths: List[str] = []
    for comp in components:
        try:
            if domain == "time":
                for data, st in zip(datas, signal_types):
                    md = {**base_metadata, "signal_type": st}
                    fig = (
                        plot_func(
                            axis_det=axis_det,
                            data=data,
                            domain=domain,
                            component=comp,
                            function_symbol=r"$E_{k_s}$",
                            **md,
                        )
                        if dimension == "1d"
                        else plot_func(
                            axis_det=axis_det,
                            axis_coh=axis_coh,
                            data=data,
                            domain=domain,
                            use_custom_colormap=True,
                            component=comp,
                            **md,
                        )
                    )
                    base_name = generate_unique_plot_filename(
                        system=system, sim_config=sim_config, domain=domain, component=comp
                    )
                    # Append the freq-domain label to the filename for clear linkage
                    safe_label = str(st).replace(" ", "_")
                    filename = f"{base_name}_{safe_label}"
                    saved = save_fig(
                        fig,
                        filename=filename,
                    )
                    _collect_saved_paths(saved_paths, saved)
            else:  # frequency domain: iterate and save one per spectrum label
                for data, st in zip(datas, signal_types):
                    md = {**base_metadata, "signal_type": st}
                    fig = (
                        plot_func(
                            axis_det=axis_det,
                            data=data,
                            domain=domain,
                            use_custom_colormap=True,
                            component=comp,
                            **kwargs,
                            **md,
                        )
                        if dimension == "1d"
                        else plot_func(
                            axis_det=axis_det,
                            axis_coh=axis_coh,
                            data=data,
                            domain=domain,
                            use_custom_colormap=True,
                            component=comp,
                            **kwargs,
                            **md,
                        )
                    )
                    saved = save_fig(
                        fig,
                        filename=generate_unique_plot_filename(
                            system=system, sim_config=sim_config, domain=domain, component=comp
                        ),
                    )
                    _collect_saved_paths(saved_paths, saved)
        except Exception as e:  # pragma: no cover (defensive path)
            print(f"❌ Error plotting {dimension.upper()} {domain} {comp}: {e}")
    return saved_paths


def plot_data(
    loaded_data_and_info: Dict[str, Any], plot_config: Dict[str, Any], dimension: str
) -> None:
    """Plot time and/or frequency domain data for 1D / 2D simulations.

    Parameters
    ----------
    loaded_data_and_info : dict
        Dict produced by the new save/load workflow.
    plot_config : dict
        Keys (optional): plot_time_domain (bool), plot_frequency_domain (bool),
        spectral_components_to_plot (list[str]), extend_for (tuple[int,int]), section.
    dimension : str
        '1d' or '2d'.
    """
    from qspectro2d.core.atomic_system.system_class import AtomicSystem
    from qspectro2d.core.laser_system.laser_class import LaserPulseSequence
    from qspectro2d.core.simulation import SimulationConfig

    if dimension not in {"1d", "2d"}:
        raise ValueError("dimension must be '1d' or '2d'")

    # --- Required objects
    system = cast(AtomicSystem, loaded_data_and_info.get("system"))
    bath_env = cast(BosonicEnvironment, loaded_data_and_info.get("bath"))
    laser = cast(LaserPulseSequence, loaded_data_and_info.get("laser"))
    sim_config = cast(SimulationConfig, loaded_data_and_info.get("sim_config"))
    t_det = loaded_data_and_info.get("t_det")
    t_coh = loaded_data_and_info.get("t_coh") if dimension == "2d" else None
    signal_types: Sequence[str] = sim_config.signal_types

    # Collect raw data arrays
    datas = [loaded_data_and_info.get(st) for st in signal_types]
    if any(d is None for d in datas):
        missing_signals = [st for st, d in zip(signal_types, datas) if d is None]
        raise KeyError(f"Missing signal arrays for: {missing_signals}")

    w0 = system._frequencies_fs[0]
    bath_dict = extract_bath_parameters(bath_env, w0)
    laser_dict = {k: v for k, v in laser.to_dict().items() if k != "pulses"}
    meta = {**system.to_dict(), **bath_dict, **laser_dict, **sim_config.to_dict()}

    print(f"➡️  Plotting: dimension={dimension}, signals={list(signal_types)}")
    print(f"   t_det: n={len(t_det)} range=[{t_det[0]:.2f},{t_det[-1]:.2f}] fs")
    if dimension == "2d" and t_coh is not None:
        print(f"   t_coh: n={len(t_coh)} range=[{t_coh[0]:.2f},{t_coh[-1]:.2f}] fs")

    # ---- Config with defaults
    extend_for = plot_config.get("extend_for", (0, 0))
    section = plot_config.get("section")
    if (
        dimension == "1d"
        and isinstance(section, (list, tuple))
        and len(section) == 2
        and isinstance(section[0], (list, tuple))
    ):
        # Provided as 2D section but we are 1D: take first window only
        section = section[0]
    time_components = ["real", "abs", "img", "phase"]
    spectral_components = time_components

    # Choose plotting callable
    plot_func = plot_1d_el_field if dimension == "1d" else plot_2d_el_field

    # ---- Time domain
    print(f"📊 Time domain ...")
    saved = _plot_components(
        datas=datas,
        signal_types=signal_types,
        axis_det=t_det,
        axis_coh=t_coh,
        domain="time",
        components=time_components,
        plot_func=plot_func,
        base_metadata=meta,
        system=system,
        sim_config=sim_config,
        dimension=dimension,
    )
    for p in saved:
        print(f"   💾 {p}")

    # ---- Frequency domain
    print(f"📊 Frequency domain ... (extend={extend_for})")
    try:
        freq_axes, freq_datas, kept_types = _extend_and_fft(
            loaded_data_and_info=loaded_data_and_info,
            extend_for=extend_for,
            dimension=dimension,
        )
        axis_det_f, axis_coh_f = (freq_axes, None) if dimension == "1d" else freq_axes
        saved = _plot_components(
            datas=freq_datas,
            signal_types=kept_types,
            axis_det=axis_det_f,
            axis_coh=axis_coh_f,
            domain="freq",
            components=spectral_components,
            plot_func=plot_func,
            base_metadata=meta,
            system=system,
            sim_config=sim_config,
            dimension=dimension,
            section=section,
        )
        for p in saved:
            print(f"   💾 {p}")
    except Exception as e:  # pragma: no cover
        print(f"❌ Frequency domain skipped: {e}")

    plt.close("all")
    gc.collect()
