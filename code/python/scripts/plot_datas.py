"""
Unified 1D/2D Electronic Spectroscopy Data Plotting Script

Clean, flexible CLI to load results and produce time/frequency-domain plots.

Examples (Windows PowerShell):
    # Base path (no suffix) or an .npz file
    python plot_datas.py --abs_path "C:/path/to/data/run_001"
    python plot_datas.py --abs_path "C:/path/to/data/run_042.npz"
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from typing import Any, List, Optional, Sequence, Tuple, Dict, cast
from qutip import BosonicEnvironment
import sys
import argparse
import numpy as np
import warnings

### Project-specific imports
from qspectro2d.visualization.plotting import (
    plot_1d_el_field,
    plot_2d_el_field,
    add_text_box,
)
from qspectro2d.spectroscopy.post_processing import (
    extend_time_axes,
    compute_1d_fft_wavenumber,
    compute_2d_fft_wavenumber,
)
from qspectro2d import generate_unique_plot_filename
from qspectro2d.core.bath_system.bath_fcts import extract_bath_parameters
from qspectro2d import load_simulation_data

from plotstyle import init_style, save_fig
from project_config.paths import FIGURES_PYTHON_DIR

# Style is initialised once on import.
init_style()


# Suppress noisy but harmless warnings
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="overflow encountered in exp"
)


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
                ext_t_det, ext_data = extend_time_axes(
                    data=data, t_det=t_det, pad_t_det=extend_for
                )
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
                        system=system,
                        sim_config=sim_config,
                        domain=domain,
                        component=comp,
                        figures_root=FIGURES_PYTHON_DIR,
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
                    base_name = generate_unique_plot_filename(
                        system=system,
                        sim_config=sim_config,
                        domain=domain,
                        component=comp,
                        figures_root=FIGURES_PYTHON_DIR,
                    )
                    safe_label = str(st).replace(" ", "_")
                    filename = f"{base_name}_{safe_label}"
                    saved = save_fig(
                        fig,
                        filename=filename,
                    )
                    _collect_saved_paths(saved_paths, saved)
        except Exception as e:
            # Defensive: continue other components even if one fails
            print(f"❌ Error plotting {dimension.upper()} {domain} {comp}: {e}")
    return saved_paths


def plot_data(
    loaded_data_and_info: Dict[str, Any], plot_config: Dict[str, Any], dimension: str
) -> List[str]:
    """Plot time and/or frequency domain data for 1D / 2D simulations.

    Parameters
    ----------
    loaded_data_and_info : dict
        Dict produced by the new save/load workflow.
    plot_config : dict
        Keys (optional):
            - plot_time_domain (bool)
            - plot_frequency_domain (bool)
            - components (list[str])
            - extend_for (tuple[int,int])
            - section (1D: (min,max) | 2D: ((min,max),(min,max)))
            - verbose (bool)
    dimension : str
        '1d' or '2d'.
    """
    from qspectro2d.core.atomic_system.system_class import AtomicSystem
    from qspectro2d.core.laser_system.laser_class import LaserPulseSequence
    from qspectro2d.core.simulation import SimulationConfig

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

    w0 = system.frequencies_fs[0]
    bath_dict = extract_bath_parameters(bath_env, w0)
    laser_dict = {k: v for k, v in laser.to_dict().items() if k != "pulses"}
    meta = {**system.to_dict(), **bath_dict, **laser_dict, **sim_config.to_dict()}

    # Announce plotting context
    print(f"➡️  Plotting: dimension={dimension}, signals={list(signal_types)}")
    print(f"   t_det: n={len(t_det)} range=[{t_det[0]:.2f},{t_det[-1]:.2f}] fs")
    if dimension == "2d" and t_coh is not None:
        print(f"   t_coh: n={len(t_coh)} range=[{t_coh[0]:.2f},{t_coh[-1]:.2f}] fs")

    # ---- Config with defaults
    extend_for = plot_config.get("extend_for", (0, 0))
    section = plot_config.get("section")
    components = ["real", "abs", "img", "phase"]  # default all
    plot_time = bool(plot_config.get("plot_time_domain", True))
    plot_freq = bool(plot_config.get("plot_frequency_domain", True))
    if (
        dimension == "1d"
        and isinstance(section, (list, tuple))
        and len(section) == 2
        and isinstance(section[0], (list, tuple))
    ):
        # Provided as 2D section but we are 1D: take first window only
        section = section[0]

    # For 2D frequency plots, accept flexible section formats and normalize:
    # - (min, max)              -> ((min, max), (min, max))
    # - (min1, max1, min2, max2) -> ((min1, max1), (min2, max2))
    if dimension == "2d" and section is not None and isinstance(section, (list, tuple)):
        try:
            if len(section) == 2 and not isinstance(section[0], (list, tuple)):
                section = (
                    (float(section[0]), float(section[1])),
                    (float(section[0]), float(section[1])),
                )
            elif len(section) == 4 and not isinstance(section[0], (list, tuple)):
                section = (
                    (float(section[0]), float(section[1])),
                    (float(section[2]), float(section[3])),
                )
            # else: assume already in the shape ((a,b),(c,d))
        except Exception:
            # On any parsing issue, drop the section to avoid plotting errors
            section = None
    time_components = list(components)
    spectral_components = list(components)

    # Choose plotting callable
    plot_func = plot_1d_el_field if dimension == "1d" else plot_2d_el_field

    # ---- Time domain
    saved_all: List[str] = []
    if plot_time:
        print("📊 Time domain ...")
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
        saved_all.extend(saved)
        for p in saved:
            print(f"   💾 {p}")

    # ---- Frequency domain
    if plot_freq:
        print(f"📊 Frequency domain ... (extend={extend_for})")
        try:
            freq_axes, freq_datas, kept_types = _extend_and_fft(
                loaded_data_and_info=loaded_data_and_info,
                extend_for=extend_for,
                dimension=dimension,
            )
            axis_det_f, axis_coh_f = (
                (freq_axes, None) if dimension == "1d" else freq_axes
            )
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
            saved_all.extend(saved)
            for p in saved:
                print(f"   💾 {p}")
        except Exception as e:  # pragma: no cover
            print(f"❌ Frequency domain skipped: {e}")

    plt.close("all")
    return saved_all


def main():
    parser = argparse.ArgumentParser(
        description="Plot 1D or 2D electronic spectroscopy data (time/frequency domains)."
    )

    # Required input
    parser.add_argument(
        "--abs_path",
        type=str,
        default=None,
        help=(
            "Absolute path to the saved results file (ends with '_data.npz' or '_info.pkl'). "
            "Alternatively, pass the base path without suffix to auto-resolve."
        ),
    )

    parser.add_argument(
        "--extend",
        type=int,
        nargs=2,
        default=(1, 10),
        help="Zero-padding factors for (<0, >t_max) before FFT (1 1 disables)",
    )
    parser.add_argument(
        "--section",
        type=float,
        nargs="+",
        default=(0, 2),
        help="Frequency window: 1D -> two floats (min max), 2D -> four floats (min max min max)",
    )

    args = parser.parse_args()

    try:
        # Parse section into internal structure: None | (a,b) | ((a,b),(c,d))
        section: Optional[Any] = None
        if args.section is not None:
            if len(args.section) == 2:
                section = (args.section[0], args.section[1])
            elif len(args.section) == 4:
                section = (
                    (args.section[0], args.section[1]),
                    (args.section[2], args.section[3]),
                )
            else:
                raise ValueError("--section expects 2 (1D) or 4 (2D) floats")

        plot_time = True
        plot_freq = True

        all_saved: List[str] = []
        print(f"🔄 Loading: {args.abs_path}")
        loaded_data_and_info = load_simulation_data(abs_path=args.abs_path)

        # Quick probe to decide dimension and basic info
        t_det_axis = loaded_data_and_info.get("t_det")
        t_coh_axis = loaded_data_and_info.get("t_coh")
        sim_config = loaded_data_and_info["sim_config"]

        try:
            is_2d = (
                t_coh_axis is not None
                and hasattr(t_coh_axis, "__len__")
                and len(t_coh_axis) > 0
            )
        except Exception:
            is_2d = False

        # Axes info
        try:
            n_t_det = len(t_det_axis) if t_det_axis is not None else 0
            det_rng = (
                f"[{float(t_det_axis[0]):.2f},{float(t_det_axis[-1]):.2f}] fs"
                if n_t_det > 0
                else "[—]"
            )
        except Exception:
            n_t_det, det_rng = 0, "[—]"
        if is_2d:
            try:
                n_t_coh = len(t_coh_axis) if t_coh_axis is not None else 0
                coh_rng = (
                    f"[{float(t_coh_axis[0]):.2f},{float(t_coh_axis[-1]):.2f}] fs"
                    if n_t_coh > 0
                    else "[—]"
                )
            except Exception:
                n_t_coh, coh_rng = 0, "[—]"
        else:
            n_t_coh, coh_rng = 0, "—"
        print(
            f"   Axes: t_det n={n_t_det} {det_rng}; t_coh n={n_t_coh if is_2d else '—'} {coh_rng if is_2d else ''}"
        )

        # If 1D, do not use a 2D section tuple even if provided
        eff_section = section
        if (
            not is_2d
            and isinstance(section, tuple)
            and len(section) == 2
            and isinstance(section[0], tuple)
        ):
            # ((a,b),(c,d)) → (a,b)
            eff_section = section[0]

        # Detect if time-domain signals are all-zero (informative warning only)
        try:
            sigs = [str(s) for s in sim_config.signal_types]
            datas = [loaded_data_and_info.get(s) for s in sigs]
            if datas and all(
                isinstance(a, np.ndarray) and a.size > 0 and np.allclose(a, 0)
                for a in datas
            ):
                print("⚠️  All-zero time-domain signals detected.")
        except Exception:
            pass

        saved = plot_data(
            loaded_data_and_info=loaded_data_and_info,
            plot_config={
                "extend_for": tuple(args.extend),
                "section": eff_section,
                "plot_time_domain": plot_time,
                "plot_frequency_domain": plot_freq,
            },
            dimension="2d" if is_2d else "1d",
        )
        all_saved.extend(saved)

        # Final summary
        print(f"✅ Done. Saved {len(all_saved)} file(s).")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
