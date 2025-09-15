# short figure for Anna
# - Defines trajectories for magnetron, modified cyclotron, and axial motions
# - Uses unique colors/linestyles and LaTeX labels

from pathlib import Path
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from plotstyle import init_style, set_size, save_fig, COLORS

init_style(quiet=True)

# Use interactive Matplotlib in Jupyter (with safe fallback)
from IPython import get_ipython

get_ipython().run_line_magic("matplotlib", "widget")

import matplotlib

print(f"Matplotlib backend: {matplotlib.get_backend()}")

# Parameters (units are arbitrary schematic)
r_minus = 1.0  # magnetron radius
r_plus = 0.35  # modified cyclotron radius
z_amp = 0.60  # axial amplitude
omega_m = 2 * np.pi * 50.0  # magnetron angular freq
omega_p = 2 * np.pi * 1008.0  # modified cyclotron angular freq
omega_z = 2 * np.pi * 80.0  # axial angular freq

# Base curves for schematic
phi = np.linspace(0, 2 * np.pi, 4000)
# Magnetron circle in xy-plane
x_mag = r_minus * np.cos(phi)
y_mag = r_minus * np.sin(phi)
z_mag = np.zeros_like(phi)
# Modified cyclotron circle in xy-plane
x_circ = r_plus * np.cos(phi)
y_circ = -r_plus * np.sin(phi)  # opposite sense for visual contrast
z_circ = np.zeros_like(phi)
# Axial oscillation on z-axis
x_axial = np.zeros_like(phi)
y_axial = np.zeros_like(phi)
z_axial = z_amp * np.cos(phi)


# Combined short trajectory segment (front side)
def traj(
    t: np.ndarray, phi_minus: float = 0.0, phi_plus: float = 0.0, phi_z: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = r_minus * np.cos(omega_m * t + phi_minus) + r_plus * np.cos(omega_p * t + phi_plus)
    y = r_minus * np.sin(omega_m * t + phi_minus) - r_plus * np.sin(omega_p * t + phi_plus)
    z = z_amp * np.cos(omega_z * t + phi_z)
    return x, y, z


t_single = np.linspace(0.0, 0.05, 5000)

# Figure setup using thesis sizing (square figure)
fig = plt.figure()
fig.set_size_inches(*set_size(fraction=0.62, height_ratio=1.0))
ax = fig.add_subplot(111, projection="3d")

# Frame axes
L = 1.7 * max(r_minus, r_plus)
ax.plot([-L, L], [0, 0], [0, 0], linewidth=1.0, color="0.6", linestyle="dotted")
ax.plot([0, 0], [-L, L], [0, 0], linewidth=1.0, color="0.6", linestyle="dotted")
ax.plot([0, 0], [0, 0], [-1.4 * z_amp, 1.4 * z_amp], linewidth=1.0, color="0.6", linestyle="dotted")

# Base elements (unique colors and linestyles from plotstyle palette)
ax.plot(x_mag, y_mag, z_mag, linewidth=2.0, color=COLORS[0], linestyle="solid", label=r"Magnetron")
ax.plot(
    x_circ,
    y_circ,
    z_circ,
    linewidth=2.0,
    color=COLORS[1],
    linestyle="dashed",
    label=r"Modified\ cyclotron",
)
ax.plot(
    x_axial, y_axial, z_axial, linewidth=2.0, color=COLORS[2], linestyle="dashdot", label=r"Axial"
)

# One short overall-motion strand at front
x, y, z = traj(t_single, phi_minus=0.0, phi_plus=0.0, phi_z=0.0)
ax.plot(
    x, y, z, linewidth=1.6, color=COLORS[3], linestyle="solid", label=r"Overall\ motion (segment)"
)

# Labels and limits
ax.set_xlim(-L, L)
ax.set_ylim(-L, L)
ax.set_zlim(-1.4 * z_amp, 1.4 * z_amp)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_zlabel(r"$z$")
ax.set_title(rf"Penning trap motions: $r_-={r_minus:.2f}$, $r_+={r_plus:.2f}$, $z_0={z_amp:.2f}$")

# Style cleanups (no grid; transparent panes)
ax.grid(False)
for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
    pane.set_edgecolor("white")
    pane.fill = False

ax.legend(frameon=True, fontsize=10)
ax.view_init(elev=22, azim=35)
fig.tight_layout()
plt.show()
# Save figure to thesis figures folder
out_base = Path("../../../figures/figures_from_python/misc/penning_trap_schematic_single_segment")
saved_paths = save_fig(fig, out_base, formats=["png", "svg"])  # returns list of saved files
saved_paths[-1] if saved_paths else str(out_base)
