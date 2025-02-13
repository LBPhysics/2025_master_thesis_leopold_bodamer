#%%
import matplotlib.pyplot as plt
from qutip import *
import numpy as np
import os
output_dir = os.getcwd()  # Gets the current working directory
os.makedirs(output_dir, exist_ok=True)
# =============================
# SYSTEM PARAMETERS
# =============================
# Energy Landscape (Natural Units: c = 1, hbar = 1)
fixed_lam = 1.0                 # Fixed wavelength (arbitrary units)
alpha = 1.0                     # Coupling strength of dipoles (Fine structure constant?)

# Atomic Parameters
omega_a = 2 * np.pi / fixed_lam # Energy splitting of the atom (ground state = 0)
omega_c = 1 * omega_a           # energysplitting of the field  Rabi freq coupling to laser mu = 1 * omega_a                    # Dipole matrix element of each atom
mu      = 1 * omega_a           # Dipole matrix element of each atom
omega_R = 0.05 * omega_a        # energysplitting of the field  Rabi freq coupling to laser field  for first 2 lasers  -> E_field_0 dot dot Dip_op, parallel

# Lindblad Operators (Decay and Dephasing Rates)
gamma_0 = 0.05                  # Decay rate of the atoms
gamma_kappa = 0.005             # Decay rate of the cavity
gamma_phi = 0.                  # Dephasing rate of the atoms

# =============================
# TOPOLOGY CONFIGURATION
# =============================
n_th_a = 5.0                    # avg number of thermal bath excitation

n_chains = 1                    # Number of chains
n_rings = 1                     # Number of rings
N_atoms = n_chains * n_rings    # Total number of atoms
distance = 1.0                  # Distance between atoms (scaled by fixed_lam)

# =============================
# TIME EVOLUTION PARAMETERS
# =============================
t_max           = 10                   # Maximum simulation time
fine_steps      = 1000                   # Number of steps for high-resolution
fine_spacing    = t_max / fine_steps    # high resolution
sparse_spacing  = 10 * fine_spacing     # for T
# Time Arrays

Delta_ts = [1*fine_spacing] * 3 # Pulse widths

times   = np.arange(0, t_max, fine_spacing)       # High-resolution times array to do the evolutions
times   = np.linspace(0, 100, 1001)

first_entry = (Delta_ts[1] + Delta_ts[2])
last_entry = (t_max - (2 * Delta_ts[0] + Delta_ts[1] + Delta_ts[2]))
times_T = np.arange(first_entry, last_entry, sparse_spacing)

# =============================
# GEOMETRY DEFINITIONS
# =============================
def chain_positions(distance, N_atoms):
    """ Generate atomic positions in a linear chain. """
    return np.array([[0, 0, i * distance] for i in range(N_atoms)])

def z_rotation(angle):
    """ Generate a 3D rotation matrix around the z-axis. """
    return np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])

def ring_positions(distance, n_chains):
    """ Generate atomic positions in a ring. """
    dphi = 2 * np.pi / n_chains
    radius = 0 if n_chains == 1 else distance / (2 * np.sin(np.pi / n_chains))
    return np.array([z_rotation(dphi * i) @ [radius, 0, 0] for i in range(n_chains)])

def cyl_positions(distance, N_atoms, n_chains):
    """ Generate atomic positions in a cylindrical structure. """
    Pos_chain = chain_positions(distance, N_atoms // n_chains)
    Pos_ring = ring_positions(distance, n_chains)
    return np.vstack([Pos_chain + Pos_ring[i] for i in range(n_chains)])

# =============================
# MORE FUNCTIONS
# =============================
def sample_frequencies(E0, Delta, N_atoms):
    """ Sample atomic frequencies from a Gaussian distribution. """
    return np.random.normal(loc=E0, scale=Delta / 2, size=N_atoms)

def plot_positive_color_map(x, y, data, T = np.inf, space="real", type="real", positive=False, safe=False):
    """
    Create a color plot of 2D functional data for positive x and y values only.

    Parameters:
        x (np.array): 1D array of x values.
        y (np.array): 1D array of y values.
        data (np.array): 2D array of data values aligned with x and y.
        T (float): Temperature parameter to include in plot title and file name.
        space (str): Either 'real' or 'freq' specifying the space of the data.
        type (str): Type of data ('real', 'imag', 'abs', or 'phase'). Used only if space="freq".
        positive (bool): Whether to use only positive values of x and y.
        safe (bool): If True, saves the plot to a file.

    Returns:
        None
    """
    # Convert x and y into 1D arrays if they're not
    x = np.array(x)
    y = np.array(y)

    if positive:
        # Filter for positive x and y values
        positive_x_indices = np.where(x > 0)[0]  # Indices where x > 0
        positive_y_indices = np.where(y > 0)[0]  # Indices where y > 0
        x = x[positive_x_indices]
        y = y[positive_y_indices]
        data = data[np.ix_(positive_y_indices, positive_x_indices)]

    label = r"$\propto E_{out} / E_0$"# \quad \propto \quad P / E_0
    if space == "real":
        colormap = "viridis"
        title = f"Real space 2D Spectrum (arb. units)"
        if T != np.inf:
            title += f" at T ={T:.2f}"
        x_title = r"$t_{det}$ (arb. units)"
        y_title = r"$\tau_{coh}$ (arb. units)"
    elif space == "freq":
        colormap = "plasma"
        if type == "real":
            title = f"Freq space, Real 2D Spectrum (arb. units)"
            data = np.real(data)
        elif type == "imag":
            title = f"Freq space, Imag 2D Spectrum (arb. units)"
            data = np.imag(data)
        elif type == "abs":
            title = f"Freq space, Abs 2D Spectrum (arb. units)"
            data = np.abs(data)
        elif type == "phase":
            title = "Freq space, Phase 2D Spectrum (arb. units)"
            data = np.angle(data)
        else:
            raise ValueError("Invalid Type. Must be 'real', 'imag', 'abs', or 'phase'.")

        if T != np.inf:
            title += f" at T ={T:.2f}"

        title += f" at T ={T:.2f}"
        x_title = r"$\omega_{t_{det}}$ (arb. units)"
        y_title = r"$\omega_{\tau_{coh}}$ (arb. units)"

    else:
        raise ValueError("Invalid space. Must be 'real' or 'freq'.")

    # Check and adjust the dimensions of x, y, and data
    if data.shape[1] != len(x):
        raise ValueError(f"Length of x ({len(x)}) must match the number of columns in data ({data.shape[1]}).")
    if data.shape[0] != len(y):
        raise ValueError(f"Length of y ({len(y)}) must match the number of rows in data ({data.shape[0]}).")

    # Add 1 to the dimensions of x and y for correct pcolormesh behavior
    x = np.concatenate([x, [x[-1] + (x[-1] - x[-2])]])  # Add an extra value for the last x edge
    y = np.concatenate([y, [y[-1] + (y[-1] - y[-2])]])  # Add an extra value for the last y edge

    # Plot the color map
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(x, y, data / omega_R, shading="auto", cmap=colormap)
    plt.colorbar(label=label)
    plt.title(title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)

    # Save the plot if safe is True
    if safe and output_dir is not None:
        file_name_combined = (
            f"Classical_lam{fixed_lam:.1f}_alpha={alpha:.2f}_g_0{gamma_0:.2f}_g_phi{gamma_phi:.2f}_"
            f"{n_rings}x{n_chains}_dist={distance:.2f}_positive={positive}_space={space}"
        )
        if space == "freq":
            file_name_combined += f"_type={type}"
        file_name_combined += ".svg"
        save_path_combined = os.path.join(output_dir, file_name_combined)
        plt.savefig(save_path_combined)

    plt.show()

def jcm_hamilton0(distance, n_rings, n_chains):
    N_atoms = n_chains * n_rings
    Pos = cyl_positions(distance, N_atoms, n_chains)
    H_a = 0
    for a in range(N_atoms):
        for b in range(N_atoms):
            op = sm_list[a].dag() * sm_list[b]
            if a != b:
                ra, rb = Pos[a, :], Pos[b, :]
                H_a += alpha / (np.linalg.norm(rb-ra)) * op
            else:
                H_a += omega_a * op                 # Diagonals except for |0><0|
    H_c = omega_c * destroy_cav.dag() * destroy_cav

    return tensor(H_c, qeye(N_atoms+1)) + tensor(qeye(cavity_modes + 1), H_a)

def get_times_for_T(T, spacing=fine_spacing):
    # Adjust the first and last entries of times_tau
    first_tau = (Delta_ts[0] + Delta_ts[1])
    last_tau = (t_max - Delta_ts[2] - T - Delta_ts[0])
    times_tau = np.arange(first_tau, last_tau, spacing)
    # Adjust the first and last entries of times_t
    first_t = (Delta_ts[2])
    last_t = (t_max - Delta_ts[0] - T - 2 * Delta_ts[2])
    times_t = np.arange(first_t, last_t, spacing)

    return times_tau, times_t

def El_field(t, args):
    """ Define a time-dependent electric field pulse. """
    t0, Delta, omega, phi, E0 = args['time'], args['Delta'], args['omega'], args['phi'], args['E0']
    E = np.exp(1j * (omega * t + phi))
    E += np.conjugate(E)
    E *= np.cos(np.pi * (t - t0) / (2 * Delta))**2 * np.heaviside(t - (t0 - Delta), 0) * np.heaviside(t0 + Delta - t, 0)
    return E0 * np.real(E) / 2

# =============================
# QUANTUM SYSTEM INITIALIZATION
# =============================
# Define Atomic States
# Define the ground & the excited states
# atomic dofs
atom_g  =  basis(N_atoms + 1, 0)
atom_es = [basis(N_atoms + 1, i) for i in range(1, N_atoms + 1)]

# cavity dofs
# New cavity Hilbert space dimension (6-level system for 0 to 5 photons)
cavity_modes = 1

# Create cavity states for 0 to 5 photons
cav_g   =  basis(cavity_modes + 1, 0)                                        # vacuum state (0 photons)
cav_es  = [basis(cavity_modes + 1, i) for i in range(1, cavity_modes + 1)]

# Modify operators accordingly (annihilation and creation)
destroy_cav = destroy(cavity_modes + 1)

# initial state
psi_ini = tensor(cav_g, atom_es[0])
# combined dofs
sm_list  = []   # lowering operators of atomic system
dip_list = []   # dipole operators for every state  (|i><0|+|0><i|) * mu
Dip_op   = 0    # collective sigma_x operator for the system
Sigma_m  = 0
for i in range(N_atoms):
    op = atom_g * atom_es[i].dag()     #sigma_m = |0><i|, cavity is unaffected
    sm_list.append(op)
    Sigma_m += op
    dip_op = (op.dag() + op)
    dip_list.append(dip_op)
    Dip_op += dip_op
comm_list = [commutator(op.dag(), op) for op in sm_list]
Deph_op = sum(comm_list)

# Define System Hamiltonian
H0 = jcm_hamilton0(distance, n_rings, n_chains)

# =============================
# COLLAPSE AND EXPECT OPERATORS
# =============================
def jcm_c_ops(cavity_modes, n_th_a):
    destroy_cav = destroy(cavity_modes + 1)

    c_op_list = []
    # Collapse operators for the cavity (decay to photon numbers 0 to 5)
    rate = gamma_kappa * (1 + n_th_a)
    if rate > 0.0:
        c_op_list.append(np.sqrt(rate) * tensor(destroy_cav, qeye(N_atoms+1)))
    # excitation operators of the cavity, if temperature > 0
    rate = gamma_kappa * n_th_a
    if rate > 0.0:
        c_op_list.append(np.sqrt(rate) * tensor(destroy_cav.dag(), qeye(N_atoms+1)))
    # Decay operators for the atoms
    rate = gamma_0
    if rate > 0.0:
       c_op_list.append([np.sqrt(rate) * tensor(qeye(cavity_modes + 1), op) for op in sm_list])

    # Dephasing operators for the atoms
    rate = gamma_phi
    if rate > 0.0:
        c_op_list.append([np.sqrt(rate) * tensor(qeye(cavity_modes + 1), commutator(op.dag(), op)) for op in sm_list])
    return c_op_list

a = tensor(destroy_cav, qeye(N_atoms+1))
sm = tensor(qeye(cavity_modes+1), sm_list[0])
eop_a = a.dag() * a
eop_sm = sm.dag() * sm
e_op_list = [eop_a, eop_sm]

# =============================
# SYSTEM EVOLUTION
# =============================
Omegas = [omega_R, omega_R, omega_R * 1e-1]   # Rabi frequencies
omegas = [omega_a] * 3                      # Laser frequencies (on resonance)

#I = (tensor(destroy_cav, Sigma_m.dag()) + tensor(destroy_cav.dag(), Sigma_m)).unit() # RWA
I = (tensor(qeye(cavity_modes + 1), Dip_op) * tensor((destroy_cav.dag() + destroy_cav), qeye(N_atoms+1)))
HI = [I, El_field]                    # Interaction Hamiltonian
H = [H0, HI]

phases = [-i * np.pi / 2 for i in range(4)]

## Calculationsç

### Compute Pulse
def compute_pulse(psi_ini, times, phi, i):
    progress_bar = ""
    if i == 2 and times[0] >= times[len(times)//2]:
        progress_bar='tqdm'           # Enable tqdm progress bar (or 'text', 'enhanced')

    options = Options(nsteps=15000, rtol=1e-12, atol=1e-12,
    store_states=True,             # Save the states from the Qutip calculation
    progress_bar=progress_bar,     # Enable tqdm progress bar (or 'text', 'enhanced')
    )
    sx = tensor(qeye(cavity_modes+1), Dip_op)
    a = tensor(destroy(cavity_modes+1), qeye(N_atoms+1))
    x = a + a.dag()
    Deph = tensor(qeye(cavity_modes+1), Deph_op)

    cavity_spectrum = "0 if (w <= 0) else {kappa}".format(kappa=gamma_kappa)
    atom_spectrum_decay = "0 if (w <= 0) else {gamma}".format(gamma=gamma_0)
    atom_spectrum_dephase = "0 if (w <= 0) else {gamma}".format(gamma=gamma_phi)
    a_ops = [[x, cavity_spectrum], [sx, atom_spectrum_decay],[Deph, atom_spectrum_dephase],]

    args = {
        'phi': phi,
        'time': times[0] + Delta_ts[i],
        'omega': omegas[i],
        'Delta': Delta_ts[i],
        'E0': Omegas[i]
    }
    #result = mesolve(H, psi_ini, times, e_ops=e_op_list, c_ops=c_op_list, args=args, options=options)
    result = brmesolve(H, psi_ini, times, a_ops=a_ops, e_ops=e_op_list, args=args, options=options) #
    return result

### Compute Two-Dimensional Polarization
def compute_two_dimensional_polarization(T_val, phi_1, phi_2):
    # get the symmetric times, t / tau
    tau_values, t_values = get_times_for_T(T_val)

    # initialize the time domain Spectroscopy
    data = np.zeros((len(tau_values), len(t_values)))

    # only make the necessary steps (don't calculate too many states that we don't need)
    index_0 = np.abs(times - (tau_values[-1] - Delta_ts[1] + Delta_ts[0])).argmin()  # Find the closest index to reduce computation time
    times_0 = times[:index_0+1]

    # calculate the evolution of the first pulse in the desired range
    data_1  = compute_pulse(psi_ini, times_0, phi_1, 0)

    # for every tau value -> calculate the next pulses
    for tau_idx, tau in enumerate(tau_values):
        # find the position in times, which corresponds to the current tau
        i       = np.abs((times_0 + Delta_ts[1] - Delta_ts[0]) - tau_values[tau_idx]).argmin()

        # take this state and evolve it with the second pulse, but only as far as desired
        psi_1   = data_1.states[i]

        # select range
        j = np.abs((times - times_0[i] - Delta_ts[1] + Delta_ts[2]) - T_val).argmin()  # Find the closest index to reduce computation time
        times_1 = times[i:j+1]

        data_2  = compute_pulse(psi_1, times_1, phi_2, 1)
        psi_2   = data_2.states[j-i]

        times_2 = times[j:]

        data_f  = compute_pulse(psi_2, times_2, 0, 2)

        for t_idx, t in enumerate(t_values):
            if t_idx + tau_idx < len(tau_values):

                k = np.abs(t_values[t_idx] - (times_2 - times_2[0] - Delta_ts[2])).argmin()
                psi_f = data_f.states[k]
                data[tau_idx, t_idx] = expect(tensor(qeye(cavity_modes + 1), Dip_op), psi_f)

                # make one plot for this case
                if t == t_values[len(t_values)//2] and tau == tau_values[len(t_values)//3]:
                    #print(data[tau_idx, t_idx])
                    args0 = {
                        'phi': phi_1,
                        'time': times_0[0] + Delta_ts[0],
                        'omega': omegas[0],
                        'Delta': Delta_ts[0],
                        'E0': Omegas[0]
                    }
                    args1 = {
                        'phi': phi_2,
                        'time': times_0[i] + Delta_ts[1],
                        'omega': omegas[1],
                        'Delta': Delta_ts[1],
                        'E0': Omegas[1]
                    }
                    args2 = {
                        'phi': 0,
                        'time': times_1[j-i] + Delta_ts[2],
                        'omega': omegas[2],
                        'Delta': Delta_ts[2],
                        'E0': Omegas[2]
                    }

                    times1 = times_0[:i]
                    times2 = times_1[:j]
                    times3 = times_2

                    # Calculate the electric fields for each time range
                    E_1 = [(El_field(t, args0) / omega_R) for t in times1]
                    E_2 = [(El_field(t, args1) / omega_R) for t in times2]
                    E_3 = [(El_field(t, args2) / omega_R) for t in times3]
                    times_plot = np.concatenate([times1, times2, times3])
                    E_total = E_1 + E_2 + E_3

                    fig, axs = plt.subplots(2, 1, figsize=(15, 5), sharex=True)

                    axs[0].plot(times_plot, E_total, label='Electric Field (Real part)', color='black')
                    axs[0].set_ylabel('E')

                    # Add vertical lines to mark pulse times
                    axs[0].axvline(times_0[0] + Delta_ts[0], color='r', linestyle='--', label='Pulse 1')
                    axs[0].axvline(times_0[i] + Delta_ts[1], color='g', linestyle='--', label='Pulse 2')
                    axs[0].axvline(times_1[j-i] + Delta_ts[2], color='g', linestyle='--', label='Pulse 3')
                    axs[0].axvline(times_2[k], color='b', linestyle='--', label='Detection')
                    axs[0].legend()

                    # Plot Expectation Value in the second subplot
                    data_cav_e = []
                    data_atom_e = []

                    # Concatenate the data for different segments
                    data_cav_e.append(data_1.expect[0][:i])  # First part (data_1)
                    data_cav_e.append(data_2.expect[0][:j])  # Second part (data_2)
                    data_cav_e.append(data_f.expect[0])      # Final part (data_f)
                    data_cav_e = np.concatenate(data_cav_e)
                    data_atom_e.append(data_1.expect[1][:i])  # First part (data_1)
                    data_atom_e.append(data_2.expect[1][:j])  # Second part (data_2)
                    data_atom_e.append(data_f.expect[1])      # Final part (data_f)
                    data_atom_e = np.concatenate(data_atom_e)

                    axs[1].plot(times_plot, data_cav_e,label = r"$|e>_{cav}$")
                    axs[1].plot(times_plot, data_atom_e,label = r"|e>_{atom}")
                    #axs[1].plot(times_plot, data_cav_e+data_atom_e,label = r"sum")
                    axs[1].set_xlabel('Time (t)')
                    axs[1].set_ylabel('||g>|²')

                    fig.suptitle(f"tau = {tau:.2f}, T = {T_val:.2f}")
                    plt.legend()
                    plt.show()

    return tau_values, t_values, data


import numpy as np
import matplotlib.pyplot as plt


# Ensure your functions are properly imported (e.g., compute_two_dimensional_polarization, plot_positive_color_map)
# Example import statement for these functions:
# from your_module import compute_two_dimensional_polarization, plot_positive_color_map

def main():
    # Example values (ensure these are set correctly in your script)
    T_test = times_T[0]
    data_test_0 = compute_two_dimensional_polarization(T_test, phases[0], phases[0])

    # Plot the first color map
    plot_positive_color_map(data_test_0[1], data_test_0[0], data_test_0[2], T=T_test, safe=True)

    # Frequency domain calculations and plotting
    tfreqs = np.fft.fftfreq(len(data_test_0[1]), d=(data_test_0[1][1] - data_test_0[1][0]))
    taufreqs = np.fft.fftfreq(len(data_test_0[0]), d=(data_test_0[0][1] - data_test_0[0][0]))

    plot_positive_color_map(tfreqs, taufreqs, np.fft.fft2(data_test_0[2]), positive=True, type="real", space="freq",
                            T=T_test, safe=True)


# If this script is executed directly, run the main function
if __name__ == "__main__":
    main()
