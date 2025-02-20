from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
from qutip import *
import numpy as np
import itertools
import os
from tqdm import tqdm

# Matplotlib Einstellungen gemäß den LaTeX-Caption-Formatierungen
plt.rcParams.update({
#    'text.usetex': True,              # Enable LaTeX for text rendering
#    'font.family': 'serif',           # Use a serif font family
#    'font.serif': 'Palatino',         # Set Palatino as the serif font
#    'text.latex.preamble': r'\usepackage{amsmath}',
#    'font.size': 20,                   # Font size for general text
#    'axes.titlesize': 20,              # Font size for axis titles
#    'axes.labelsize': 20,              # Font size for axis labels
#    'xtick.labelsize': 20,             # Font size for x-axis tick labels
#    'ytick.labelsize': 20,             # Font size for y-axis tick labels
#    'legend.fontsize': 20,             # Font size for legends
#    'figure.figsize': [8, 6],          # Size of the plot (width x height)
#    'figure.autolayout': True,         # Automatic layout adjustment
#    'savefig.format': 'svg',           # Default format for saving figures
#    'figure.facecolor': 'none',        # Make the figure face color transparent
#    'axes.facecolor': 'none',          # Make the axes face color transparent
#    'savefig.transparent': True        # Save figures with transparent background
})
output_dir = os.getcwd()  # Gets the current working directory
os.makedirs(output_dir, exist_ok=True)

# allows for interactive plots #%matplotlib notebook

# =============================
# SYSTEM PARAMETERS
# =============================
# Energy Landscape (Natural Units: c = 1, hbar = 1)
fixed_lam = 1.0                 # Fixed wavelength (arbitrary units)
alpha = 1.0                     # Coupling strength of dipoles (Fine structure constant?)

# Atomic Parameters
omega_a = 2 * np.pi / fixed_lam # Energy splitting of the atom (ground state = 0)
mu      = .2 * omega_a          # Dipole matrix element of each atom
E0      = 10                     # energysplitting of the field  Rabi freq coupling to laser field  for first 2 lasers  -> E_field_0 dot dot Dip_op, parallel

E0s     = [E0, E0, .1*E0]       # omega_Rabi = mu * E0 (energysplitting of the field  Rabi freq coupling to laser field  -> E_field_0 dot dot Dip_op, parallel

# Lindblad Operators (Decay and Dephasing Rates)
gamma_0   = 2                  # Decay rate of the atoms
gamma_phi = .0                  # Dephasing rate of the atoms

# =============================
# TOPOLOGY CONFIGURATION
# =============================
n_chains = 1                    # Number of chains
n_rings = 1                     # Number of rings
N_atoms = n_chains * n_rings    # Total number of atoms
distance = 1.0                  # Distance between atoms (scaled by fixed_lam)

# =============================
# TIME EVOLUTION PARAMETERS
# =============================
t_max           = 100                  # Maximum simulation time
fine_steps      = 1200                 # Number of steps for high-resolution
fine_spacing    = t_max / fine_steps   # high resolution
sparse_spacing  = 30 * fine_spacing    # for T

Delta_ts = [1*fine_spacing] * 3 # Pulse widths
omegas   = [omega_a] * 3                        # Laser frequencies (on resonance)

# Time Arrays

times   = np.arange(0, t_max, fine_spacing)       # High-resolution times array to do the evolutions

first_entry = (Delta_ts[1] + Delta_ts[2])
last_entry = np.floor((t_max - (2 * Delta_ts[0] + Delta_ts[1] + Delta_ts[2])) / sparse_spacing) * sparse_spacing
times_T = np.arange(first_entry, last_entry, sparse_spacing)

# Phase Cycling for Averaging
phases = [-i * np.pi / 2 for i in range(4)] # GIVES WEIRD RESULTS

# =============================
# QUANTUM SYSTEM INITIALIZATION
# =============================
# Define Atomic States
atom_g = basis(N_atoms + 1, 0)                                      # Ground state
atom_es = [basis(N_atoms + 1, i) for i in range(1, N_atoms + 1)]    # Excited states

# Define Lowering Operators
sm_list = [atom_g * atom_es[i].dag() for i in range(N_atoms)] # lowering operators of atomic system
dip_op_list = [mu * (op + op.dag()) for op in sm_list]             # dipole operators for every state  (|i><0|+|0><i|) * mu
Sigma_m = sum(sm_list)
Deph_op = sum([commutator(op_a.dag(), op_b) for op_a in sm_list for op_b in sm_list])
Dip_op  = sum(dip_op_list)                # Collective dipole operator


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
    plt.pcolormesh(x, y, data / E0, shading="auto", cmap=colormap)
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

def Plot_example_evo(T_val, tau, phi_1, phi_2, times_0, times_1, times_2, data_1, data_2, data_f, i, j, k):
    ylabels = [r"$|\text{g}\rangle$", r"$|\text{e}\rangle$"]#, r"$|\text{e_1}\rangle$", r"$|\text{e_2}\rangle$"]

    args0 = {
        'phi': phi_1,
        'time': times_0[0] + Delta_ts[0],
        'omega': omegas[0],
        'Delta': Delta_ts[0],
        'E0': E0
    }
    args1 = {
        'phi': phi_2,
        'time': times_0[i] + Delta_ts[1],
        'omega': omegas[1],
        'Delta': Delta_ts[1],
        'E0': E0s[1]
    }
    args2 = {
        'phi': 0,
        'time': times_1[j-i] + Delta_ts[2],
        'omega': omegas[2],
        'Delta': Delta_ts[2],
        'E0': E0s[2]
    }

    times1 = times_0[:i]
    times2 = times_1[:j]
    times3 = times_2

    # Calculate the electric fields for each time range
    E_1 = [(El_field(t, args0) / E0) for t in times1]
    E_2 = [(El_field(t, args1) / E0) for t in times2]
    E_3 = [(El_field(t, args2) / E0) for t in times3]
    times_plot = np.concatenate([times1, times2, times3])
    E_total = E_1 + E_2 + E_3

    fig, axs = plt.subplots(len(e_op_list)+1, 1, figsize=(15, 10), sharex=True)

    axs[0].plot(times_plot, E_total, label='Electric Field (Real part)', color='black')
    axs[0].set_ylabel('E')

    # Add vertical lines to mark pulse times
    axs[0].axvline(times_0[0] + Delta_ts[0], color='r', linestyle='--', label='Pulse 1')
    axs[0].axvline(times_0[i] + Delta_ts[1], color='g', linestyle='--', label='Pulse 2')
    axs[0].axvline(times_1[j-i] + Delta_ts[2], color='g', linestyle='--', label='Pulse 3')
    axs[0].axvline(times_2[k], color='b', linestyle='--', label='Detection')
    axs[0].legend()

    # Initialize an empty list to store all the data for plotting
    datas = []

    # Append the data for each quantum state based on the times
    for idx in range(len(e_op_list)):
        datas.append(np.concatenate([data_1.expect[idx][:i], data_2.expect[idx][:j], data_f.expect[idx]]))  # data for first state

    # Loop over the datas and plot each expectation value
    for idx, (data, label) in enumerate(zip(datas, ylabels)):
        axs[idx+1].plot(times_plot, data)
        axs[idx+1].set_ylabel(label)
        axs[idx+1].legend()

    fig.suptitle(f"tau = {tau:.2f}, T = {T_val:.2f}")
    axs[-1].set_xlabel('Time (t)')
    plt.show()

def Hamilton0(distance, n_rings, n_chains):
    """ Construct the system Hamiltonian including dipole interactions. """
    N_atoms = n_chains * n_rings
    Pos = cyl_positions(distance, N_atoms, n_chains)
    atom_frequencies = [omega_a]*N_atoms # sample_frequencies(omega_a, 0.0125 * omega_a, N_atoms)
    H = 0
    for a in range(N_atoms):
        for b in range(N_atoms):
            op = sm_list[a].dag() * sm_list[b]
            if a != b:
                H += alpha / (np.linalg.norm(Pos[a] - Pos[b]))**3 * op
            else:
                H += atom_frequencies[a] * op
    return H

def get_times_for_T(T, spacing=fine_spacing):
    # Adjust the first and last entries of times_tau
    first_tau = (Delta_ts[0] + Delta_ts[1])
    last_tau = np.floor((t_max - Delta_ts[2] - T - Delta_ts[0]) / spacing) * spacing
    times_tau = np.arange(first_tau, last_tau, spacing)
    # Adjust the first and last entries of times_t
    first_t = (Delta_ts[2])
    last_t = np.floor((t_max - Delta_ts[0] - T - 2 * Delta_ts[2]) / spacing) * spacing
    times_t = np.arange(first_t, last_t, spacing)

    return times_tau, times_t

def envelope(t, t0, Delta):
    E = np.cos(np.pi * (t - t0) / (2 * Delta))**2 * np.heaviside(t - (t0 - Delta), 0) * np.heaviside(t0 + Delta - t, 0)
    return E
def El_field(t, args):
    """ Define a time-dependent electric field pulse. """
    t0, Delta, omega, phi, E0 = args['time'], args['Delta'], args['omega'], args['phi'], args['E0']
    E  = np.exp(1j * (omega * t + phi))
    E *= envelope(t, t0, Delta)
    return  E0 * E
def El_field_op(t, args):
    E    = El_field(t, args)
    E_op = E + np.conj(E)
    return E_op

# =============================
# COLLAPSE AND EXPECT OPERATORS
# =============================
def c_ops():
    c_op_list = []

    rate = gamma_0      # Atomic decay operators
    if rate > 0.0:
       c_op_list.append([np.sqrt(rate) * op for op in sm_list]) # [np.sqrt(gamma_0) * Sigma_m]

    rate = gamma_phi    # Dephasing operators for the atoms
    if rate > 0.0:
        c_op_list.append([np.sqrt(rate) * commutator(op.dag(), op) for op in sm_list])  # [np.sqrt(gamma_phi) * Deph_op]

    return c_op_list
e_op_list = [ket2dm(basis(N_atoms + 1, i)) for i in range(N_atoms + 1)] # Expectation operators  |g> and |e>
wc = 100
#def ohmic_spectrum_at(w):
#    if w == 0.0: # dephasing inducing noise
#        return gamma_phi
#    else: # relaxation inducing noise
#        return gamma_phi * w/wc * np.exp(-w/wc) * (w > 0.0)
def ohmic_spectrum_at(w):
    if w == 0.0: # dephasing inducing noise
        return gamma_phi
    else: # relaxation inducing noise
        return gamma_0 * w/wc * np.exp(-w/wc) * (w > 0.0)
        return gamma_0 / 2 * (w / (2 * np.pi)) * (w > 0.0)

a_op_list = [[Dip_op, ohmic_spectrum_at],]


# =============================
# SYSTEM EVOLUTION
# =============================
def H_int(t, args):
    return - Dip_op * El_field_op(t, args)
test_args = {
        'phi': 0,
        'time': times[0] + Delta_ts[0],
        'omega': omegas[0],
        'Delta': Delta_ts[0],
        'E0': E0s[0]
    }
# Define System Hamiltonian
H0 = Hamilton0(distance, n_rings, n_chains)
H_sys = H0 + QobjEvo(H_int, args=test_args)

# initial state
psi_ini = ket2dm(atom_g)                                            # Initial state = ground state

## Calculations

### Compute Pulse
def compute_pulse(psi_ini, times, phi, i):
    progress_bar = ""
    if i == 2 and times[0] >= times[len(times)//2]:
        progress_bar='tqdm'               # Enable tqdm progress bar (or 'text', 'enhanced')
    options = Options(store_states=True,  # Save the states from the Qutip calculation
    progress_bar=progress_bar,)           # Enable tqdm progress bar (or 'text', 'enhanced')
    args = {
        'phi': phi,
        'time': times[0] + Delta_ts[i],
        'omega': omegas[i],
        'Delta': Delta_ts[i],
        'E0': E0s[i]
    }
    result = brmesolve(H_sys, psi_ini, times, e_ops=e_op_list, a_ops=a_op_list, args=args, options=options)
    #result = mesolve(H_sys, psi_ini, times, e_ops=e_op_list, c_ops=c_ops(), args=args, options=options)
    return result

### Compute Two-Dimensional Polarization
def compute_two_dimensional_polarization(T_val, phi_1, phi_2):
    # get the symmetric times, t / tau
    tau_values, t_values = get_times_for_T(T_val)

    # initialize the time domain Spectroscopy
    data = np.zeros((len(tau_values), len(t_values)))

    # only make the necessary steps (don't calculate too many states that we don't need)
    index_0 = np.abs(times - (tau_values[-1] - Delta_ts[1] + Delta_ts[0])).argmin()  # Find the closest index to reduce computation time
    # select range  ->  to reduce computation time
    times_0 = times[:index_0+1]

    # calculate the evolution of the first pulse in the desired range for tau
    data_1  = compute_pulse(psi_ini, times_0, phi_1, 0)

    # for every tau value -> calculate the next pulses
    for tau_idx, tau in enumerate(tau_values):
        # find the position in times, which corresponds to the current tau
        i       = np.abs((times_0 + Delta_ts[1] - Delta_ts[0]) - tau_values[tau_idx]).argmin()

        # take this state and evolve it with the second pulse, but only as far as desired
        psi_1   = data_1.states[i]

        # select range  ->  to reduce computation time
        j = np.abs((times - times_0[i] - Delta_ts[1] + Delta_ts[2]) - T_val).argmin()
        times_1 = times[i:j+1]

        # compute second pulse for waiting time T
        data_2  = compute_pulse(psi_1, times_1, phi_2, 1)
        psi_2   = data_2.states[j-i]

        # compute the last pulse with times t
        times_2 = times[j:]
        data_f  = compute_pulse(psi_2, times_2, 0, 2)

        for t_idx, t in enumerate(t_values):
            # store the data for the case
            if t_idx + tau_idx < len(tau_values):

                k = np.abs(t_values[t_idx] - (times_2 - times_2[0] - Delta_ts[2])).argmin()
                psi_f = data_f.states[k]
                data[tau_idx, t_idx] = expect(Dip_op, psi_f)

                # make one plot for this case
                if t == t_values[len(t_values)//2] and tau == tau_values[len(tau_values)//3]:
                    Plot_example_evo(T_val, tau, phi_1, phi_2, times_0, times_1, times_2, data_1, data_2, data_f, i, j, k)

    return tau_values, t_values, data

def process_phi_combination(phi_1, phi_2, times_T):
    # Preallocate an array to store results for each T in times_T
    full_data_array = np.empty((len(times_T)), dtype=object)

    # Using ProcessPoolExecutor to parallelize the inner loop
    with ProcessPoolExecutor() as executor:
        futures = []
        for i in range(len(times_T)):
            futures.append(executor.submit(compute_two_dimensional_polarization, times_T[i], phi_1, phi_2))

        # Collect results as they complete
        for i, future in enumerate(as_completed(futures)):
            full_data_array[i] = future.result()

    return full_data_array

# Function to handle parallel processing for each phi1, phi2 combination
def parallel_process_all_combinations(phases, times_T):
    all_results = {}

    # Use ProcessPoolExecutor for outer loop
    with ProcessPoolExecutor() as executor:
        futures = []

        # Submit all phi1, phi2 combinations as separate tasks
        for phi1, phi2 in itertools.product(phases, phases):
            future = executor.submit(process_phi_combination, phi1, phi2, times_T)
            futures.append((future, phi1, phi2))  # Store both the future and the corresponding (phi1, phi2)

        # Collect the results
        for future, phi1, phi2 in futures:
            all_results[(phi1, phi2)] = future.result()  # Store the result in the dictionary

    return all_results

# the time data can even be padded
def average_and_plot_results(results, times_T):
    num_combinations = len(results)  # Number of (phi1, phi2) pairs
    num_time_steps = len(times_T)-1  # Number of time indices

    # Initialize an array to store averaged data for each time step
    averaged_data_array = np.empty(num_time_steps, dtype=object)

    for i in range(num_time_steps):  # Loop over time indices
        sum_data = None  # Placeholder for summing data arrays
        ts, taus = None, None  # Will store time axes from any valid entry

        # Sum over all (phi1, phi2) combinations
        for _, full_data_array in results.items():
            ts, taus, data_time = full_data_array[i]  # Extract data
            if sum_data is None:
                sum_data = np.zeros_like(data_time)  # Initialize sum with correct shape

            sum_data += data_time  # Accumulate data

        # Compute the average
        averaged_data = sum_data / num_combinations

        # Now pad the averaged_data, ts, and taus
        # First, calculate the desired padded size
        pad_val = 1 #>= 1
        padded_size = (int(np.round(averaged_data.shape[0] * pad_val)), int(np.round(averaged_data.shape[1] * pad_val)))

        # Pad averaged_data to twice the size
        padded_averaged_data = np.pad(averaged_data,
                                      ((0, padded_size[0] - averaged_data.shape[0]),
                                       (0, padded_size[1] - averaged_data.shape[1])),
                                      mode='constant', constant_values=0)  # Use 0 for padding

        # Extend ts and taus to match the padded size
        padded_ts = np.linspace(ts[0], np.round(ts[-1] * pad_val), padded_size[1])
        padded_taus = np.linspace(taus[0], np.round(taus[-1] * pad_val), padded_size[0])

        # Store the padded results
        averaged_data_array[i] = (padded_ts, padded_taus, padded_averaged_data)

        # Optionally plot the padded results
        #plot_positive_color_map(padded_ts, padded_taus, padded_averaged_data, times_T[i])
    #plot_positive_color_map(averaged_data_array[1][1], averaged_data_array[1][0], averaged_data_array[1][2], times_T[i])

    return averaged_data_array  # Return for further processing if needed
def extend_and_transform_averaged_results(averaged_results):
    """ Extend the first data set (i=0) to all averaged results and compute FFT """

    # 1. Define global time and frequency axes based on the first dataset
    global_taus = averaged_results[0][1]  # Global taus (excitation time)
    global_ts = averaged_results[0][0]  # Global ts (detection time)

    # Compute frequency axes for FFT
    global_t_freqs = np.fft.fftfreq(len(global_ts), d=(global_ts[1] - global_ts[0]))
    global_tau_freqs = np.fft.fftfreq(len(global_taus), d=(global_taus[1] - global_taus[0]))

    # Initialize zero arrays for global time and frequency domain results
    global_data_time = np.zeros((len(global_taus), len(global_ts)))
    global_data_freq = np.zeros((len(global_taus), len(global_ts)), dtype=np.complex64)

    # 2. Iterate over the averaged results
    def find_closest_index(local_times, global_times):
        # Find the index in global_times for each time in local_times that is closest
        closest_indices = []
        for local_time in local_times:
            # Calculate the absolute differences
            differences = np.abs(global_times - local_time)
            # Find the index with the minimum difference
            closest_idx = np.argmin(differences)
            closest_indices.append(closest_idx)

        return closest_indices

    #print("Global ts:", global_ts[0], global_ts[1])
    #print("Global taus:", global_taus[0], global_taus[1])
    # Example usage
    for i, (ts, taus, data_time) in enumerate(averaged_results):
        # Create an extended data array matching global grid
        data_extended_time = np.zeros_like(global_data_time)

        # Find closest indices in global_taus and global_ts
        tau_indices = find_closest_index(taus, global_taus)
        t_indices = find_closest_index(ts, global_ts)

        # Now map the local data_time to the global grid
        for local_tau_idx, global_tau_idx in enumerate(tau_indices):
            for local_t_idx, global_t_idx in enumerate(t_indices):
                data_extended_time[global_tau_idx, global_t_idx] = data_time[local_tau_idx, local_t_idx]        # Compute FFT of the extended time data
        data_extended_freq = np.fft.fft2(data_extended_time)

        # Accumulate results
        global_data_time += data_extended_time
        global_data_freq += data_extended_freq

        # Plot intermediate results for each time step
        plot_positive_color_map(ts, taus, data_time, times_T[i], safe=True)
        plot_positive_color_map(global_ts, global_taus, data_extended_time, times_T[i], safe=True)
        plot_positive_color_map(global_t_freqs, global_tau_freqs, data_extended_freq, times_T[i], space= "freq", positive = True, type="abs", safe=True)
        plot_positive_color_map(global_ts, global_taus, global_data_time, times_T[i], safe=True)  # Time-space data

    global_data_time /= len(averaged_results)
    global_data_freq /= len(averaged_results)

    # 3. Plot final aggregated results
    plot_positive_color_map(global_ts, global_taus, global_data_time, safe=True)  # Time-space data
    plot_positive_color_map(global_t_freqs, global_tau_freqs, global_data_freq, space="freq", type="abs", positive=True, safe=True)  # Frequency-space data

    return global_data_time, global_data_freq  # Return for further processing if needed


def main():
    # Call the function to parallelize the whole process
    all_results = parallel_process_all_combinations(phases, times_T)

    # Iterate over all results and sort based on the length of tau_values
    sorted_results = {}

    for (phi1, phi2), result in all_results.items():

        sorted_result = sorted(result, key=lambda data_item: len(data_item[0]), reverse=True)
        sorted_result = sorted_result[:-1]
        sorted_results[(phi1, phi2)] = sorted_result

    averaged_results = average_and_plot_results(sorted_results, times_T)

    global_time_result, global_freq_result = extend_and_transform_averaged_results(averaged_results)


# If this script is executed directly, run the main function
if __name__ == "__main__":
    main()

