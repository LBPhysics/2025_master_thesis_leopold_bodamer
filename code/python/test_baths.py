import numpy as np
import matplotlib.pyplot as plt
from qutip import BosonicEnvironment
import os


# =============================
# TEST OUT BOSONIC BATHS
# =============================
def spectral_density_func_drude_lorentz(w, args):
    """
    Spectral density function for a Drude-Lorentz bath.
    """
    lambda_ = args["lambda"]  # Reorganization energy (coupling strength)
    gamma = args["cutoff"]  # Drude decay rate (cutoff frequency)

    return (2 * lambda_ * gamma * w) / (w**2 + gamma**2)


def spectral_density_func_ohmic(w, args):
    """
    Spectral density function for an ohmic bath.
    """
    wc = args["cutoff"]
    eta = args["eta"]
    s = args["s"]
    return eta * w**s / wc ** (s - 1) * np.exp(-w / wc) * (w > 0)


def Power_spectrum_func_ohmic(w, args):
    """
    Power spectrum function in the frequency domain for an ohmic bath.
    Handles both positive and negative frequencies.
    """
    coth_term = 1 / np.tanh(w / (2 * Boltzmann * T))
    return np.sign(w) * spectral_density_func_ohmic(np.abs(w), args) * (coth_term + 1)


def n(w, T):
    """
    Bose-Einstein distribution function.
    """
    w = np.asarray(w)  # Ensure w is a NumPy array
    result = np.zeros_like(w)  # Initialize result array with zeros

    # Avoid division by zero for w == 0
    nonzero_mask = w != 0
    large_mask = (hbar * w[nonzero_mask] / (Boltzmann * T)) > 700  # Avoid overflow
    result[nonzero_mask & ~large_mask] = 1 / (
        np.exp(hbar * w[nonzero_mask & ~large_mask] / (Boltzmann * T)) - 1
    )
    result[nonzero_mask & large_mask] = 0  # Approximation for large values

    return result


# =============================
# BATH FUNCTIONS
# =============================
def spectral_density_func_paper(w, args):
    """
    Spectral density function for a bath as given in the paper.
    """
    g = args["g"]
    cutoff = args["cutoff"]
    return g**2 * (w / cutoff) * np.exp(-w / cutoff) * (w > 0)


def Power_spectrum_func_paper(w, args):
    """
    Power spectrum function in the frequency domain as given in the paper.
    Handles both positive and negative frequencies.
    """
    # Create an array to store the results
    result = np.zeros_like(w)

    # Positive frequencies
    positive_mask = w > 0
    result[positive_mask] = spectral_density_func_paper(w[positive_mask], args) * n(
        w[positive_mask], T
    )

    # Negative frequencies
    negative_mask = w < 0
    result[negative_mask] = spectral_density_func_paper(-w[negative_mask], args) * (
        1 + n(-w[negative_mask], T)
    )

    return result


# =============================
# TESTING FUNCTIONS
# =============================
def plot_bath_with_qutip_from_f(
    function,
    args,
    frequencies_range=(-25, 25),
    num_points=1000,
    func_name=None,
    bath=None,
):
    """
    Plot the spectral density, power spectrum, and correlation function for a bath using QuTiP.

    Parameters:
        function (function): Function defining the spectral density.
        args (dict): Arguments for the spectral density function.
        T (float): Temperature.
        frequencies_range (tuple): Range of frequencies to evaluate.
        num_points (int): Number of points for frequency and time ranges.

    Returns:
        matplotlib.figure.Figure: The generated figure containing the plots.
    """

    cutoff = args["cutoff"]  # Cutoff frequency

    # ============================= Define frequency and time ranges
    frequencies = np.linspace(
        frequencies_range[0] * cutoff, frequencies_range[1] * cutoff, num_points
    )  # Frequency range
    times = np.linspace(
        frequencies_range[0] / cutoff, frequencies_range[1] / cutoff, num_points
    )  # Time range
    normalized_frequencies = frequencies / cutoff  # Normalize frequencies
    normalized_times = times * cutoff  # Normalize time by wc

    ### Generate the bosonic environment
    if func_name == "J":
        if bath == "ohmic":
            env = BosonicEnvironment.from_spectral_density(
                lambda w: function(w, args), wMax=10 * cutoff, T=T
            )
        else:
            env = BosonicEnvironment.from_spectral_density(
                lambda w: function(w, args),  # - function(-w, args),
                wMax=10 * cutoff,
                T=T,
            )
    elif func_name == "S":
        env = BosonicEnvironment.from_power_spectrum(
            lambda w: function(w, args), wMax=10 * cutoff, T=T
        )
    else:
        print(
            "Invalid function type. Use 'J' for spectral density or 'S' for power spectrum."
        )
        return

    ### Calculate spectral density, power spectrum, and correlation function
    spectral_density_vals = env.spectral_density(frequencies)  # Spectral density
    power_spectrum_vals = env.power_spectrum(frequencies)  # Power spectrum
    correlation_vals = env.correlation_function(times)  # Correlation function

    max_J = np.max(np.abs(spectral_density_vals))
    max_P = np.max(np.abs(power_spectrum_vals))
    max_corr = np.max(np.abs(correlation_vals))

    # Handle cases where the maximum values are zero to avoid division by zero
    if max_J <= 1e-10:
        max_J = 1
    if max_P <= 1e-10:
        max_P = 1
    if max_corr <= 1e-10:
        max_corr = 1

    # =============================
    # PLOTTING RESULTS
    # =============================
    ### Create subplots for spectral density, power spectrum, and correlation function
    axes = []

    plt.suptitle(
        r"Bath at $\beta$ = "
        + f"{1/(Boltzmann * T):.2f}"
        + " with QuTiP from $"
        + func_name
        + r"(\omega)$"
    )
    ### Plot Spectral Density and Power Spectrum
    ax1 = plt.subplot(2, 1, 1)
    axes.append(ax1)
    ax1.plot(
        normalized_frequencies,
        spectral_density_vals / max_J,
        label=r"$J(\omega)$ q",
        color="C0",
        linestyle="solid",
    )
    ax1.plot(
        normalized_frequencies,
        power_spectrum_vals / max_P,
        label=r"$S(\omega)$ q",
        color="C1",
        linestyle="dashed",
    )
    ax1.set_xlabel(r"$\omega / \tilde{\omega}_{\text{c}}$")
    ax1.set_ylabel(r"$f / f(\tilde{\omega}_{\text{c}})$")
    ax1.set_title(r"Spectral Density $ J $ and Power Spectrum $ S $")
    ax1.legend()

    ### Plot Correlation Function over t / wc
    ax2 = plt.subplot(2, 1, 2)
    axes.append(ax2)
    ax2.plot(
        normalized_times,
        np.real(correlation_vals) / max_corr,
        label=r"$\mathrm{Re}[C(t)]$ q",
        linestyle="dotted",
        color="C2",
    )
    ax2.plot(
        normalized_times,
        np.imag(correlation_vals) / max_corr,
        label=r"$\mathrm{Im}[C(t)]$ q",
        linestyle="dashed",
        color="C3",
    )
    ax2.plot(
        normalized_times,
        np.abs(correlation_vals) / max_corr,
        label=r"$|C(t)|$ q",
        linestyle="solid",
        color="C4",
    )
    ax2.set_xlabel(r"Time $t \tilde{\omega}_{\text{c}}$")
    ax2.set_ylabel(r"$f / \text{max}|C|$")
    ax2.set_title(r"Correlation Function $ C $")
    ax2.legend(loc="upper right")

    plt.tight_layout()

    return axes


def plot_bath_from_paper_with_paper(args, frequencies_range=(-25, 25), num_points=1000):
    """
    Plot the spectral density, power spectrum, and correlation function for the bath as described in the paper.

    Parameters:
        args (dict): Arguments for the spectral density function from the paper.

    Returns:
        list: List of axes for the generated plots.
    """

    # Define frequency and time ranges
    cutoff = args["cutoff"]  # Cutoff frequency

    # ============================= Define frequency and time ranges
    frequencies = np.linspace(
        frequencies_range[0] * cutoff, frequencies_range[1] * cutoff, num_points
    )  # Frequency range
    normalized_frequencies = frequencies / cutoff  # Normalize frequencies

    # Generate the environment the way the paper does it and calculate Power Spectrum values
    spectral_density_vals = np.array(
        [spectral_density_func_paper(w, args) for w in frequencies]
    )
    power_spectrum_vals = np.array(
        [Power_spectrum_func_paper(w, args) for w in frequencies]
    )

    # Fourier Transform of Power Spectrum
    def compute_C_t(S_w, frequencies):
        """
        Computes C(t) using the inverse Fourier transform from given S(w) and frequency values.

        Parameters:
        S_w (numpy array): Corresponding values of S(omega)
        frequencies (numpy array): Array of frequency values (omega)

        Returns:
        C_t (numpy array): Computed values of C(t)
        t_values (numpy array): Time values corresponding to C(t)
        """
        # Determine frequency step (assuming uniform spacing)
        delta_w = frequencies[1] - frequencies[0]

        # Define the time domain
        N = len(frequencies)  # Number of points
        t_values = np.fft.fftshift(np.fft.fftfreq(N, d=delta_w / (2 * np.pi)))

        # Compute inverse Fourier transform using FFT
        C_t = (
            np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(S_w))) * len(S_w) / (2 * np.pi)
        )

        return C_t, t_values

    # Use the custom Fourier Transform function
    correlation_vals, times = compute_C_t(power_spectrum_vals, frequencies)
    normalized_times = times * cutoff  # Normalize time by wc

    max_J = np.max(np.abs(spectral_density_vals))
    max_P = np.max(np.abs(power_spectrum_vals))
    max_corr = np.max(np.abs(correlation_vals))

    # Handle cases where the maximum values are zero to avoid division by zero
    if max_J <= 1e-10:
        max_J = 1
    if max_P <= 1e-10:
        max_P = 1
    if max_corr <= 1e-10:
        max_corr = 1
    axes_paper = []

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(
        r"Bath from Paper  $\beta$ = " + f"{1/(Boltzmann * T):.2f}" + " with Paper"
    )

    # Plot J and Power Spectrum S
    ax1.plot(
        normalized_frequencies,
        spectral_density_vals / max_J,
        label=r"$J(\omega)$ p",
        color="C2",
        linestyle="solid",
    )
    ax1.plot(
        normalized_frequencies,
        power_spectrum_vals / max_P,
        label=r"$S(\omega)$ p",
        color="C3",
        linestyle="dashed",
    )
    ax1.set_xlabel(r"$\omega / \tilde{\omega}_{\text{c}}$")
    ax1.set_ylabel(r"$f / f(\tilde{\omega}_{\text{c}})$")
    ax1.set_title(r"Spectral Density $ J $ and Power Spectrum $ S $")
    ax1.legend()
    axes_paper.append(ax1)

    # Plot Fourier Transform of Power Spectrum
    # Clip the data to the range of -25 to 25 for better visualization
    clip_mask = (normalized_times >= -25) & (normalized_times <= 25)
    clipped_times = normalized_times[clip_mask]
    clipped_correlation_vals = correlation_vals[clip_mask]
    ax2.plot(
        clipped_times,
        np.real(clipped_correlation_vals) / max_corr,
        label=r"$\mathrm{Re}[C(t)]$ p",
        linestyle="dotted",
        color="C0",
    )
    ax2.plot(
        clipped_times,
        np.imag(clipped_correlation_vals) / max_corr,
        label=r"$\mathrm{Im}[C(t)]$ p",
        linestyle="dashed",
        color="C1",
    )
    ax2.plot(
        clipped_times,
        np.abs(clipped_correlation_vals) / max_corr,
        label=r"$|C(t)|$ p",
        linestyle="solid",
        color="C5",
    )
    ax2.set_xlabel(r"Time $t \tilde{\omega}_{\text{c}}$")
    ax2.set_ylabel(r"$f / \text{max}|C|$")
    ax2.set_title(r"Correlation Function $ C $ (Clipped)")
    ax2.legend(loc="upper right")
    axes_paper.append(ax2)

    plt.tight_layout()
    return axes_paper


def main():
    """
    Main function to test the plot_bath_with_qutip_from_f function with different spectral density and power spectrum functions.
    """
    # Define constants
    global Boltzmann, hbar, T
    Boltzmann = 1  # Boltzmann constant in J/K
    hbar = 1  # Reduced Planck's constant in J·s
    T = 1e0  # Temperature in Kelvin
    plt.rcParams.update(
        {
            "text.usetex": True,  # Enable LaTeX for text rendering
            "font.family": "serif",  # Use a serif font family
            "font.serif": [],  #'Palatino' or []  Set Palatino or standard latex font
            "text.latex.preamble": r"\usepackage{amsmath}",
            "font.size": 20,  # Font size for general text
            "axes.titlesize": 20,  # Font size for axis titles
            "axes.labelsize": 20,  # Font size for axis labels
            "xtick.labelsize": 20,  # Font size for x-axis tick labels
            "ytick.labelsize": 20,  # Font size for y-axis tick labels
            "legend.fontsize": 20,  # Font size for legends
            "figure.figsize": [8, 6],  # Size of the plot (width x height)
            "figure.autolayout": True,  # Automatic layout adjustment
            "savefig.format": "svg",  # Default format for saving figures
            "figure.facecolor": "none",  # Make the figure face color transparent
            "axes.facecolor": "none",  # Make the axes face color transparent
            "savefig.transparent": True,  # Save figures with transparent background
        }
    )
    # Define the output directory relative to the main directory of the repository
    repo_root_dir = os.path.abspath(
        os.path.join(os.getcwd(), "../../")
    )  # Navigate to the main directory
    output_dir = os.path.join(
        repo_root_dir, "figures", "figures_from_python"
    )  # Define the output folder path
    os.makedirs(output_dir, exist_ok=True)

    # Define arguments for the baths
    eta, cutoff = 0.1, 1e3
    args_ohmic = {"eta": eta, "cutoff": cutoff, "s": 1.0}
    args_paper = {"g": np.sqrt(eta * cutoff), "cutoff": cutoff}
    args_drude_lorentz = {"lambda": eta * cutoff / 2, "cutoff": cutoff}

    # Test the function with different baths
    """
    axs0 = plot_bath_from_paper_with_paper(args_paper)
    axs1 = plot_bath_with_qutip_from_f(
        spectral_density_func_paper, args_paper, func_name="J", bath="paper"
    )
    axs1_ = plot_bath_with_qutip_from_f(
        Power_spectrum_func_paper, args_paper, func_name="S", bath="paper"
    )

    axs2_ = plot_bath_with_qutip_from_f(
        Power_spectrum_func_ohmic, args_ohmic, func_name="S", bath="ohmic"
    )
    """

    axs2 = plot_bath_with_qutip_from_f(
        spectral_density_func_ohmic, args_ohmic, func_name="J", bath="ohmic"
    )
    axs3 = plot_bath_with_qutip_from_f(
        spectral_density_func_drude_lorentz,
        args_drude_lorentz,
        func_name="J",
        bath="drude_lorentz",
    )

    # plt.savefig("bath_comparison_Paper_Qutip.svg", dpi=300, bbox_inches='tight')
    # plt.savefig("bath_Qutip_from_J_S.png", dpi=100, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":

    main()
