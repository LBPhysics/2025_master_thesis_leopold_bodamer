import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import sys
import os

# Add the qspectro2d package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from qspectro2d.core.bath_system.bath_fcts import power_spectrum_func_ohmic

"""
Script to optimize bath parameters (temp, alpha) for ohmic bath to achieve
target power spectrum values at specific frequencies.

Target: 
- power_spectrum_func_ohmic(0) ≈ 1/100 = 0.01
- power_spectrum_func_ohmic(w0) ≈ 1/300 ≈ 0.00333
"""


# OPTIMIZATION PARAMETERS


### Target values
target_at_zero = 1 / 10000  # Target value at w=0
target_at_w0 = 1 / 300  # Target value at w=w0
w0 = 3.01  # Frequency where we want the second target

### Fixed bath parameters
cutoff = 1e2 * w0  # Cutoff frequency for ohmic bath
s = 1.0  # Ohmic exponent (s=1 for ohmic)
Boltzmann = 1.0  # Boltzmann constant (normalized)
hbar = 1.0  # Reduced Planck constant (normalized)


def objective_function(params):
    """
    Objective function to minimize the squared difference between
    actual and target power spectrum values.

    Parameters:
    -----------
    params : array-like
        [temp, alpha] - temperature and coupling strength

    Returns:
    --------
    float
        Sum of squared errors
    """
    temp, alpha = params

    # Ensure positive parameters
    if temp <= 0 or alpha <= 0:
        return 1e10  # Large penalty for invalid parameters

    # Create arguments dictionary
    args = {
        "temp": temp,
        "alpha": alpha,
        "cutoff": cutoff,
        "s": s,
        "Boltzmann": Boltzmann,
        "hbar": hbar,
    }

    try:
        # Calculate power spectrum at target frequencies
        ps_at_zero = power_spectrum_func_ohmic(0.0, **args)
        ps_at_w0 = power_spectrum_func_ohmic(w0, **args)

        # Calculate squared errors
        error_zero = (ps_at_zero - target_at_zero) ** 2
        error_w0 = (ps_at_w0 - target_at_w0) ** 2

        total_error = error_zero + error_w0

        return total_error

    except Exception as e:
        print(f"Error in objective function: {e}")
        return 1e10


def optimize_bath_parameters(initial_guess=None, method="L-BFGS-B", use_multiple_starts=True):
    """
    Optimize bath parameters to achieve target power spectrum values.
    Uses multiple random starting points to explore the wide parameter space.

    Parameters:
    -----------
    initial_guess : array-like, optional
        Initial guess for [temp, alpha]. If None, uses multiple random starts.
    method : str
        Optimization method for scipy.minimize
    use_multiple_starts : bool
        Whether to use multiple random starting points

    Returns:
    --------
    dict
        Best optimization result from all attempts
    """

    ### Set bounds for parameters (very wide range)
    bounds = [
        (1e-10, 1000.0),  # Temperature bounds
        (1e-10, 1000.0),  # Alpha bounds
    ]

    print("Starting optimization with wide parameter exploration...")
    print(f"Target at w=0: {target_at_zero:.6f}")
    print(f"Target at w={w0}: {target_at_w0:.6f}")
    print(
        f"Parameter bounds: temp=[{bounds[0][0]:.1e}, {bounds[0][1]:.1e}], alpha=[{bounds[1][0]:.1e}, {bounds[1][1]:.1e}]"
    )

    best_result = None
    best_error = float("inf")

    if use_multiple_starts and initial_guess is None:
        # Generate multiple random starting points across log space
        n_starts = 20
        print(f"\nUsing {n_starts} random starting points to explore parameter space...")

        # Generate logarithmically distributed starting points
        temp_starts = np.logspace(-8, 2, n_starts // 4)  # 1e-8 to 100
        alpha_starts = np.logspace(-6, 1, n_starts // 4)  # 1e-6 to 10

        # Create grid of starting points
        starting_points = []
        for temp in temp_starts:
            for alpha in alpha_starts:
                starting_points.append([temp, alpha])

        # Add some additional random points
        for _ in range(10):
            temp_rand = np.random.uniform(bounds[0][0], bounds[0][1])
            alpha_rand = np.random.uniform(bounds[1][0], bounds[1][1])
            starting_points.append([temp_rand, alpha_rand])

    else:
        # Use single starting point
        if initial_guess is None:
            initial_guess = [0.1, 0.1]  # [temp, alpha]
        starting_points = [initial_guess]

    # Try optimization from each starting point
    for i, start_point in enumerate(starting_points):
        print(
            f"\nAttempt {i+1}/{len(starting_points)}: temp={start_point[0]:.2e}, alpha={start_point[1]:.2e}"
        )

        try:
            result = minimize(
                objective_function,
                start_point,
                method=method,
                bounds=bounds,
                options={"disp": False, "maxiter": 1000},
            )

            current_error = result.fun
            print(f"  -> Final error: {current_error:.2e}, Success: {result.success}")

            if current_error < best_error:
                best_error = current_error
                best_result = result
                print(f"  -> New best result! Error: {best_error:.2e}")

        except Exception as e:
            print(f"  -> Failed: {e}")
            continue

    if best_result is None:
        print("All optimization attempts failed!")
        return None

    print(f"\n{'='*60}")
    print(f"BEST OPTIMIZATION RESULT")
    print(f"{'='*60}")
    print(f"Best error: {best_error:.2e}")
    print(f"Success: {best_result.success}")
    print(f"Final parameters: temp={best_result.x[0]:.2e}, alpha={best_result.x[1]:.2e}")

    return best_result


def debug_power_spectrum(temp, alpha):
    """
    Debug function to analyze the power spectrum calculation step by step.

    Parameters:
    -----------
    temp : float
        Temperature
    alpha : float
        Coupling strength
    """

    args = {
        "temp": temp,
        "alpha": alpha,
        "cutoff": cutoff,
        "s": s,
        "Boltzmann": Boltzmann,
        "hbar": hbar,
    }

    print(f"\n{'='*50}")
    print(f"DEBUGGING POWER SPECTRUM")
    print(f"{'='*50}")
    print(f"Parameters: temp={temp:.2e}, alpha={alpha:.2e}, cutoff={cutoff:.2e}")

    # Test at w=0
    ps_zero = power_spectrum_func_ohmic(0.0, **args)
    print(f"\nAt w=0:")
    print(f"  Power spectrum: {ps_zero:.6e}")
    print(f"  Target: {target_at_zero:.6e}")
    print(f"  Error: {abs(ps_zero - target_at_zero):.6e}")

    # Test at w=w0
    ps_w0 = power_spectrum_func_ohmic(w0, **args)
    print(f"\nAt w={w0}:")
    print(f"  Power spectrum: {ps_w0:.6e}")
    print(f"  Target: {target_at_w0:.6e}")
    print(f"  Error: {abs(ps_w0 - target_at_w0):.6e}")

    # Test thermal parameters
    w_th = Boltzmann * temp / hbar
    print(f"\nThermal frequency w_th = kT/ħ: {w_th:.6e}")
    print(f"Ratio w0/w_th: {w0/w_th if w_th > 0 else 'inf':.2f}")

    return ps_zero, ps_w0


def evaluate_solution(temp_opt, alpha_opt):
    """
    Evaluate the optimized solution and display results.

    Parameters:
    -----------
    temp_opt : float
        Optimized temperature
    alpha_opt : float
        Optimized coupling strength
    """

    args = {
        "temp": temp_opt,
        "alpha": alpha_opt,
        "cutoff": cutoff,
        "s": s,
        "Boltzmann": Boltzmann,
        "hbar": hbar,
    }

    # Calculate power spectrum at target frequencies
    ps_at_zero = power_spectrum_func_ohmic(0.0, **args)
    ps_at_w0 = power_spectrum_func_ohmic(w0, **args)

    print("\n" + "=" * 50)
    print("OPTIMIZATION RESULTS")
    print("=" * 50)
    print(f"Optimal temperature: {temp_opt:.6f}")
    print(f"Optimal alpha: {alpha_opt:.6f}")
    print(f"\nPower spectrum values:")
    print(f"At w=0: {ps_at_zero:.6f} (target: {target_at_zero:.6f})")
    print(f"At w={w0}: {ps_at_w0:.6f} (target: {target_at_w0:.6f})")
    print(f"\nRelative errors:")
    if target_at_zero > 0:
        print(f"At w=0: {abs(ps_at_zero - target_at_zero)/target_at_zero*100:.2f}%")
    else:
        print(f"At w=0: N/A (target is zero)")
    if target_at_w0 > 0:
        print(f"At w={w0}: {abs(ps_at_w0 - target_at_w0)/target_at_w0*100:.2f}%")
    else:
        print(f"At w={w0}: N/A (target is zero)")

    # Add debugging info
    debug_power_spectrum(temp_opt, alpha_opt)


def plot_power_spectrum(temp_opt, alpha_opt, w_range=None):
    """
    Plot the power spectrum function with optimized parameters.

    Parameters:
    -----------
    temp_opt : float
        Optimized temperature
    alpha_opt : float
        Optimized coupling strength
    w_range : tuple, optional
        Frequency range for plotting (w_min, w_max)
    """

    if w_range is None:
        w_range = (-0.1 * w0, 2.0 * w0)

    args = {
        "temp": temp_opt,
        "alpha": alpha_opt,
        "cutoff": cutoff,
        "s": s,
        "Boltzmann": Boltzmann,
        "hbar": hbar,
    }

    # Create frequency array
    w_vals = np.linspace(w_range[0], w_range[1], 10001)
    ps_vals = [power_spectrum_func_ohmic(w, **args) for w in w_vals]

    plt.figure(figsize=(10, 6))
    plt.plot(w_vals, ps_vals, label=r"$S(\omega)$ optimized", color="C0", linewidth=2)

    # Mark target points
    plt.axhline(
        y=target_at_zero,
        color="C1",
        linestyle="dashed",
        label=f"Target at $\omega=0$: {target_at_zero:.4f}",
    )
    plt.axhline(
        y=target_at_w0,
        color="C2",
        linestyle="dashed",
        label=f"Target at $\omega={w0}$: {target_at_w0:.4f}",
    )

    # Mark actual values at target frequencies
    ps_zero = power_spectrum_func_ohmic(0.0, **args)
    ps_w0 = power_spectrum_func_ohmic(w0, **args)
    plt.scatter([0], [ps_zero], color="C1", s=100, zorder=5)
    plt.scatter([w0], [ps_w0], color="C2", s=100, zorder=5)

    plt.xlabel(r"Frequency $\omega$")
    plt.ylabel(r"Power Spectrum $S(\omega)$")
    plt.title(
        f"Optimized Ohmic Bath Power Spectrum\n"
        f"$T={temp_opt:.4f}$, $\\alpha={alpha_opt:.4f}$, $\omega_c={cutoff}$"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # MAIN OPTIMIZATION

    ### Run optimization
    result = optimize_bath_parameters()

    if result.success:
        temp_optimal, alpha_optimal = result.x

        ### Evaluate and display results
        evaluate_solution(temp_optimal, alpha_optimal)

        ### Plot the power spectrum
        plot_power_spectrum(temp_optimal, alpha_optimal)

    else:
        print("Optimization failed!")
        print(f"Message: {result.message}")
        print(f"Final parameters: {result.x}")

        # Still evaluate the final result
        if len(result.x) == 2:
            evaluate_solution(result.x[0], result.x[1])

    print(f"Final temperature: {result.x[0]:.6f}, Final alpha: {result.x[1]:.6f}")
