# Import the outsourced settings / functions
from functions2DES import *

import numpy as np
import psutil  # -> estimate RAM usage
import copy  # -> copy classes
import time  # -> estimate elapsed time

import pickle  # -> safe data
import sys  # safe data
import os  # -> safe data


### Phase Cycling for Averaging
phases = [k * np.pi / 2 for k in range(4)]


def main():
    """
    Main function to run the script.
    """
    start_time = time.time()  # Start timing

    # =============================
    # SYSTEM PARAMETERS
    # =============================
    N_sample = (
        1  # inhomogenity (averaging over how many frequencies) -> the bigger the better
    )
    system = SystemParameters(
        # WHICH SYSTEM:
        N_atoms=1,
        ODE_Solver="Paper_eqs",
        RWA_laser=True,
        # SIMULATION:
        t_max=20.0,  # -> determines Δω
        fine_spacing=0.1,  # -> determines ω_max
        # NECESSARY FOR THE DELAYED PHOTON EFFECT:
        Delta_cm=200 if N_sample > 1 else 0,
    )
    t_max = system.t_max
    fine_spacing_test = system.fine_spacing

    Delta_ts = system.Delta_ts
    times = np.arange(-Delta_ts[0], t_max, fine_spacing_test)

    T_wait = times[-1] / 100
    times_T = [T_wait]  # -> wait times for the 2D spectrum

    # system.summary()

    # =============================
    test_params_copy = copy.deepcopy(system)
    # =============================
    # ALWAYS CHECK Before running a serious simulation
    # =============================
    test_params_copy.t_max = 10 * t_max
    test_params_copy.fine_spacing = 10 * fine_spacing_test
    times_test_ = np.arange(
        -Delta_ts[0], test_params_copy.t_max, test_params_copy.fine_spacing
    )
    _, time_cut = check_the_solver(times_test_, test_params_copy)
    print("\nthe evolution is actually unphysical after:", time_cut, "fs")

    omega_ats = sample_from_sigma(
        N_sample, system.Delta_cm, system.omega_A_cm, E_range=3
    )
    print(omega_ats)
    kwargs = {"plot_example": False}

    # DO THE SIMULATION
    two_d_datas = parallel_process_all_combinations_with_inhomogenity(
        omega_ats=omega_ats,
        phases=phases,
        times_T=times_T,
        times=times,
        system=system,
        kwargs=kwargs,
    )
    """
    # =============================
    # Set output directory to a subfolder named 'output'
    # =============================

    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "papers_with_proteus_output"
        )

    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    # Construct base filename for saving
    base_filename = f"data_for_tmax_{system.t_max:.0f}_dt_{system.fine_spacing}.pkl"
    save_path = os.path.join(output_dir, base_filename)

    counter = 1
    while os.path.exists(save_path):
        save_path = os.path.join(
            output_dir,
            f"data_for_tmax_{system.t_max:.0f}_dt_{system.fine_spacing}_{counter}.pkl",
        )
        counter += 1

    with open(save_path, "wb") as f:
        pickle.dump(
            {
                "system": system,
                "times": times,  # maybe not needed
                "times_T": times_T,
                "two_d_datas": two_d_datas,
            },
            f,
        )
        print(
            f"\n2D system info, times_T, times and the spectral data was saved to {save_path}"
        )
    
    # =============================
    # Print elapsed time and memory usage for performance monitoring
    # =============================
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60
    print(f"\nExecution time: {hours}h {minutes}m {seconds:.2f}s")

    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss  # Resident Set Size: memory in bytes
    mem_mb = mem_bytes / (1024**2)  # Convert to MB
    bound_mem_my_estimate = (
        2 * len(times_T) * len(phases) ** 2 * len(times) ** 2 * 8 / (1024**2)
    )  # 8 bytes per complex number
    print(
        f"\nApproximate memory usage: {mem_mb:.2f} MB vs my estimate: {bound_mem_my_estimate:.2f} MB"
    )
    """


if __name__ == "__main__":
    main()
