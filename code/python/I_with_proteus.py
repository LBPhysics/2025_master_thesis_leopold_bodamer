# Import the outsourced settings / functions
from functions_for_both_cases import *

import numpy as np
import psutil  # -> estimate RAM usage
import copy  # -> copy classes
import time  # -> estimate elapsed time

import pickle  # -> safe data
import sys  # safe data
import os  # -> safe data

# =============================
# SYSTEM PARAMETERS     (**changeable**)
# =============================

### Phase Cycling for Averaging
phases = [k * np.pi / 2 for k in range(4)]


def main():
    start_time = time.time()  # Start timing

    """
    Main function to run the script.
    """
    # =============================
    # SYSTEM PARAMETERS
    # =============================
    system = SystemParameters(
        N_atoms=1,
        ODE_Solver="Paper_eqs",
        RWA_laser=True,
        Delta_cm=200.0,
        omega_A_cm=16000.0,
        mu_eg_cm=1.0,
        omega_laser_cm=16000.0,
        E0=0.1,
        pulse_duration=15.0,
        t_max=100.0,  # -> determines Δω ∝ 1/t_max
        fine_spacing=0.5,  # -> determines ω_max ∝ 1/Δt
        gamma_0=1 / 300,
        T2=100.0,
    )

    system.summary()

    t_max = system.t_max
    fine_spacing_test = system.fine_spacing

    Delta_ts = system.Delta_ts
    times = np.arange(-Delta_ts[0], t_max, fine_spacing_test)
    # print("times: ", times[0], times[1], "...", times[-1], "len", len(times))

    # =============================
    test_params_copy = copy.deepcopy(system)
    if "time_cut" not in globals() or test_params_copy.t_max != system.t_max:
        # =============================
        # ALWAYS CHECK Before running a serious simulation
        # =============================
        test_params_copy.t_max = 10 * t_max
        test_params_copy.fine_spacing = 10 * fine_spacing_test
        times_test_ = np.arange(
            -Delta_ts[0], test_params_copy.t_max, test_params_copy.fine_spacing
        )
        result, time_cut = check_the_solver(times_test_, test_params_copy)
        print("the evolution is actually unphisical after:", time_cut, "fs")

    T_wait = times[-1] / 100
    times_T = [T_wait]

    two_d_datas = parallel_process_all_combinations(
        phases,
        times_T,
        times=times,
        system=system,
    )

    # =============================
    # Set output directory to a subfolder named 'output'
    # =============================

    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    # Construct base filename for saving
    base_filename = f"two_d_data_tmax{system.t_max}_spacing{system.fine_spacing}.pkl"
    save_path = os.path.join(output_dir, base_filename)

    counter = 1
    while os.path.exists(save_path):
        save_path = os.path.join(
            output_dir,
            f"two_d_data_tmax{system.t_max}_spacing{system.fine_spacing}_{counter}.pkl",
        )
        counter += 1

    with open(save_path, "wb") as f:
        pickle.dump({"two_d_datas": two_d_datas, "times_T": times_T, "times": times}, f)
        print(f"2D data and times_T and times saved to {save_path}")

    """
    with open(save_path, "rb") as f:
        data = pickle.load(f)
        two_d_datas = data["two_d_datas"]
        times_T = data["times_T"]
        times = data["times"]
        print(two_d_datas)
    """
    # =============================
    # Print elapsed time and memory usage for performance monitoring
    # =============================
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60
    print(f"Execution time: {hours}h {minutes}m {seconds:.2f}s")

    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss  # Resident Set Size: memory in bytes
    mem_mb = mem_bytes / (1024**2)  # Convert to MB
    print(f"Approximate memory usage: {mem_mb:.2f} MB")


if __name__ == "__main__":
    main()
