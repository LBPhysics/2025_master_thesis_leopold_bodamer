def H_int(
    t: float,
    pulse_seq: PulseSequence,
    system: SystemParameters,
) -> Qobj:
    """
    Define the interaction Hamiltonian for the system with multiple pulses using the PulseSequence class.

    Parameters:
        t (float): Time at which the interaction Hamiltonian is evaluated.
        pulse_seq (PulseSequence): PulseSequence object containing all pulse parameters.
        system (SystemParameters): System parameters.
        SM_op (Qobj): Lowering operator (system-specific).
        Dip_op (Qobj): Dipole operator (system-specific).

    Returns:
        Qobj: Interaction Hamiltonian at time t.
    """
    if not isinstance(pulse_seq, PulseSequence):
        raise TypeError("pulse_seq must be a PulseSequence instance.")

    SM_op = system.SM_op
    Dip_op = system.Dip_op

    if system.RWA_laser:
        E_field = E_pulse(t, pulse_seq)  # Combined electric field under RWA
        H_int = -(
            SM_op.dag() * E_field + SM_op * np.conj(E_field)
        )  # RWA interaction Hamiltonian
    else:
        E_field = Epsilon_pulse(t, pulse_seq)  # Combined electric field with carrier
        H_int = -Dip_op * (E_field + np.conj(E_field))  # Full interaction Hamiltonian

    return H_int


def apply_RWA_phase_factors(
    rho: Qobj, t: float, omega: float, system: SystemParameters
) -> Qobj:
    """
    Apply time-dependent phase factors to the density matrix entries.
    Dispatches to the appropriate implementation based on N_atoms.

    Parameters:
        rho (Qobj): Density matrix (Qobj) to modify.
        t (float): Current time.
        omega (float): Frequency of the phase factor.
        system (SystemParameters): System parameters.

    Returns:
        Qobj: Modified density matrix with phase factors applied.
    """
    if system.N_atoms == 1:
        return _apply_RWA_phase_factors_1atom(rho, t, omega)
    elif system.N_atoms == 2:
        return _apply_RWA_phase_factors_2atom(rho, t, omega)
    else:
        raise ValueError("Only N_atoms=1 or 2 are supported.")


def _apply_RWA_phase_factors_1atom(rho: Qobj, t: float, omega: float) -> Qobj:
    """
    Apply time-dependent phase factors to the density matrix entries.

    Parameters:
        rho (Qobj): Density matrix (Qobj) to modify.
        omega (float): Frequency of the phase factor.
        t (float): Current time.

    Returns:
        Qobj: Modified density matrix with phase factors applied.
    """
    # Extract the density matrix as a NumPy array
    rho_array = rho.full()
    # print(rho.isherm)

    # Apply the phase factors to the specified elements
    phase_1 = np.exp(-1j * omega * t)  # e^(-i * omega * t)

    # Modify the elements
    rho_array[1, 0] *= phase_1  # rho_alpha_0 = sigma_alpha_0 * e^(-i * omega * t)
    rho_array[0, 1] *= np.conj(phase_1)
    rho_result = Qobj(rho_array, dims=rho.dims)
    # print(rho_array[0, 1], rho_array[1,0])

    # assert rho_result.isherm, "The resulting density matrix is not Hermitian."

    return rho_result


def _apply_RWA_phase_factors_2atom(rho: Qobj, t: float, omega: float) -> Qobj:
    """
    Apply time-dependent phase factors to the density matrix entries.

    Parameters:
        rho (Qobj): Density matrix (Qobj) to modify.
        omega (float): Frequency of the phase factor.
        t (float): Current time.

    Returns:
        Qobj: Modified density matrix with phase factors applied.
    """
    # Extract the density matrix as a NumPy array
    rho_array = rho.full()
    # print(rho.isherm)

    # Apply the phase factors to the specified elements
    phase_1 = np.exp(-1j * omega * t)  # e^(-i * omega * t)
    phase_2 = np.exp(-1j * 2 * omega * t)  # e^(-i * 2 * omega * t)

    # Modify the elements
    bar_alpha = 3
    for alpha in range(1, 3):
        rho_array[
            alpha, 0
        ] *= phase_1  # rho_alpha_0 = sigma_alpha_0 * e^(-i * omega * t)
        rho_array[0, alpha] *= np.conj(phase_1)

        rho_array[
            bar_alpha, alpha
        ] *= phase_1  # rho_bar_alpha_alpha = sigma_bar_alpha_alpha * e^(-i * omega * t)
        rho_array[alpha, bar_alpha] *= np.conj(phase_1)

    rho_array[
        bar_alpha, 0
    ] *= phase_2  # rho_bar_alpha_0 = sigma_bar_alpha_0 * e^(-i * 2 * omega * t)
    rho_array[0, bar_alpha] *= np.conj(phase_2)

    rho_result = Qobj(rho_array, dims=rho.dims)
    # print(rho_array[0, 1], rho_array[1,0])

    # assert rho_result.isherm, "The resulting density matrix is not Hermitian."

    return rho_result


def get_expect_vals_with_RWA(
    states: list[qutip.Qobj], times: np.array, system: SystemParameters
):
    """
    Calculate the expectation values in the result with RWA phase factors.

    Parameters:
        states= data.states (where data = qutip.Result): Results of the pulse evolution.
        times (list): Time points at which the expectation values are calculated.
        e_ops (list): the operators for which the expectation values are calculated
        omega (float): omega_laser (float): Frequency of the laser.
        RWA (bool): Whether to apply the RWA phase factors.
    Returns:
        list of lists: Expectation values for each operator of len(states).
    """
    omega = system.omega_laser
    e_ops = system.e_ops_list + [system.Dip_op]

    if system.RWA_laser:
        # Apply RWA phase factors to each state
        states = [
            apply_RWA_phase_factors(state, time, omega, system)
            for state, time in zip(states, times)
        ]
    updated_expects = [np.real(expect(states, e_op)) for e_op in e_ops]
    return updated_expects
