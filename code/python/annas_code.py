# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 16:26:00 2024

@author: FAST-TRAPSENSOR
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.fft import fft, fftfreq
import time


def axial_crystal_real(y, t, omega_0, mu, d, kappa):
    zs, auxzs, zt, auxzt = y
    dydt = [
        auxzs,
        omega_0**2
        * (
            -zs
            + kappa / (kappa + 1) * d
            + (kappa / (kappa + 1)) * d**3 * (zs - zt - d) / np.abs((zt - zs + d) ** 3)
        ),
        auxzt,
        omega_0**2
        * kappa
        / mu
        * (
            -zt
            - 1 / (kappa + 1) * d
            + 1 / (kappa + 1) * d**3 * (zt - zs + d) / np.abs((zt - zs + d) ** 3)
        ),
    ]
    return dydt


def calculate_frequency(ms, mt, qs, qt, freq, zs0, mode, t, res, label):

    ref = time.time()

    # Universal constants
    q = 1.602176634e-19
    epsilon_0 = 8.8541878188e-12
    uma_to_kg = 1.66053906892e-27

    # Derived parameters
    mu = mt / ms
    kappa = qt / qs
    omega_0 = 2 * math.pi * freq
    d = ((kappa + 1) * q**2 / (4 * math.pi * epsilon_0 * ms * uma_to_kg * omega_0**2)) ** (1 / 3)
    alpha_1 = kappa / mu - 1
    beta_1 = 1 / mu - 1
    alpha_2 = kappa / mu + 1
    beta_2 = 1 / mu + 1
    gamma = 2 * kappa / (kappa + 1)
    print("Ion distance: {} um".format(d * 1e6))

    # Initial conditions
    vzs0 = 0
    if mode == "common":
        zt0 = (
            -zs0
            * 0.5
            * (
                alpha_1
                + beta_1 * gamma
                - np.sqrt((alpha_2 + beta_2 * gamma) ** 2 - 12 * kappa / mu)
            )
            / gamma
        )
    elif mode == "stretch":
        zt0 = (
            -zs0
            * 0.5
            * (
                alpha_1
                + beta_1 * gamma
                + np.sqrt((alpha_2 + beta_2 * gamma) ** 2 - 12 * kappa / mu)
            )
            / gamma
        )
    else:
        print("unknown mode")
    vzt0 = 0
    y0 = [zs0, vzs0, zt0, vzt0]

    # Time binning
    N = int(t / res) + 1
    t = np.linspace(0, t, N)

    # Solve differential equation
    sol = odeint(axial_crystal_real, y0, t, args=(omega_0, mu, d, kappa))
    # plt.plot(t, sol[:,0]); plt.show()

    # Perform FFT
    f = fftfreq(N, res)[: N // 2]
    sol_fft = np.abs(fft(sol[:, 0]))[0 : N // 2]
    plt.plot(f, sol_fft / max(sol_fft), label=label)

    # Save data
    np.savetxt(
        "kappa_{}ms_{}_mt_{}_freq_{}_zs0_{}_mode_{}_t_{}_res_{}.dat".format(
            kappa, ms, mt, freq, zs0, mode, t[-1], res
        ),
        np.column_stack((f, sol_fft)),
    )

    print(time.time() - ref)


ms = 40
mt = 44
qs = 1
qt = 1
freq = 591.2e3
zs0 = 0.01e-6
mode = "common"
# mode = 'stretch'
t = 1
res = 0.5e-6
calculate_frequency(ms, mt, qs, qt, freq, zs0, mode, t, res, str(zs0))

# for zs0 in np.linspace(1e-6,10e-6,10):
# calculate_frequency(ms, mt, qs, qt, freq, zs0, mode, t, res, str(zs0))

# plt.legend()
plt.show()
