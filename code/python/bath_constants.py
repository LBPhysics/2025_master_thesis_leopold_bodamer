import numpy as np
"""
    This file contains constants and parameters for the bath models.
    The different baths are modelled to agree as much as possible.
    The defining functions are found and tested in test_baths
    The constants are used there and also in the main code.
"""
# Define constants
Boltzmann = 1.0  # Boltzmann constant in J/K
hbar = 1.0  # Reduced Planck's constant in J·s
c = 1.0  # Speed of light in m/s

Temp = 1e-3  # Temperature in Kelvin
eta = 1e-2  # Coupling strength
cutoff = 1e2  # Cutoff frequency

# Define the args_bath dictionaries
args_paper = {
    "g": np.sqrt(eta * cutoff),
    "cutoff": cutoff,
    "Boltzmann": Boltzmann,
    "hbar": hbar,
    "Temp": Temp,
}
args_ohmic = {"eta": eta, "cutoff": cutoff, "s": 1.0}
args_drude_lorentz = {"lambda": eta * cutoff / 2, "cutoff": cutoff}
