from dataclasses import dataclass
from typing import Optional
import json

# bath functions
from qspectro2d.core.bath_system.bath_fcts import (
    power_spectrum_func_paper,
    power_spectrum_func_ohmic,
    power_spectrum_func_drude_lorentz,
    spectral_density_func_drude_lorentz,
    spectral_density_func_ohmic,
    spectral_density_func_paper,
)
from qspectro2d.core.utils_and_config import BOLTZMANN, HBAR


@dataclass
class BathClass:
    """
    Parameters for the bath coupling and spectral density function.
    """

    # =============================
    # BATH PARAMETERS
    # =============================
    bath: str = "paper"  # Default bath type or "ohmic" or "dl"
    # Temperature / cutoff of the bath
    Temp: float = 1e-5  # zero temperature
    cutoff_: float = 1e2  # later * omega_A
    # decay  rates
    gamma_0: Optional[float] = 1 / 300.0
    gamma_phi: Optional[float] = 1 / 100.0

    def _args_bath(
        self, alpha: Optional[float] = None
    ) -> dict:  # TODO DONT KNOW HOW TO PUT THIS HERE CAUSE cutoff = cutoff_ * omega_A
        """
        Generate arguments for the bath functions.

        Parameters:
            alpha (Optional[float]): Coupling constant. Defaults to self.gamma_0.

        Returns:
            dict: Arguments for the bath functions.
        """
        if alpha is None:
            alpha = self.gamma_0

        # OMEGA is hardcoded as 1 for now; consider making it configurable
        OMEGA = 1

        return {
            "alpha": alpha,
            "cutoff": self.cutoff_ * OMEGA,
            "Boltzmann": BOLTZMANN,
            "hbar": HBAR,
            "Temp": self.Temp,
            "s": 1.0,  # ohmic spectrum
        }

    def power_spectrum_func(self, w: float, args: dict) -> float:
        """
        Calculate the power spectrum function based on the bath type.

        Parameters:
            w (float): Frequency in fs^-1.
            args (dict): Arguments for the bath functions.

        Returns:
            float: The value of the power spectrum function.

        Raises:
            ValueError: If the bath type is unknown.
        """
        if self.bath == "paper":
            return power_spectrum_func_paper(w, args)
        elif self.bath == "ohmic":
            return power_spectrum_func_ohmic(w, args)
        elif self.bath == "dl":
            return power_spectrum_func_drude_lorentz(w, args)
        else:
            raise ValueError(
                f"Unknown bath type: {self.bath}. Valid types are 'paper', 'ohmic', 'dl'."
            )

    def spectral_density_func(self, w: float, args: dict) -> float:
        """
        Calculate the spectral density function based on the bath type.

        Parameters:
            w (float): Frequency in fs^-1.
            args (dict): Arguments for the bath functions.

        Returns:
            float: The value of the spectral density function.

        Raises:
            ValueError: If the bath type is unknown.
        """
        if self.bath == "paper":
            return spectral_density_func_paper(w, args)
        elif self.bath == "ohmic":
            return spectral_density_func_ohmic(w, args)
        elif self.bath == "dl":
            return spectral_density_func_drude_lorentz(w, args)
        else:
            raise ValueError(
                f"Unknown bath type: {self.bath}. Valid types are 'paper', 'ohmic', 'dl'."
            )

    @property
    def Gamma(self):
        return self.gamma_0 / 2 + self.gamma_phi

    def summary(self):
        """
        Print a summary of the bath parameters.
        """
        print("\n# Summary of Bath Parameters:")
        for key, value in self.__dict__.items():
            print(f"    {key:<20}: {value}")

        # Bath Parameters
        print("\n# With parameters for the BATH:")
        print(f"    {'gamma_0':<20}: {self.gamma_0:.4f} fs^-1")
        print(f"    {'gamma_phi':<20}: {self.gamma_phi:.4f} fs^-1")
        print(f"    {'Temp':<20}: {self.Temp} K")
        print(f"    {'cutoff_':<20}: {self.cutoff_} fs^-1")

    def to_dict(self) -> dict:
        """
        Convert the BathClass instance to a dictionary.

        Returns:
            dict: Dictionary representation of the instance.
        """
        return {
            "bath": self.bath,
            "Temp": self.Temp,
            "cutoff_": self.cutoff_,
            "gamma_0": self.gamma_0,
            "gamma_phi": self.gamma_phi,
        }

    @classmethod
    def from_dict(cls, data: dict):
        """
        Create a BathClass instance from a dictionary.

        Parameters:
            data (dict): Dictionary containing the parameters.

        Returns:
            BathClass: Instance of BathClass.
        """
        return cls(
            bath=data.get("bath", "paper"),
            Temp=data.get("Temp", 1e-5),
            cutoff_=data.get("cutoff_", 1e2),
            gamma_0=data.get("gamma_0", 1 / 300.0),
            gamma_phi=data.get("gamma_phi", 1 / 100.0),
        )

    def to_json(self) -> str:
        """
        Convert the BathClass instance to a JSON string.

        Returns:
            str: JSON string representation of the instance.
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str):
        """
        Create a BathClass instance from a JSON string.

        Parameters:
            json_str (str): JSON string containing the parameters.

        Returns:
            BathClass: Instance of BathClass.
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
