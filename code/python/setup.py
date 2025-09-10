from setuptools import setup, find_packages

# TODO bring up to date
setup(
    name="qspectro2d",
    version="0.1.0",
    packages=find_packages(include=["qspectro2d", "qspectro2d.*", "project_config"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "qutip>=4.7.0",
        "psutil>=5.8.0",
        "pickle-mixin>=1.0.2",
        "joblib>=1.1.0",
    ],
    author="Leopold Bodamer",
    description="Quantum 2D Electronic Spectroscopy simulation package",
)
