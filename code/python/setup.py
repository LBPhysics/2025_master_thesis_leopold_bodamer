from setuptools import setup, find_packages

setup(
    name="master_thesis_2des",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "qutip",
        "matplotlib",
        "dataclasses; python_version<'3.7'",
    ],
    author="Leopold",
    description="2D Electronic Spectroscopy simulation package",
)
