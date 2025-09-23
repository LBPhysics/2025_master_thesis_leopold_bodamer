from setuptools import setup, find_packages

# Provide only the local project utilities (project_config) from this repo.
# The qspectro2d package is now installed from the external submodule.
setup(
    name="project-config-tools",
    version="0.1.0",
    packages=find_packages(include=["project_config", "project_config.*"]),
    python_requires=">=3.8",
    install_requires=[],
    author="Leopold Bodamer",
    description="Local project configuration utilities (paths, helpers)",
)
