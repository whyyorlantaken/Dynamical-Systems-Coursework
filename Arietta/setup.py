# Import
from setuptools import setup, find_packages

# Setup
setup(
    name = "Arietta",
    version = "1.0",
    description = "Various cellular automata implementations",
    author = "Males-Araujo Yorlan",
    author_email = "yorlan.males@yachaytech.edu.ec",
    packages = find_packages(),
    install_requires = [
        "numpy"
    ],
    python_requires = "==3.9.*",
)