#!/usr/bin/env python
"""
Adapted from setup.py authored by Colm Talbot at https://github.com/ColmTalbot/gwpopulation/blob/master/setup.py
"""
import os

from setuptools import find_packages
from setuptools import setup


def get_long_description():
    """Finds the README and reads in the description"""
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, "README.md")) as f:
        long_description = f.read()
    return long_description


VERSION = "0.0.2"
long_description = get_long_description()

with open("pip_requirements.txt", "r") as ff:
    requirements = ff.readlines()
    requirements.append("numpyro")
    requirements.append("jax")
    requirements.append("jaxlib")
    requirements.append("h5py")

setup(
    name="gwinferno",
    description="Gravitational-Wave Hierarchical Inference with NumPyro",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://git.ligo.org/bruce.edelman/gwinferno",
    author="Bruce Edelman, Jaxen Godfrey, Ben Farr",
    author_email="bedelman@uoregon.edu, jaxeng@uoregon.edu, bfarr@uoregon.edu",
    license="MIT",
    version=VERSION,
    packages=find_packages(exclude=["tests"]),
    package_dir={"gwinferno": "gwinferno"},
    scripts=["bin/create_py310_cpu_env.sh", "bin/create_py310_gpu_env.sh"],
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
