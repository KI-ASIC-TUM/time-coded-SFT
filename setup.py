#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name="spikingFT",
    version="0.1",
    description="Package for implementing the FT in neuromorphic hardware",
    url="https://github.com/KI-ASIC-TUM/time-coded-SFT",
    author="Technical University of Munich. Informatik VI",
    packages=find_packages(exclude=["examples"]),
    install_requires=[
        "numpy>=1.16",
        "matplotlib>=3.1.2",
    ],
)
