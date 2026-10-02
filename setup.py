"""Compatibility packaging shim for offline runtimes with older setuptools.

Canonical metadata lives in ``pyproject.toml``.  This file repeats the small
static package declaration so that `pip install -e .` still works in research
images that predate PEP 621 support in setuptools.
"""

from setuptools import find_packages, setup


setup(
    name="experthtr",
    version="0.1.0",
    description="Reproducibility artifact for sparse MoE full-input handwritten text recognition",
    author="Dang Hoai Nam, Nguyen Duy Hieu, Vo Nguyen Le Duy",
    license="Apache-2.0",
    keywords="handwritten-text-recognition htr mixture-of-experts multitask-learning",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    packages=find_packages(),
    install_requires=["PyYAML>=5.4"],
    entry_points={"console_scripts": ["experthtr=expert.cli:main"]},
)
