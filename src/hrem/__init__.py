"""
RTEM - RT Emulator

A proof of concept for emulated 3D radiative transfer using neural networks.
"""

__version__ = "0.1.0"
__author__ = "Simon"

from .models import *
from . import datasets

__all__ = ["__version__", "__author__", "datasets"]