"""
Dataset classes for loading and processing 3D radiative transfer data.

This module contains PyTorch dataset classes for loading LASSO zarr data
for 3D heating rate emulation. The data structure follows the create_data.py
script which generates zarr files with CNN_input and CNN_output arrays.
"""

from .rt3d_dataset import (
    RT3DDataset,
    RT3DLogDataset,
    RT3DSplitBandDataset,
    RT3DSplitBandWithPressureDataset,
    create_rt3d_dataloaders
)

__all__ = ["RT3DDataset", "create_rt3d_dataloaders"]
