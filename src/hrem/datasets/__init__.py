"""
hrem.datasets
=============

This module provides dataset classes to load the training, validation, and testing data for
the HREM study.
"""
from functools import cache
import os
from pathlib import Path
from typing import Dict, Optional, Union, Tuple, List
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr
import zarr


class HREMDataset0(Dataset):
    """
    Dataset to load version X.0 of the input data providing three input features:

        - Direct downwelling broadband flux
        - Gas optical depth (GOD)
        - Cloud optical depth (COD)
    """
    def __init__(
        self,
        zarr_file_path: str,
        cache_in_memory: bool = False,
        validation: bool = True
    ):
        """
        Initialize the dataset.

        Args:
            zarr_file_path: Path to the zarr file (HR_train_patches.zarr, HR_val_patches.zarr, HR_test_patches.zarr)
            target_transform: Optional transform to apply to targets
            cache_in_memory: Whether to cache data in memory for faster access
        """
        self.zarr_file_path = Path(zarr_file_path)
        self.validation = validation

        # Load zarr store
        self._load_zarr_store()

        # Cache data if requested
        self._cached_data = None
        if cache_in_memory:
            self._cache_data()

    def _load_zarr_store(self):
        """Load the zarr store and extract CNN_input and CNN_output arrays."""
        if not os.path.exists(self.zarr_file_path):
            raise FileNotFoundError(f"Zarr file not found: {self.zarr_file_path}")

        try:
            self.store = zarr.open(self.zarr_file_path, mode='r')
            if 'CNN_input' not in self.store:
                raise KeyError("CNN_input array not found in zarr file")
            if 'CNN_output' not in self.store:
                raise KeyError("CNN_output array not found in zarr file")
            self.cnn_input = self.store['CNN_input']
            self.cnn_output = self.store['CNN_output']
            # Determine dataset length
            self.length = self.cnn_input.shape[0]
        except Exception as e:
            raise RuntimeError(f"Failed to load zarr file {self.zarr_file_path}: {e}")

    def _get_sample(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single heating rate sample from the dataset."""
        inputs = np.array(self.cnn_input[idx])[..., [1, 2, 0]]
        target = np.array(self.cnn_output[idx])
        inputs = np.transpose(inputs, (3, 0, 1, 2))
        target = np.transpose(target, (3, 0, 1, 2))
        inputs = torch.from_numpy(inputs).float()
        target = torch.from_numpy(target).float()
        return inputs, target

    def get_cloud_mask(self, idx: int) -> np.ndarray:
        """Get cloud mask for given sample."""
        return 1e-9 < np.array(self.cnn_input[idx, ..., 0])

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")

        inputs, target = self._get_sample(idx)

        if (not self.validation)  and 0.5  < self.rng.random():
            inputs = torch.flip(inputs, (-2,))
            target = torch.flip(target, (-2,))

        if (not self.validation) and 0.5  < self.rng.random():
            inputs = torch.flip(inputs, (-3,))
            target = torch.flip(target, (-3,))

        # Transform input values
        inputs[0] = torch.log10(inputs[0])
        inputs[1] = inputs[1]
        inputs[2] = torch.log10(torch.maximum(inputs[2], torch.tensor(1e-2)))

        return inputs, target

    def get_normalization_params(self) -> Dict:
        """Get normalization parameters for denormalization."""
        if not hasattr(self, 'norm_params'):
            return {}
        return self.norm_params.copy()

    def denormalize_prediction(self, prediction: torch.Tensor) -> torch.Tensor:
        """Denormalize model predictions back to original scale."""
        # Note: This is a placeholder - actual denormalization would depend on
        # how the target variables were normalized in the original LASSO data
        warnings.warn("Denormalization not implemented - returning prediction as-is")
        return prediction

    def get_sample_metadata(self, idx: int) -> Dict:
        """Get metadata for a specific sample."""
        sample_inputs, sample_target = self._get_sample(idx)

        metadata = {
            'index': idx,
            'zarr_file': self.zarr_file_path,
            'input_shape': sample_inputs.shape,
            'target_shape': sample_target.shape,
            'input_channels': self.input_channels,
            'target_channels': self.output_channels,
            'spatial_dimensions': {
                'height': 250,  # Vertical cross-section height
                'width': 11,    # Cross-section width (crop_width)
                'depth': 150    # Vertical levels
            }
        }

        return metadata


class HREMDataset1(Dataset):
    """
    Dataset to load version X.1 of the input data providing four input features:

        - Direct downwelling broadband flux, log-transformed
        - Gas optical depth (GOD)
        - Liquid water path, log transformed
        - Cloud droplet effective radius in microns
    """
    def __init__(
        self,
        zarr_file_path: str,
        cache_in_memory: bool = False,
        validation: bool = True
    ):
        """
        Initialize the RT3DDataset.

        Args:
            zarr_file_path: Path to the zarr file (HR_train_patches.zarr, HR_val_patches.zarr, HR_test_patches.zarr)
            target_transform: Optional transform to apply to targets
            cache_in_memory: Whether to cache data in memory for faster access
        """
        self.zarr_file_path = Path(zarr_file_path)
        self.validation = validation

        # Load zarr store
        self._load_zarr_store()

        # Cache data if requested
        self._cached_data = None
        if cache_in_memory:
            self._cache_data()

    def _load_zarr_store(self):
        """Load the zarr store and extract CNN_input and CNN_output arrays."""
        if not os.path.exists(self.zarr_file_path):
            raise FileNotFoundError(f"Zarr file not found: {self.zarr_file_path}")

        try:
            self.store = zarr.open(self.zarr_file_path, mode='r')
            if 'CNN_input' not in self.store:
                raise KeyError("CNN_input array not found in zarr file")
            if 'CNN_output' not in self.store:
                raise KeyError("CNN_output array not found in zarr file")
            self.cnn_input = self.store['CNN_input']
            self.cnn_output = self.store['CNN_output']
            # Determine dataset length
            self.length = self.cnn_input.shape[0]
        except Exception as e:
            raise RuntimeError(f"Failed to load zarr file {self.zarr_file_path}: {e}")

    def _get_sample(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single heating rate sample from the dataset."""
        inputs = np.array(self.cnn_input[idx])[..., [3, 4, 1, 2]]
        target = np.array(self.cnn_output[idx])
        inputs = np.transpose(inputs, (3, 0, 1, 2))
        target = np.transpose(target, (3, 0, 1, 2))
        inputs = torch.from_numpy(inputs).float()
        target = torch.from_numpy(target).float()
        return inputs, target

    def get_cloud_mask(self, idx: int) -> np.ndarray:
        """Get cloud mask for given sample."""
        return 1e-9 < np.array(self.cnn_input[idx, ..., 0])

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")

        inputs, target = self._get_sample(idx)

        if (not self.validation)  and 0.5  < self.rng.random():
            inputs = torch.flip(inputs, (-2,))
            target = torch.flip(target, (-2,))

        if (not self.validation) and 0.5  < self.rng.random():
            inputs = torch.flip(inputs, (-3,))
            target = torch.flip(target, (-3,))

        # Transform input values
        inputs[0] = torch.log10(inputs[0])
        inputs[1] = inputs[1]
        inputs[2] = torch.log10(torch.maximum(inputs[2], torch.tensor(1e-5)))
        inputs[3] = inputs[3] * 1e-6

        return inputs, target


class RT3DSplitBandWithPressureDataset(Dataset):
    """
    PyTorch Dataset for loading 3D heating rate data from LASSO zarr files.

    This dataset loads 3D cross-sectional data for heating rate prediction.
    Based on create_data.py, the zarr files contain:
    - CNN_input: shape (N, 250, 11, 150, 3) with channels [COD, DownDir, GOD]
    - CNN_output: shape (N, 250, 11, 150, 1) with flux divergence (heating rates)

    Where:
    - COD: Cloud Optical Depth
    - DownDir: Downwelling Direct Flux
    - GOD: Gas + Aerosol Optical Depth
    - N: Number of patches
    - 250: Vertical cross-section height
    - 11: Cross-section width (crop_width)
    - 150: Vertical levels
    """

    def __init__(
        self,
        zarr_file_paths: Union[str, List[str]],
        profile_file_paths: Union[str, List[str]],
        validation: bool = False
    ):
        """
        Initialize the RT3DDataset.

        Args:
            zarr_file_path: Path to the zarr file (HR_train_patches.zarr, HR_val_patches.zarr, HR_test_patches.zarr)
            transform: Optional transform to apply to inputs
            target_transform: Optional transform to apply to targets
            cache_in_memory: Whether to cache data in memory for faster access
        """
        if isinstance(zarr_file_paths, str):
            zarr_file_paths = [zarr_file_paths]
        self.zarr_file_paths = [Path(path) for path in zarr_file_paths]
        if isinstance(profile_file_paths, str):
            profile_file_paths = [profile_file_paths]
        self.profile_file_paths = profile_file_paths
        self._load_zarr_store()
        self.validation = validation
        self.rng = np.random.default_rng(42)

    def _load_zarr_store(self):
        """Load the zarr store and extract CNN_input and CNN_output arrays."""
        self.cnn_inputs = []
        self.cnn_outputs = []

        for path in self.zarr_file_paths:
            if not path.exists():
                raise FileNotFoundError(f"Zarr file not found: {path}")
            try:
                store = zarr.open(path, mode='r')
                self.cnn_inputs.append(store['CNN_input'])
                self.cnn_outputs.append(store['CNN_output'])

            except Exception as e:
                raise RuntimeError(f"Failed to load zarr file {path}: {e}")

    @cache
    def get_profiles(self, file_ind: int, ind: int) -> np.ndarray:
        """
        Load profiles for given case.

        Args:
            ind: The index of the scene for which to load  the profile.
        """
        ind = ind // 192
        with xr.open_dataset(self.profile_file_paths[file_ind]) as profile_data:
            temp = profile_data.Temp[{"index": ind}].compute().data
            pres = profile_data.Pres[{"index": ind}].compute().data
            qH2O = profile_data.qH2O[{"index": ind}].compute().data
            qO3 = profile_data.qO3[{"index":ind}].compute().data

        return np.stack([temp, pres, qH2O, qO3], axis=0)


    def _get_sample(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single heating rate sample from the dataset."""
        # Load input data: shape (250, 11, 150, 3)
        ind = 0
        file_ind = 0
        while True:
            inputs = self.cnn_inputs[ind]
            outputs = self.cnn_outputs[ind]
            if idx < inputs.shape[0]:
                break
            else:
                file_ind += 1
                idx -= inputs.shape[0]

        inputs = np.array(inputs[idx])
        target = np.array(outputs[idx])

        inputs = np.transpose(inputs, (3, 0, 1, 2))
        target = np.transpose(target, (3, 0, 1, 2))

        # Convert to tensors
        inputs = torch.from_numpy(inputs).float()
        target = torch.from_numpy(target).float()

        profiles = torch.from_numpy(self.get_profiles(file_ind, idx)).float()[..., None, None, :]
        profiles = torch.broadcast_to(profiles, (4,) + inputs.shape[1:])

        inputs[:4] = profiles

        return inputs, target

    def get_cloud_mask(self, idx: int) -> np.ndarray:
        ind = 0
        file_ind = 0
        while True:
            inputs = self.cnn_inputs[ind]
            outputs = self.cnn_outputs[ind]
            if idx < inputs.shape[0]:
                break
            else:
                file_ind += 1
                idx -= inputs.shape[0]

        return (1e-10 < np.array(inputs[idx, ..., -1]))


    def __len__(self) -> int:
        """Return the length of the dataset."""
        return sum([inpt.shape[0] for inpt in self.cnn_inputs])

    def worker_init_fn(self, w_id: int) -> None:
        """
        Seeds the dataset loader's random number generator.
        """
        seed = int.from_bytes(os.urandom(4), "big") + w_id
        self.rng = np.random.default_rng(seed)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")

        inputs, target = self._get_sample(idx)

        inputs[2] = torch.maximum(np.log10(inputs[2]), torch.tensor(-7.0))
        inputs[3] = 1e8 * inputs[3]

        for ind in range(4, 15):
            inputs[ind] = torch.maximum(np.log10(inputs[ind]), torch.tensor(-7.0))
        for ind in range(15, 30):
            inputs[ind] = torch.maximum(np.log10(inputs[ind]), torch.tensor(-8.0))
        inputs[30] = 1e4 * inputs[30]
        inputs[31] = torch.maximum(np.log10(inputs[31]), torch.tensor(-5.0))
        inputs[32] = torch.maximum(np.log10(inputs[32]), torch.tensor(-7.0))
        inputs[33] = torch.maximum(np.log10(inputs[33]), torch.tensor(-3.0))

        if (not self.validation)  and 0.5  < self.rng.random():
            inputs = torch.flip(inputs, (-2,))
            target = torch.flip(target, (-2,))

        if (not self.validation) and 0.5  < self.rng.random():
            inputs = torch.flip(inputs, (-3,))
            target = torch.flip(target, (-3,))

        return inputs, target

    def get_normalization_params(self) -> Dict:
        """Get normalization parameters for denormalization."""
        if not hasattr(self, 'norm_params'):
            return {}
        return self.norm_params.copy()

    def denormalize_prediction(self, prediction: torch.Tensor) -> torch.Tensor:
        """Denormalize model predictions back to original scale."""
        # Note: This is a placeholder - actual denormalization would depend on
        # how the target variables were normalized in the original LASSO data
        warnings.warn("Denormalization not implemented - returning prediction as-is")
        return prediction

    def get_sample_metadata(self, idx: int) -> Dict:
        """Get metadata for a specific sample."""
        sample_inputs, sample_target = self._get_sample(idx)

        metadata = {
            'index': idx,
            'zarr_file': self.zarr_file_path,
            'input_shape': sample_inputs.shape,
            'target_shape': sample_target.shape,
            'input_channels': self.input_channels,
            'target_channels': self.output_channels,
            'spatial_dimensions': {
                'height': 250,  # Vertical cross-section height
                'width': 11,    # Cross-section width (crop_width)
                'depth': 150    # Vertical levels
            }
        }

        return metadata


class HREMDataset2(Dataset):
    """
    Dataset to load version X.2 of the input data providing 14 input features:

        - Broadband downwelling broadband flux, 11 bands, log-transformed
        - Gas optical depth (GOD)
        - Liquid water path, log transformed
        - Cloud droplet effective radius in microns
    """
    def __init__(
        self,
        zarr_file_path: str,
        cache_in_memory: bool = False,
        validation: bool = True
    ):
        """
        Initialize the RT3DDataset.

        Args:
            zarr_file_path: Path to the zarr file (HR_train_patches.zarr, HR_val_patches.zarr, HR_test_patches.zarr)
            target_transform: Optional transform to apply to targets
            cache_in_memory: Whether to cache data in memory for faster access
        """
        self.zarr_file_path = Path(zarr_file_path)
        self.validation = validation

        # Load zarr store
        self._load_zarr_store()

        # Cache data if requested
        self._cached_data = None
        if cache_in_memory:
            self._cache_data()

    def _load_zarr_store(self):
        """Load the zarr store and extract CNN_input and CNN_output arrays."""
        if not os.path.exists(self.zarr_file_path):
            raise FileNotFoundError(f"Zarr file not found: {self.zarr_file_path}")

        try:
            self.store = zarr.open(self.zarr_file_path, mode='r')
            if 'CNN_input' not in self.store:
                raise KeyError("CNN_input array not found in zarr file")
            if 'CNN_output' not in self.store:
                raise KeyError("CNN_output array not found in zarr file")
            self.cnn_input = self.store['CNN_input']
            self.cnn_output = self.store['CNN_output']
            # Determine dataset length
            self.length = self.cnn_input.shape[0]
        except Exception as e:
            raise RuntimeError(f"Failed to load zarr file {self.zarr_file_path}: {e}")

    def _get_sample(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single heating rate sample from the dataset."""
        inputs = np.array(self.cnn_input[idx])[..., [4, 5, 6 ,7, 8, 9, 10, 11, 12, 13, 14, 18, 15, 16]]
        target = np.array(self.cnn_output[idx])
        inputs = np.transpose(inputs, (3, 0, 1, 2))
        target = np.transpose(target, (3, 0, 1, 2))
        inputs = torch.from_numpy(inputs).float()
        target = torch.from_numpy(target).float()
        return inputs, target

    def get_cloud_mask(self, idx: int) -> np.ndarray:
        """Get cloud mask for given sample."""
        return 1e-9 < np.array(self.cnn_input[idx, ..., -4])

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")

        inputs, target = self._get_sample(idx)

        if (not self.validation)  and 0.5  < self.rng.random():
            inputs = torch.flip(inputs, (-2,))
            target = torch.flip(target, (-2,))

        if (not self.validation) and 0.5  < self.rng.random():
            inputs = torch.flip(inputs, (-3,))
            target = torch.flip(target, (-3,))

        # Transform input values
        inputs[:11] = torch.log10(inputs[:11])
        inputs[11] = inputs[11]
        inputs[12] = torch.log10(torch.maximum(inputs[12], torch.tensor(1e-5)))
        inputs[13] = inputs[13] * 1e-6

        return inputs, target


class HREMDataset3(Dataset):
    """
    Dataset to load version X.3 of the input data providing 19 input features:

        - Broadband downwelling broadband flux, log-transformed
        - Gas optical depth, 15 bands
        - Aerosol optical depth
        - Liquid water path, log transformed
        - Cloud droplet effective radius in microns
    """
    def __init__(
        self,
        zarr_file_path: str,
        cache_in_memory: bool = False,
        validation: bool = True
    ):
        """
        Initialize the RT3DDataset.

        Args:
            zarr_file_path: Path to the zarr file (HR_train_patches.zarr, HR_val_patches.zarr, HR_test_patches.zarr)
            target_transform: Optional transform to apply to targets
            cache_in_memory: Whether to cache data in memory for faster access
        """
        self.zarr_file_path = Path(zarr_file_path)
        self.validation = validation

        # Load zarr store
        self._load_zarr_store()

        # Cache data if requested
        self._cached_data = None
        if cache_in_memory:
            self._cache_data()

    def _load_zarr_store(self):
        """Load the zarr store and extract CNN_input and CNN_output arrays."""
        if not os.path.exists(self.zarr_file_path):
            raise FileNotFoundError(f"Zarr file not found: {self.zarr_file_path}")

        try:
            self.store = zarr.open(self.zarr_file_path, mode='r')
            if 'CNN_input' not in self.store:
                raise KeyError("CNN_input array not found in zarr file")
            if 'CNN_output' not in self.store:
                raise KeyError("CNN_output array not found in zarr file")
            self.cnn_input = self.store['CNN_input']
            self.cnn_output = self.store['CNN_output']
            # Determine dataset length
            self.length = self.cnn_input.shape[0]
        except Exception as e:
            raise RuntimeError(f"Failed to load zarr file {self.zarr_file_path}: {e}")

    def _get_sample(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single heating rate sample from the dataset."""
        inputs = np.array(self.cnn_input[idx])
        inputs_dr = inputs[..., 4:15].sum(-1)[..., None]
        inputs_god = inputs[..., 15:30]
        inputs_aod = inputs[..., [30]]
        inputs_cloud = inputs[..., 31:33]
        inputs = np.concatenate([inputs_dr, inputs_god, inputs_aod, inputs_cloud], axis=-1)
        target = np.array(self.cnn_output[idx])
        inputs = np.transpose(inputs, (3, 0, 1, 2))
        target = np.transpose(target, (3, 0, 1, 2))
        inputs = torch.from_numpy(inputs).float()
        target = torch.from_numpy(target).float()
        return inputs, target

    def get_cloud_mask(self, idx: int) -> np.ndarray:
        """Get cloud mask for given sample."""
        return 1e-9 < np.array(self.cnn_input[idx, ..., -1])

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")

        inputs, target = self._get_sample(idx)

        if (not self.validation)  and 0.5  < self.rng.random():
            inputs = torch.flip(inputs, (-2,))
            target = torch.flip(target, (-2,))

        if (not self.validation) and 0.5  < self.rng.random():
            inputs = torch.flip(inputs, (-3,))
            target = torch.flip(target, (-3,))

        # Transform input values
        inputs[:1] = torch.log10(inputs[:1])
        inputs[17] = torch.log10(torch.maximum(inputs[17], torch.tensor(1e-5)))
        inputs[18] = inputs[18]

        return inputs, target


class HREMDataset4(Dataset):
    """
    Dataset to load version X.4 of the input data providing 29 input features:

        - Downwelling broadband flux, 11 bands, log-transformed
        - Gas optical depth, 15 bands
        - Aerosol optical depth
        - Liquid water path, log transformed
        - Cloud droplet effective radius in microns
    """
    def __init__(
        self,
        zarr_file_path: str,
        cache_in_memory: bool = False,
        validation: bool = True
    ):
        """
        Initialize the HREMDataset.

        Args:
            zarr_file_path: Path to the zarr file (HR_train_patches.zarr, HR_val_patches.zarr, HR_test_patches.zarr)
            target_transform: Optional transform to apply to targets
            cache_in_memory: Whether to cache data in memory for faster access
        """
        self.zarr_file_path = Path(zarr_file_path)
        self.validation = validation

        # Load zarr store
        self._load_zarr_store()

        # Cache data if requested
        self._cached_data = None
        if cache_in_memory:
            self._cache_data()

    def _load_zarr_store(self):
        """Load the zarr store and extract CNN_input and CNN_output arrays."""
        if not os.path.exists(self.zarr_file_path):
            raise FileNotFoundError(f"Zarr file not found: {self.zarr_file_path}")

        try:
            self.store = zarr.open(self.zarr_file_path, mode='r')
            if 'CNN_input' not in self.store:
                raise KeyError("CNN_input array not found in zarr file")
            if 'CNN_output' not in self.store:
                raise KeyError("CNN_output array not found in zarr file")
            self.cnn_input = self.store['CNN_input']
            self.cnn_output = self.store['CNN_output']
            # Determine dataset length
            self.length = self.cnn_input.shape[0]
        except Exception as e:
            raise RuntimeError(f"Failed to load zarr file {self.zarr_file_path}: {e}")

    def _get_sample(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single heating rate sample from the dataset."""
        inputs = np.array(self.cnn_input[idx])
        inputs_dr = inputs[..., 4:15]
        inputs_god = inputs[..., 15:30]
        inputs_aod = inputs[..., [30]]
        inputs_cloud = inputs[..., 31:33]
        inputs = np.concatenate([inputs_dr, inputs_god, inputs_aod, inputs_cloud], axis=-1)
        target = np.array(self.cnn_output[idx])
        inputs = np.transpose(inputs, (3, 0, 1, 2))
        target = np.transpose(target, (3, 0, 1, 2))
        inputs = torch.from_numpy(inputs).float()
        target = torch.from_numpy(target).float()
        return inputs, target

    def get_cloud_mask(self, idx: int) -> np.ndarray:
        """Get cloud mask for given sample."""
        return 1e-9 < np.array(self.cnn_input[idx, ..., -1])

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")

        inputs, target = self._get_sample(idx)

        if (not self.validation)  and 0.5  < self.rng.random():
            inputs = torch.flip(inputs, (-2,))
            target = torch.flip(target, (-2,))

        if (not self.validation) and 0.5  < self.rng.random():
            inputs = torch.flip(inputs, (-3,))
            target = torch.flip(target, (-3,))

        # Transform input values
        inputs[:11] = torch.log10(inputs[:11])
        inputs[27] = torch.log10(torch.maximum(inputs[27], torch.tensor(1e-5)))
        inputs[28] = inputs[28]

        return inputs, target


class HREMDataset5(Dataset):
    """
    Dataset to load version X.5 of the input data providing 33 input features:

        - Temperature, Pressure, H2O mixing ratios, O3 mixing ratios
        - Downwelling broadband flux, 11 bands, log-transformed
        - Gas optical depth, 15 bands
        - Aerosol optical depth
        - Liquid water path, log transformed
        - Cloud droplet effective radius in microns
    """

    def __init__(
        self,
        zarr_file_paths: Union[str, List[str]],
        profile_file_paths: Union[str, List[str]],
        validation: bool = False
    ):
        """
        Initialize the RT3DDataset.

        Args:
            zarr_file_path: Path to the zarr file (HR_train_patches.zarr, HR_val_patches.zarr, HR_test_patches.zarr)
            transform: Optional transform to apply to inputs
            target_transform: Optional transform to apply to targets
            cache_in_memory: Whether to cache data in memory for faster access
        """
        if isinstance(zarr_file_paths, str):
            zarr_file_paths = [zarr_file_paths]
        self.zarr_file_paths = [Path(path) for path in zarr_file_paths]
        if isinstance(profile_file_paths, str):
            profile_file_paths = [profile_file_paths]
        self.profile_file_paths = profile_file_paths
        self._load_zarr_store()
        self.validation = validation
        self.rng = np.random.default_rng(42)

    def _load_zarr_store(self):
        """Load the zarr store and extract CNN_input and CNN_output arrays."""
        self.cnn_inputs = []
        self.cnn_outputs = []

        for path in self.zarr_file_paths:
            if not path.exists():
                raise FileNotFoundError(f"Zarr file not found: {path}")
            try:
                store = zarr.open(path, mode='r')
                self.cnn_inputs.append(store['CNN_input'])
                self.cnn_outputs.append(store['CNN_output'])

            except Exception as e:
                raise RuntimeError(f"Failed to load zarr file {path}: {e}")

    @cache
    def get_profiles(self, file_ind: int, ind: int) -> np.ndarray:
        """
        Load profiles for given case.

        Args:
            ind: The index of the scene for which to load  the profile.
        """
        ind = ind // 192
        with xr.open_dataset(self.profile_file_paths[file_ind]) as profile_data:
            temp = profile_data.Temp[{"index": ind}].compute().data
            pres = profile_data.Pres[{"index": ind}].compute().data
            qH2O = profile_data.qH2O[{"index": ind}].compute().data
            qO3 = profile_data.qO3[{"index":ind}].compute().data

        return np.stack([temp, pres, qH2O, qO3], axis=0)


    def _get_sample(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single heating rate sample from the dataset."""
        # Load input data: shape (250, 11, 150, 3)
        ind = 0
        file_ind = 0
        while True:
            inputs = self.cnn_inputs[ind]
            outputs = self.cnn_outputs[ind]
            if idx < inputs.shape[0]:
                break
            else:
                file_ind += 1
                idx -= inputs.shape[0]

        inputs = np.array(inputs[idx])
        inputs_dr = inputs[..., 4:15]
        inputs_god = inputs[..., 15:30]
        inputs_aod = inputs[..., [30]]
        inputs_cloud = inputs[..., 31:33]
        inputs = np.concatenate([inputs_dr, inputs_god, inputs_aod, inputs_cloud], axis=-1)

        inputs = np.transpose(inputs, (3, 0, 1, 2))
        target = np.array(outputs[idx])
        target = np.transpose(target, (3, 0, 1, 2))

        profiles = torch.from_numpy(self.get_profiles(file_ind, idx)).float()[..., None, None, :]
        profiles = torch.broadcast_to(profiles, (4,) + inputs.shape[1:])
        inputs = np.concatenate([inputs, profiles], axis=0)

        # Convert to tensors
        inputs = torch.from_numpy(inputs).float()
        target = torch.from_numpy(target).float()

        return inputs, target

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return sum([inpt.shape[0] for inpt in self.cnn_inputs])

    def worker_init_fn(self, w_id: int) -> None:
        """
        Seeds the dataset loader's random number generator.
        """
        seed = int.from_bytes(os.urandom(4), "big") + w_id
        self.rng = np.random.default_rng(seed)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")

        inputs, target = self._get_sample(idx)

        if (not self.validation)  and 0.5  < self.rng.random():
            inputs = torch.flip(inputs, (-2,))
            target = torch.flip(target, (-2,))

        if (not self.validation) and 0.5  < self.rng.random():
            inputs = torch.flip(inputs, (-3,))
            target = torch.flip(target, (-3,))

        # Transform input values
        inputs[:11] = torch.log10(inputs[:11])
        inputs[27] = torch.log10(torch.maximum(inputs[27], torch.tensor(1e-5)))
        inputs[28] = inputs[28]

        inputs[31] = torch.maximum(np.log10(inputs[31]), torch.tensor(-7.0))
        inputs[32] = 1e8 * inputs[32]

        return inputs, target
