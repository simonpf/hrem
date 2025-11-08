"""
RT3DDataset class for loading 3D heating rate data from LASSO zarr files.

This module provides a PyTorch dataset interface for loading and processing
LASSO zarr data for training 3D heating rate emulators. The data structure is based
on the create_data.py script which creates zarr files with CNN_input and CNN_output arrays.
"""

import os
import zarr
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional, Union, Tuple, List
import warnings


class RT3DDataset(Dataset):
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
        zarr_file_path: str,
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
        cache_in_memory: bool = False,
        validation: bool = True
    ):
        """
        Initialize the RT3DDataset.
        
        Args:
            zarr_file_path: Path to the zarr file (HR_train_patches.zarr, HR_val_patches.zarr, HR_test_patches.zarr)
            transform: Optional transform to apply to inputs
            target_transform: Optional transform to apply to targets
            cache_in_memory: Whether to cache data in memory for faster access
        """
        self.zarr_file_path = zarr_file_path
        self.transform = transform
        self.target_transform = target_transform
        self.cache_in_memory = cache_in_memory
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
            # Open zarr store using zarr v3
            self.store = zarr.open(self.zarr_file_path, mode='r')
            
            # Check for required arrays (based on create_data.py)
            if 'CNN_input' not in self.store:
                raise KeyError("CNN_input array not found in zarr file")
            if 'CNN_output' not in self.store:
                raise KeyError("CNN_output array not found in zarr file")
                
            self.cnn_input = self.store['CNN_input']
            self.cnn_output = self.store['CNN_output']
            
            # Determine dataset length
            self.length = self.cnn_input.shape[0]
            
            # Store channel information
            self.input_channels = ['cloud_optical_depth', 'downwelling_direct_flux', 'gas_aerosol_optical_depth']
            self.output_channels = ['flux_divergence']
                
        except Exception as e:
            raise RuntimeError(f"Failed to load zarr file {self.zarr_file_path}: {e}")
    
    
    def _cache_data(self):
        """Cache all data in memory for faster access."""
        print(f"Caching {len(self)} samples in memory...")
        self._cached_data = []
        
        for idx in range(len(self)):
            sample = self._get_sample(idx)
            self._cached_data.append(sample)
        
        print("Data cached successfully.")
    
    def _get_sample(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single heating rate sample from the dataset."""
        # Load input data: shape (250, 11, 150, 3)
        inputs = np.array(self.cnn_input[idx])
        
        # Load target data: shape (250, 11, 150, 1)
        target = np.array(self.cnn_output[idx])
        
        
        # Transpose to PyTorch format: (C, H, W, D) from (H, W, D, C)
        # Where H=250 (height), W=11 (width), D=150 (depth), C=3 (channels)
        inputs = np.transpose(inputs, (3, 0, 1, 2))  # (3, 250, 11, 150)
        target = np.transpose(target, (3, 0, 1, 2))  # (1, 250, 11, 150)
        
        # Convert to tensors
        inputs = torch.from_numpy(inputs).float()
        target = torch.from_numpy(target).float()
        
        return inputs, target
    
    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self.length
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")
        
        # Use cached data if available
        if self._cached_data is not None:
            inputs, target = self._cached_data[idx]
        else:
            inputs, target = self._get_sample(idx)
        
        # Apply transforms
        if self.transform is not None:
            inputs = self.transform(inputs)

        if not self.validation:
            if np.random.rand() < 0.5:
                input = torch.flip(inputs, (2,))
            if np.random.rand() < 0.5:
                input = torch.flip(inputs, (1,))

        inputs[4] = 1e3 * inputs[4]
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
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


def create_rt3d_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    cache_in_memory: bool = False,
    **kwargs
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create PyTorch DataLoaders for heating rate training, validation, and testing.
    
    Args:
        data_dir: Directory containing HR_train_patches.zarr, HR_val_patches.zarr, HR_test_patches.zarr
        batch_size: Batch size for DataLoaders
        num_workers: Number of worker processes for data loading
        cache_in_memory: Whether to cache data in memory
        **kwargs: Additional arguments passed to DataLoader
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader
    
    # Create datasets
    train_dataset = RT3DDataset(
        zarr_file_path=os.path.join(data_dir, "HR_train_patches.zarr"),
        cache_in_memory=cache_in_memory
    )
    
    val_dataset = RT3DDataset(
        zarr_file_path=os.path.join(data_dir, "HR_val_patches.zarr"),
        cache_in_memory=cache_in_memory
    )
    
    test_dataset = RT3DDataset(
        zarr_file_path=os.path.join(data_dir, "HR_test_patches.zarr"),
        cache_in_memory=cache_in_memory
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        **kwargs
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        **kwargs
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        **kwargs
    )
    
    return train_loader, val_loader, test_loader


class RT3DLogDataset(Dataset):
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
        zarr_file_path: str,
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
        cache_in_memory: bool = False
    ):
        """
        Initialize the RT3DDataset.

        Args:
            zarr_file_path: Path to the zarr file (HR_train_patches.zarr, HR_val_patches.zarr, HR_test_patches.zarr)
            transform: Optional transform to apply to inputs
            target_transform: Optional transform to apply to targets
            cache_in_memory: Whether to cache data in memory for faster access
        """
        self.zarr_file_path = zarr_file_path
        self.transform = transform
        self.target_transform = target_transform
        self.cache_in_memory = cache_in_memory

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
            # Open zarr store using zarr v3
            self.store = zarr.open(self.zarr_file_path, mode='r')

            # Check for required arrays (based on create_data.py)
            if 'CNN_input' not in self.store:
                raise KeyError("CNN_input array not found in zarr file")
            if 'CNN_output' not in self.store:
                raise KeyError("CNN_output array not found in zarr file")

            self.cnn_input = self.store['CNN_input']
            self.cnn_output = self.store['CNN_output']

            # Verify expected shapes
            expected_input_shape = (None, 250, 11, 150, 3)  # (N, height, width, depth, channels)
            expected_output_shape = (None, 250, 11, 150, 1)

            # Determine dataset length
            self.length = self.cnn_input.shape[0]

            # Store channel information
            self.input_channels = ['cloud_optical_depth', 'downwelling_direct_flux', 'gas_aerosol_optical_depth']
            self.output_channels = ['flux_divergence']

        except Exception as e:
            raise RuntimeError(f"Failed to load zarr file {self.zarr_file_path}: {e}")


    def _cache_data(self):
        """Cache all data in memory for faster access."""
        print(f"Caching {len(self)} samples in memory...")
        self._cached_data = []

        for idx in range(len(self)):
            sample = self._get_sample(idx)
            self._cached_data.append(sample)

        print("Data cached successfully.")

    def _get_sample(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single heating rate sample from the dataset."""
        # Load input data: shape (250, 11, 150, 3)
        inputs = np.array(self.cnn_input[idx])

        # Load target data: shape (250, 11, 150, 1)
        target = np.array(self.cnn_output[idx])


        # Transpose to PyTorch format: (C, H, W, D) from (H, W, D, C)
        # Where H=250 (height), W=11 (width), D=150 (depth), C=3 (channels)
        inputs = np.transpose(inputs, (3, 0, 1, 2))  # (3, 250, 11, 150)
        target = np.transpose(target, (3, 0, 1, 2))  # (1, 250, 11, 150)

        # Convert to tensors
        inputs = torch.from_numpy(inputs).float()
        target = torch.from_numpy(target).float()

        return inputs, target

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")

        # Use cached data if available
        if self._cached_data is not None:
            inputs, target = self._cached_data[idx]
        else:
            inputs, target = self._get_sample(idx)

        inputs[0] = torch.maximum(np.log10(inputs[0]), torch.tensor(-10.0))
        inputs[1] = torch.maximum(np.log10(inputs[1]), torch.tensor(-10.0))
        inputs[2] = torch.maximum(np.log10(inputs[2]), torch.tensor(-10.0))
        inputs[3] = torch.maximum(np.log10(inputs[3]), torch.tensor(-10.0))
        inputs[4] = 1e3 * inputs[4]

        # Apply transforms
        if self.transform is not None:
            inputs = self.transform(inputs)
        
        if self.target_transform is not None:
            target = self.target_transform(target)

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


def create_rt3d_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    cache_in_memory: bool = False,
    **kwargs
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create PyTorch DataLoaders for heating rate training, validation, and testing.

    Args:
        data_dir: Directory containing HR_train_patches.zarr, HR_val_patches.zarr, HR_test_patches.zarr
        batch_size: Batch size for DataLoaders
        num_workers: Number of worker processes for data loading
        cache_in_memory: Whether to cache data in memory
        **kwargs: Additional arguments passed to DataLoader

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader

    # Create datasets
    train_dataset = RT3DDataset(
        zarr_file_path=os.path.join(data_dir, "HR_train_patches.zarr"),
        cache_in_memory=cache_in_memory
    )

    val_dataset = RT3DDataset(
        zarr_file_path=os.path.join(data_dir, "HR_val_patches.zarr"),
        cache_in_memory=cache_in_memory
    )

    test_dataset = RT3DDataset(
        zarr_file_path=os.path.join(data_dir, "HR_test_patches.zarr"),
        cache_in_memory=cache_in_memory
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        **kwargs
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        **kwargs
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        **kwargs
    )

    return train_loader, val_loader, test_loader


class RT3DSplitBandDataset(Dataset):
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
        zarr_file_path: str,
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
        cache_in_memory: bool = False
    ):
        """
        Initialize the RT3DDataset.

        Args:
            zarr_file_path: Path to the zarr file (HR_train_patches.zarr, HR_val_patches.zarr, HR_test_patches.zarr)
            transform: Optional transform to apply to inputs
            target_transform: Optional transform to apply to targets
            cache_in_memory: Whether to cache data in memory for faster access
        """
        self.zarr_file_path = zarr_file_path
        self.transform = transform
        self.target_transform = target_transform
        self.cache_in_memory = cache_in_memory

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
            # Open zarr store using zarr v3
            self.store = zarr.open(self.zarr_file_path, mode='r')

            # Check for required arrays (based on create_data.py)
            if 'CNN_input' not in self.store:
                raise KeyError("CNN_input array not found in zarr file")
            if 'CNN_output' not in self.store:
                raise KeyError("CNN_output array not found in zarr file")

            self.cnn_input = self.store['CNN_input']
            self.cnn_output = self.store['CNN_output']

            # Verify expected shapes
            expected_input_shape = (None, 250, 11, 150, 3)  # (N, height, width, depth, channels)
            expected_output_shape = (None, 250, 11, 150, 1)

            # Determine dataset length
            self.length = self.cnn_input.shape[0]

            # Store channel information
            self.input_channels = ['cloud_optical_depth', 'downwelling_direct_flux', 'gas_aerosol_optical_depth']
            self.output_channels = ['flux_divergence']

        except Exception as e:
            raise RuntimeError(f"Failed to load zarr file {self.zarr_file_path}: {e}")


    def _cache_data(self):
        """Cache all data in memory for faster access."""
        print(f"Caching {len(self)} samples in memory...")
        self._cached_data = []

        for idx in range(len(self)):
            sample = self._get_sample(idx)
            self._cached_data.append(sample)

        print("Data cached successfully.")

    def _get_sample(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single heating rate sample from the dataset."""
        # Load input data: shape (250, 11, 150, 3)
        inputs = np.array(self.cnn_input[idx])

        # Load target data: shape (250, 11, 150, 1)
        target = np.array(self.cnn_output[idx])


        # Transpose to PyTorch format: (C, H, W, D) from (H, W, D, C)
        # Where H=250 (height), W=11 (width), D=150 (depth), C=3 (channels)
        inputs = np.transpose(inputs, (3, 0, 1, 2))  # (3, 250, 11, 150)
        target = np.transpose(target, (3, 0, 1, 2))  # (1, 250, 11, 150)

        # Convert to tensors
        inputs = torch.from_numpy(inputs).float()
        target = torch.from_numpy(target).float()

        return inputs, target

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")

        # Use cached data if available
        if self._cached_data is not None:
            inputs, target = self._cached_data[idx]
        else:
            inputs, target = self._get_sample(idx)

        for ind in range(15):
            inputs[ind] = torch.maximum(np.log10(inputs[ind]), torch.tensor(-7.0))
        inputs[15] = torch.maximum(np.log10(inputs[15]), torch.tensor(-3.0))
        inputs[16] = torch.maximum(np.log10(inputs[16]), torch.tensor(-5.0))
        inputs[17] = torch.maximum(np.log10(inputs[17]), torch.tensor(-7.0))
        #inputs[18] = torch.maximum(np.log10(inputs[18]), torch.tensor(-10.0))
        inputs[18] = 1e3 * inputs[18]

        # Apply transforms
        if self.transform is not None:
            inputs = self.transform(inputs)

        if self.target_transform is not None:
            target = self.target_transform(target)

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




def create_rt3d_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    cache_in_memory: bool = False,
    **kwargs
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create PyTorch DataLoaders for heating rate training, validation, and testing.

    Args:
        data_dir: Directory containing HR_train_patches.zarr, HR_val_patches.zarr, HR_test_patches.zarr
        batch_size: Batch size for DataLoaders
        num_workers: Number of worker processes for data loading
        cache_in_memory: Whether to cache data in memory
        **kwargs: Additional arguments passed to DataLoader

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader

    # Create datasets
    train_dataset = RT3DDataset(
        zarr_file_path=os.path.join(data_dir, "HR_train_patches.zarr"),
        cache_in_memory=cache_in_memory
    )

    val_dataset = RT3DDataset(
        zarr_file_path=os.path.join(data_dir, "HR_val_patches.zarr"),
        cache_in_memory=cache_in_memory
    )

    test_dataset = RT3DDataset(
        zarr_file_path=os.path.join(data_dir, "HR_test_patches.zarr"),
        cache_in_memory=cache_in_memory
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        **kwargs
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        **kwargs
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        **kwargs
    )

    return train_loader, val_loader, test_loader
