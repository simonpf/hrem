"""
Helper functions for evaluting radiative transfer emulators.
"""
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


def flux_to_hr(flux: np.ndarray) -> np.ndarray:
    """
    Convert flux divergence to heating rate.

    Args:
        flux: Array containing the fluxes to convert to heating rates.
    """
    fac = (1 / (1005 * 1.17)) * 86400 * 0.001
    return flux * fac


def simulate_fluxes(
        model: nn.Module,
        inpt: torch.Tensor,
        device="cuda:1",
        dtype=torch.bfloat16
):
    """
    Simulate fluxes for a single input slice.

    Args:
        model: A PyTorch module implementing the emulator.
        inpt: The input tensor.
        device: The device to run the prediction on.
        dtype: The dtype to use for the model.

    Return:
        A numpy array containing the predicted fluxes.
    """
    if inpt.dim() < 5:
        inpt = inpt[None]
    model = model.to(device=device, dtype=dtype)
    inpt = inpt.to(device=device, dtype=dtype)
    with torch.no_grad():
        pred = model(inpt).expected_value().float().cpu().numpy()
    return pred


def evaluate_scene(
    model: nn.Module,
    data_loader: Dataset,
    scene: int,
    device="cuda:1",
    dtype=torch.bfloat16
):
    """
    Calculate fluxes for a full scene, i.e., 192 slices and return COD, and predicted and reference
    fluxes.

    Args:
        model: A PyTorch module implementing the emulator.
        data_loader: A data loader returning the samples from the validation or testing datasets.
        device: The device to run the prediction on.
        dtype: The dtype to use for the model.

    Return:
        A numpy array containing the predicted fluxes.
    """
    start_ind = 192 * scene
    end_ind = start_ind + 192
    hr_pred = []
    hr_true = []
    cod = []
    lwp = []
    god = []

    for ind in range(start_ind, end_ind):
        inpt, ref = data_loader[ind]
        pred = simulate_fluxes(model, inpt, device=device, dtype=dtype)
        inpt = inpt[None]

        cod.append(inpt[0, -4, :, 5].float().numpy())
        lwp.append(inpt[0, -3, :, 5].float().numpy())
        god.append(inpt[0, -1, :, 5].float().numpy())

        hr_pred.append(flux_to_hr(pred.squeeze()[:, 5]))
        hr_true.append(flux_to_hr(ref[0, :, 5]))

    return np.stack(cod), np.stack(lwp), np.stack(god), np.stack(hr_pred), np.stack(hr_true)


def plot_stats_and_hist(
        arrays: List[np.ndarray],
        num_bins: int = 100,
        title=None
):
    """Plot a histogram of a numpy array and display basic statistics.

    Args:
        arrays: The arrays containing the feature data for training, validation, and input splits.
        log_bins (bool, optional): If True, compute histogram using logarithmic bins.
            Defaults to False.
        title: Title to use for the figure panels.

    Returns:
        The matplotlib.Figure object containing the histograms.
    """
    if not isinstance(arrays, list):
        arrays = [arrays]

    # Compute statistics
    stats = {
        'min': [np.nanmin(arr) for arr in arrays],
        'max': [np.nanmax(arr) for arr in arrays],
        'mean': [np.nanmean(arr) for arr in arrays]
    }

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    min_val = min(stats['min'])
    max_val = max(stats['max'])
    bins = np.linspace(min_val, max_val, num_bins + 1)

    names = ["Train", "Validation", "Test"]
    for name, arr in zip(names, arrays):
        y = np.histogram(arr, bins=bins, density=True)[0]
        x = 0.5 * (bins[1:] + bins[:1])
        axs[0].plot(x, y, label=name)
    if title is not None:
        axs[0].set_title(title)
    axs[0].set_xlabel("Value")
    axs[0].set_ylabel("Frequency")
    axs[1].set_yscale('log')

    # Add statistics as text box
    stats_text = ""
    for name, values in stats.items():
        stats_text += f"\n {name}: " + "/".join([f"{stat:.2}" for stat in values])

    axs[0].text(
        0.95, 0.95, stats_text,
        transform=axs[0].transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
    )

    # Handle histogram bins
    log_min = np.log10(min([arr[0 < arr].min() for arr in arrays]))
    log_max = np.log10(max([arr[0 < arr].max() for arr in arrays]))
    bins = np.logspace(log_min, log_max, num_bins + 1)

    for name, arr in zip(names, arrays):
        y = np.histogram(arr, bins=bins, density=True)[0]
        x = 0.5 * (bins[1:] + bins[:1])
        axs[1].plot(x, y, label=name)

    if title is not None:
        axs[1].set_title(title)
    axs[1].set_xlabel("Value")
    axs[1].set_ylabel("Frequency")
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    axs[1].legend()

    plt.tight_layout()
    return fig
import torch
import matplotlib.pyplot as plt
import math


def plot_tensor_slices(
        tensor: torch.Tensor,
        cmap: str = 'viridis',
        ncols: int = 4,
        figsize: Tuple[float, float] = None
):
    """Plot slices of a multi-channel PyTorch tensor.

    This function visualizes each channel of a 3D tensor (C, H, W) as a separate
    subplot in a grid layout, where C is the number of channels. The channel
    dimension is assumed to be the first dimension of the tensor.

    Args:
        tensor (torch.Tensor): A 3D tensor with shape (C, H, W). The function
            will visualize each slice along the first dimension.
        cmap (str, optional): The colormap to apply to each channel slice.
            Defaults to 'viridis'.
        ncols (int, optional): Number of columns to use for the subplot grid.
            Defaults to 4.
        figsize (tuple, optional): Tuple specifying the figure size in inches,
            e.g., (10, 10). If None, it will be inferred from `ncols`.

    Returns:
        tuple:
            - fig (matplotlib.figure.Figure): The figure object for further
                customization or saving.
            - axs (numpy.ndarray of matplotlib.axes.Axes): Array of subplot axes.
    """
    # Prepare tensor
    tensor = tensor.detach().cpu()
    if tensor.dim() < 3:
        raise ValueError("Input tensor must have at least 3 dimensions (C, H, W).")

    C = tensor.shape[0]

    # Compute rows and columns
    nrows = math.ceil(C / ncols)
    if figsize is None:
        figsize = (ncols * 3.2, nrows * 3)

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axs = axs.ravel()

    for i in range(C):
        img = tensor[i]
        if img.dim() > 2:  # If extra dims exist, take the first 2D slice
            img = img[..., 0]

        m = axs[i].pcolormesh(img, cmap=cmap)
        axs[i].set_title(f"Channel {i}")
        plt.colorbar(m)

    # Turn off any unused axes if C < nrows * ncols
    for j in range(C, len(axs)):
        axs[j].axis('off')

    plt.tight_layout()
    plt.show()
    return fig, axs
