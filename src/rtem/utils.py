"""
Helper functions for evaluting radiative transfer emulators.
"""
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
