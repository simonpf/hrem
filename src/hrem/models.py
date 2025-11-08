"""
hrem.models
===========

This module provide PyTorch model implementation for the HREM study.
"""
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_retrieve.tensors import MeanTensor
from pytorch_retrieve.modules.input import StandardizationLayer
from pytorch_retrieve.modules.output import Mean, Quantiles
from pytorch_retrieve.modules.transformations import Log


class Reflect(nn.Module):
    """
    Pad input by reflecting the input tensor.
    """
    def __init__(self, pad: Union[int, Tuple[int]]):
        """
        Instantiates a padding layer.

        Args:
            pad: N-tuple defining the padding added to the n-last dimensions
                of the tensor. If an int, the same padding will be added to the
                two last dimensions of the tensor.
        """
        super().__init__()
        if isinstance(pad, int):
            pad = (pad,) * 2

        full_pad = []
        for n_elems in pad:
            if isinstance(n_elems, (tuple, list)):
                full_pad += [n_elems[0], n_elems[1]]
            elif isinstance(n_elems, int):
                full_pad += [n_elems, n_elems]
            else:
                raise ValueError(
                    "Expected elements of pad tuple to be tuples of integers or integers. "
                    "Got %s.", type(n_elems)
                )

        full_pad = tuple(full_pad[::-1])
        self.pad = full_pad


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add padding to tensor.

        Args:
            x: The input tensor.

        Return:
            The padded tensor.
        """
        return nn.functional.pad(x, self.pad, "reflect")



def calculate_padding(
        kernel_size: Union[int, Tuple[int]],
        dilation: Union[int, Tuple[int]] = 1
) -> Tuple[int]:
    """
    Calculate padding for a kernel filter with given kernel size and dilation.

    Args:
        kernel_size: The filters kernel isze.
        dilations: The dilation of the kernel.

    Return:
        Tuple specifying the padding to apply along each of the n-last dimensions,
        where n is determined from the length of 'kernel_size' or set to 2 if
        'kernel_size' is an integer.
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * 2
        n_dim = 2
    else:
        n_dim = len(kernel_size)


    if isinstance(dilation, int):
        dilation = (dilation,) * n_dim
    else:
        assert len(dilation) == n_dim


    padding = tuple([
        (s_k - 1) * dil // 2 for s_k, dil in zip(kernel_size, dilation)
    ])

    return padding


def get_projection(
        in_channels: int,
        out_channels: int,
        stride: Tuple[int],
        anti_aliasing: bool = False,
        padding_factory: Callable[[Tuple[int]], nn.Module] = Reflect
):
    """
    Get a projection module that adapts an input tensor to a smaller input tensor that is
    downsampled using the strides defined in 'stride'.

    Args:
        in_channels: The number of channels in the input tensor.
        out_channels: The number of channels in the output tensor.
        stride: The stride by which the input should be downsampled.
        anti_aliasing: Wether or not to apply anti-aliasing before downsampling.
        padding_factory: A factor for producing the padding blocks used in the model.

    Return:
        A projection module to project the input to the dimensions of the output.
    """
    if max(stride) == 1:
        if in_channels == out_channels:
            return nn.Identity()
        if len(stride) == 3:
            return nn.Conv3d(in_channels, out_channels, kernel_size=1)
        return nn.Conv3d(in_channels, out_channels, kernel_size=1)

    blocks = []

    if len(stride) == 3:
        blocks += [
            nn.Conv3d(in_channels, out_channels, kernel_size=stride, stride=stride),
            LayerNorm3d(out_channels)
        ]
    else:
        blocks.append(
            nn.Conv3d(in_channels, out_channels, kernel_size=stride, stride=stride)
        )
    return nn.Sequential(*blocks)


class LayerNorm3d(nn.Module):
    """
    Layer norm performed along the first dimension.
    """

    def __init__(self, n_channels, eps=1e-5):
        """
        Args:
            n_channels: The number of channels in the input.
            eps: Epsilon added to variance to avoid numerical issues. """
        super().__init__()
        self.n_channels = n_channels
        self.scaling = nn.Parameter(torch.ones(n_channels), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(n_channels), requires_grad=True)
        self.eps = eps

    def forward(self, x):
        """
        Apply normalization to x.
        """
        dtype = x.dtype
        mu = x.mean(1, keepdim=True)
        x_n = (x - mu).to(dtype=torch.float32)
        var = x_n.pow(2).mean(1, keepdim=True)
        x_n = x_n / torch.sqrt(var + self.eps)
        shape_ext = (self.n_channels,) + (1,) * (x_n.dim() - 2)
        x = self.scaling.reshape(shape_ext) * x_n.to(dtype=dtype) + self.bias.reshape(shape_ext)
        return x


class Reflect(nn.Module):
    """
    Pad input by reflecting the input tensor.
    """
    def __init__(self, pad: Union[int, Tuple[int]]):
        """
        Instantiates a padding layer.

        Args:
            pad: N-tuple defining the padding added to the n-last dimensions
                of the tensor. If an int, the same padding will be added to the
                two last dimensions of the tensor.
        """
        super().__init__()
        if isinstance(pad, int):
            pad = (pad,) * 2

        full_pad = []
        for n_elems in pad:
            if isinstance(n_elems, (tuple, list)):
                full_pad += [n_elems[0], n_elems[1]]
            elif isinstance(n_elems, int):
                full_pad += [n_elems, n_elems]
            else:
                raise ValueError(
                    "Expected elements of pad tuple to be tuples of integers or integers. "
                    "Got %s.", type(n_elems)
                )

        full_pad = tuple(full_pad[::-1])
        self.pad = full_pad


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add padding to tensor.

        Args:
            x: The input tensor.

        Return:
            The padded tensor.
        """
        return nn.functional.pad(x, self.pad, "reflect")


class InvertedBottleneckBlock(nn.Module):
    """
    Inverted-bottleneck block is used in MobileNet and Efficient net where it is referred
    to as MBConv
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            expansion_factor: int = 4,
            kernel_size: int = 3,
            activation_factory: Callable[[], nn.Module] = nn.GELU,
            normalization_factory: Callable[[int], nn.Module] = LayerNorm3d,
            padding: Optional[Tuple[int]] = None,
            padding_factory: Callable[[Union[Tuple[int], int]], nn.Module] = Reflect,
            downsample: Optional[int] = None,
            fused: bool = False,
    ):
        super().__init__()
        self.act = activation_factory()
        act = activation_factory()

        hidden_channels = int(out_channels * expansion_factor)

        stride = (1, 1, 1)
        if downsample is not None:
            if isinstance(downsample, int):
                downsample = (downsample,) * 2
            if max(downsample) > 1:
                stride = downsample

        self.projection = get_projection(
            in_channels,
            out_channels,
            stride=stride,
            padding_factory=padding_factory
        )

        if padding is None:
            padding = calculate_padding(kernel_size)



        blocks = []
        if not fused:

            blocks += [
                nn.Conv3d(in_channels, hidden_channels, kernel_size=1),
                normalization_factory(hidden_channels),
                act
            ]
            if max(stride) == 1:
                blocks += [padding_factory(padding)]
            blocks += [
                nn.Conv3d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=kernel_size if (max(stride) < 2) else stride,
                    stride=stride,
                    groups=hidden_channels,
                ),
                normalization_factory(hidden_channels),
                act
            ]
        else:
            blocks += [
                padding_factory(padding),
                nn.Conv3d(
                    in_channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                ),
                normalization_factory(hidden_channels),
                act
            ]

        blocks += [
            nn.Conv3d(hidden_channels, out_channels, kernel_size=1),
            normalization_factory(out_channels),
            act
        ]
        self.body = nn.Sequential(*blocks)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagate input through layer.
        """
        shortcut = self.projection(x)

        return shortcut + self.body(x)


class HeatingRateUNet3dSimple(nn.Module):
    """
    Simple 3D U-Net for predicting vertical profiles of heating rates.
    
    Based on the architecture described in JAMES-2025-3DEmulator-v2.pdf:
    - Input: Vertical cross-sections of direct flux, gas/aerosol optical depth, cloud optical depth
    - Output: Cross-sections of flux divergence (heating rates)
    - Architecture: 3D U-Net with depth 3, 16 filters, LeakyReLU activation
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 1,
        filters: int = 16,
        leaky_slope: float = 0.3
    ):
        """
        Initialize the heating rate 3D U-Net.
        
        Args:
            input_channels: Number of input channels (3: direct flux, gas/aerosol OD, cloud OD)
            output_channels: Number of output channels (1: flux divergence)
            filters: Number of filters throughout the network
            leaky_slope: Negative slope for LeakyReLU activation
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.filters = filters
        self.leaky_slope = leaky_slope

        # Encoder (downsampling path)
        self.enc_conv1_1 = nn.Conv3d(input_channels, filters, 3, padding=1)
        self.enc_norm1_1 = LayerNorm3d(filters)
        self.enc_conv1_2 = nn.Conv3d(filters, filters, 3, padding=1)
        self.enc_norm1_2 = LayerNorm3d(filters)
        self.enc_conv1_3 = nn.Conv3d(filters, filters, 3, padding=1)
        self.enc_norm1_3 = LayerNorm3d(filters)
        self.pool1 = nn.MaxPool3d(kernel_size=(5, 1, 5), stride=(5, 1, 5))
        
        self.enc_conv2_1 = nn.Conv3d(filters, filters, 3, padding=1)
        self.enc_norm2_1 = LayerNorm3d(filters)
        self.enc_conv2_2 = nn.Conv3d(filters, filters, 3, padding=1)
        self.enc_norm2_2 = LayerNorm3d(filters)
        self.enc_conv2_3 = nn.Conv3d(filters, filters, 3, padding=1)
        self.enc_norm2_3 = LayerNorm3d(filters)
        self.pool2 = nn.MaxPool3d(kernel_size=(5, 1, 3), stride=(5, 1, 3))
        
        self.enc_conv3_1 = nn.Conv3d(filters, filters, 3, padding=1)
        self.enc_norm3_1 = LayerNorm3d(filters)
        self.enc_conv3_2 = nn.Conv3d(filters, filters, 3, padding=1)
        self.enc_norm3_2 = LayerNorm3d(filters)
        self.enc_conv3_3 = nn.Conv3d(filters, filters, 3, padding=1)
        self.enc_norm3_3 = LayerNorm3d(filters)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 1, 2), stride=(2, 1, 2))
        
        # Decoder (upsampling path)
        self.up3 = nn.Upsample(scale_factor=(2, 1, 2), mode='nearest')
        self.dec_conv3_1 = nn.Conv3d(filters * 2, filters, 3, padding=1)
        self.dec_norm3_1 = LayerNorm3d(filters)
        self.dec_conv3_2 = nn.Conv3d(filters, filters, 3, padding=1)
        self.dec_norm3_2 = LayerNorm3d(filters)
        self.dec_conv3_3 = nn.Conv3d(filters, filters, 3, padding=1)
        self.dec_norm3_3 = LayerNorm3d(filters)
        
        self.up2 = nn.Upsample(scale_factor=(5, 1, 3), mode='nearest')
        self.dec_conv2_1 = nn.Conv3d(filters * 2, filters, 3, padding=1)
        self.dec_norm2_1 = LayerNorm3d(filters)
        self.dec_conv2_2 = nn.Conv3d(filters, filters, 3, padding=1)
        self.dec_norm2_2 = LayerNorm3d(filters)
        self.dec_conv2_3 = nn.Conv3d(filters, filters, 3, padding=1)
        self.dec_norm2_3 = LayerNorm3d(filters)
        
        self.up1 = nn.Upsample(scale_factor=(5, 1, 5), mode='nearest')
        self.dec_conv1_1 = nn.Conv3d(filters * 2, filters, 3, padding=1)
        self.dec_norm1_1 = LayerNorm3d(filters)
        self.dec_conv1_2 = nn.Conv3d(filters, filters, 3, padding=1)
        self.dec_norm1_2 = LayerNorm3d(filters)
        self.dec_conv1_3 = nn.Conv3d(filters, filters, 3, padding=1)
        self.dec_norm1_3 = LayerNorm3d(filters)
        
        # Final output layer
        self.final_conv = nn.Conv3d(filters, output_channels, 3, padding=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the 3D U-Net.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, depth, width)
               Channel 0: δ-scaled direct flux
               Channel 1: gas and aerosol optical depth
               Channel 2: cloud optical depth
               
        Returns:
            Output tensor of shape (batch_size, 1, height, depth, width)
            Flux divergence (heating rates)
        """
        # Encoder path
        # Level 1
        x1 = F.gelu(self.enc_norm1_1(self.enc_conv1_1(x)))
        x1 = x1 + F.gelu(self.enc_norm1_2(self.enc_conv1_2(x1)))
        x1 = x1 + F.gelu(self.enc_norm1_2(self.enc_conv1_3(x1)))
        p1 = self.pool1(x1)
        
        # Level 2
        x2 = F.gelu(self.enc_norm2_1(self.enc_conv2_1(p1)))
        x2 = x2 + F.gelu(self.enc_norm2_2(self.enc_conv2_2(x2)))
        x2 = x2 + F.gelu(self.enc_norm2_3(self.enc_conv2_3(x2)))
        p2 = self.pool2(x2)
        
        # Level 3
        x3 = F.gelu(self.enc_norm3_1(self.enc_conv3_1(p2)))
        x3 = x3 + F.gelu(self.enc_norm3_2(self.enc_conv3_2(x3)))
        x3 = x3 + F.gelu(self.enc_norm3_3(self.enc_conv3_3(x3)))
        p3 = self.pool3(x3)
        
        # Decoder path
        # Level 3
        u3 = self.up3(p3)
        if u3.shape[2:] != x3.shape[2:]:
            u3 = F.interpolate(u3, size=x3.shape[2:], mode='nearest')
        u3 = torch.cat([u3, x3], dim=1)
        u3 = F.gelu(self.dec_norm3_1(self.dec_conv3_1(u3)))
        u3 = u3 + F.gelu(self.dec_norm3_2(self.dec_conv3_2(u3)))
        u3 = u3 + F.gelu(self.dec_norm3_3(self.dec_conv3_3(u3)))
        
        # Level 2
        u2 = self.up2(u3)
        if u2.shape[2:] != x2.shape[2:]:
            u2 = F.interpolate(u2, size=x2.shape[2:], mode='nearest')
        u2 = torch.cat([u2, x2], dim=1)
        u2 = F.gelu(self.dec_norm2_1(self.dec_conv2_1(u2)))
        u2 = u2 + F.gelu(self.dec_norm2_2(self.dec_conv2_2(u2)))
        u2 = u2 + F.gelu(self.dec_norm2_3(self.dec_conv2_3(u2)))
        
        # Level 1
        u1 = self.up1(u2)
        if u1.shape[2:] != x1.shape[2:]:
            u1 = F.interpolate(u1, size=x1.shape[2:], mode='nearest')
        u1 = torch.cat([u1, x1], dim=1)
        u1 = F.gelu(self.dec_norm1_1(self.dec_conv1_1(u1)))
        u1 = u1 + F.gelu(self.dec_norm1_2(self.dec_conv1_2(u1)))
        u1 = u1 + F.gelu(self.dec_norm1_3(self.dec_conv1_3(u1)))
        
        # Final output
        output = self.final_conv(u1)

        return MeanTensor(output)


class HeatingRateUNet3d(nn.Module):
    """
    3D ResNeXt network for predicting vertical profiles of heating rates.

    Uses ResNeXt blocks with separable 3D convolutions, GELU activation functions,
    and LayerNorm normalizations for improved performance and stability.
    """

    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 1,
        base_filters: int = 32,
        cardinality: int = 32,
        base_width: int = 4,
        num_blocks_per_stage: int = 2,
        transform: bool = False
    ):
        """
        Initialize the heating rate ResNeXt 3D network.

        Args:
            input_channels: Number of input channels (3: direct flux, gas/aerosol OD, cloud OD)
            output_channels: Number of output channels (1: flux divergence)
            base_filters: Base number of filters
            cardinality: Number of groups for ResNeXt blocks
            base_width: Base width for ResNeXt blocks
            num_blocks_per_stage: Number of ResNeXt blocks per stage
        """
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.base_filters = base_filters
        self.cardinality = cardinality
        self.base_width = base_width
        self.transform = transform

        # Initial convolution
        self.initial_conv = nn.Sequential(
            StandardizationLayer("forcing", input_channels),
            self._make_stage(input_channels, 32, 1, 1.0, None, True),
        )

        # Encoder stages with ResNeXt blocks
        self.encode1 = self._make_stage(32, 48, 2, 2.0, (2, 2, 2), True)
        self.encode2 = self._make_stage(48, 64, 4, 4.0, (5, 2, 3), False)
        self.encode3 = self._make_stage(64, 128, 6, 6.0, (5, 1, 5), False)

        self.up3 = nn.Upsample(scale_factor=(5, 1, 5), mode="trilinear")
        self.decode3 = self._make_stage(128 + 64, 64, 4, 4.0, None, False)

        self.up2 = nn.Upsample(scale_factor=(5, 2, 3), mode="trilinear")
        self.decode2 = self._make_stage(64 + 48, 48, 2, 2.0, None, True)

        self.up1 = nn.Upsample(scale_factor=(2, 2, 2), mode="trilinear")
        self.decode1 = self._make_stage(48 + 32, 32, 2, 2.0, None, True)

        quantiles = np.linspace(0, 1, 32)
        quantiles[0] = 1e-2
        quantiles[-1] = 1e-2
        # Final output layer
        if self.transform:
            self.final_conv = nn.Sequential(*[
                nn.Conv3d(32, 1, kernel_size=1),
                Mean("flux_divergence", (1,), transformation=Log()),
                #Quantiles("flux_divergence", (1,), torch.tensor(quantiles))
            ])
        else:
            self.final_conv = nn.Sequential(*[
                nn.Conv3d(32, 1, kernel_size=1),
                Mean("flux_divergence", (1,)),
                #Quantiles("flux_divergence", (1,), torch.tensor(quantiles))
            ])


    def _make_stage(
            self,
            in_channels: int,
            out_channels: int,
            num_blocks: int,
            expansion_factor: float = 4.0,
            downsample: Optional[Tuple[int, int, int]] = None,
            fused: bool = False
    ) -> nn.Sequential:
        """
        Create a stage with multiple ResNeXt blocks.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            num_blocks: Number of ResNeXt blocks in the stage

        Returns:
            Sequential module containing ResNeXt blocks
        """
        layers = []

        # First block handles channel dimension change
        layers.append(InvertedBottleneckBlock(
            in_channels,
            out_channels,
            expansion_factor=expansion_factor,
            downsample=downsample,
            fused=fused,
            padding=(1, 1, 1)
        ))

        # Remaining blocks maintain the same dimensions
        for _ in range(num_blocks - 1):
            layers.append(InvertedBottleneckBlock(
                out_channels, out_channels,
                expansion_factor=expansion_factor,
                fused=fused,
                padding=(1, 1, 1)
            ))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ResNeXt 3D network.

        Args:
            x: Input tensor of shape (batch_size, 3, depth, height, width)
               Channel 0: δ-scaled direct flux
               Channel 1: gas and aerosol optical depth
               Channel 2: cloud optical depth

        Returns:
            Output tensor of shape (batch_size, 1, depth, height, width)
            Flux divergence (heating rates)
        """
        # Initial convolution
        skip1 = self.initial_conv(x)
        skip1 =  nn.functional.pad(skip1, (0, 0, 1, 0, 0, 0), "reflect")

        # Encoder path with skip connections
        # Stage 1
        skip2 = self.encode1(skip1)
        skip3 = self.encode2(skip2)
        skip4 = self.encode3(skip3)

        x = self.up3(skip4)
        x = self.decode3(torch.cat([x, skip3], 1))
        x = self.up2(x)
        x = self.decode2(torch.cat([x, skip2], 1))
        x = self.up1(x)
        x = self.decode1(torch.cat([x, skip1], 1))

        x = x[:, :, :, 1:]
        output = self.final_conv(x)

        return output

__all__ = [
    "HeatingRateUNet3dSimple",
    "HeatingRateUNet3d",
]
