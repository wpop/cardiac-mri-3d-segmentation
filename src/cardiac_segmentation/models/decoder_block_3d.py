from typing import Final, cast

import torch
from torch import Tensor, nn

from cardiac_segmentation.models.double_convolution_block_3d import (
    DoubleConvolutionBlock3D,
)

_SPATIAL_DIMENSION_COUNT: Final[int] = 3
_TENSOR_DIMENSION_COUNT: Final[int] = 5


class DecoderBlock3D(nn.Module):
    """Upsample, concatenate a skip connection, and refine with convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsample_scale: tuple[int, int, int],
    ) -> None:
        """Initialize transposed convolution and refinement block."""
        super().__init__()
        self._validate_channel_count(
            in_channels,
            name="Input channels",
        )
        self._validate_channel_count(
            out_channels,
            name="Output channels",
        )
        self._validate_spatial_triplet(
            upsample_scale,
            name="Upsample scale",
        )
        self._out_channels = out_channels
        self._upsample = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=upsample_scale,
            stride=upsample_scale,
        )
        self._convolutions = DoubleConvolutionBlock3D(
            in_channels=2 * out_channels,
            out_channels=out_channels,
        )

    def forward(
        self,
        input_tensor: Tensor,
        skip_tensor: Tensor,
    ) -> Tensor:
        """Return the decoded tensor after validating the skip connection."""
        self._validate_input_tensors(
            input_tensor=input_tensor,
            skip_tensor=skip_tensor,
        )
        upsampled_tensor = self._upsample(input_tensor)
        self._validate_skip_contract(
            upsampled_tensor=upsampled_tensor,
            skip_tensor=skip_tensor,
        )
        concatenated_tensor = torch.cat(
            (
                upsampled_tensor,
                skip_tensor,
            ),
            dim=1,
        )

        return cast(Tensor, self._convolutions(concatenated_tensor))

    def _validate_input_tensors(
        self,
        input_tensor: Tensor,
        skip_tensor: Tensor,
    ) -> None:
        """Validate tensor rank and batch compatibility before upsampling."""
        if input_tensor.ndim != _TENSOR_DIMENSION_COUNT:
            raise ValueError(
                "Decoder input tensor must be five-dimensional [B, C, D, H, W]."
            )

        if skip_tensor.ndim != _TENSOR_DIMENSION_COUNT:
            raise ValueError(
                "Decoder skip tensor must be five-dimensional [B, C, D, H, W]."
            )

        if input_tensor.shape[0] != skip_tensor.shape[0]:
            raise ValueError(
                "Decoder input and skip tensors must have matching batch dimensions."
            )

    def _validate_skip_contract(
        self,
        upsampled_tensor: Tensor,
        skip_tensor: Tensor,
    ) -> None:
        """Validate skip tensor channels and spatial dimensions."""
        if skip_tensor.shape[1] != self._out_channels:
            raise ValueError(
                "Decoder skip tensor channel count must equal the decoder output "
                f"channels: {skip_tensor.shape[1]} != {self._out_channels}."
            )

        if tuple(upsampled_tensor.shape[2:]) != tuple(skip_tensor.shape[2:]):
            raise ValueError(
                "Decoder upsampled tensor spatial dimensions must match the "
                "skip tensor spatial dimensions."
            )

    @staticmethod
    def _validate_channel_count(
        value: int,
        *,
        name: str,
    ) -> None:
        """Validate a positive channel count."""
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")

    @staticmethod
    def _validate_spatial_triplet(
        value: tuple[int, int, int],
        *,
        name: str,
    ) -> None:
        """Validate a positive three-dimensional integer tuple."""
        if len(value) != _SPATIAL_DIMENSION_COUNT:
            raise ValueError(f"{name} must contain exactly three values.")

        if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value):
            raise ValueError(f"{name} values must be positive integers.")
