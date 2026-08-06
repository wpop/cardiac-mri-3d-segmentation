from typing import Final

from torch import Tensor, nn

from cardiac_segmentation.models.double_convolution_block_3d import (
    DoubleConvolutionBlock3D,
)

_SPATIAL_DIMENSION_COUNT: Final[int] = 3


class EncoderBlock3D(nn.Module):
    """Apply a 3D U-Net encoder block and return skip plus pooled tensors."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_size: tuple[int, int, int],
    ) -> None:
        """Initialize convolution and pooling layers."""
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
            pool_size,
            name="Pool size",
        )
        self._convolutions = DoubleConvolutionBlock3D(
            in_channels=in_channels,
            out_channels=out_channels,
        )
        self._pool = nn.MaxPool3d(
            kernel_size=pool_size,
            stride=pool_size,
        )

    def forward(
        self,
        input_tensor: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return the skip tensor and pooled tensor."""
        skip_tensor = self._convolutions(input_tensor)
        pooled_tensor = self._pool(skip_tensor)

        return skip_tensor, pooled_tensor

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
