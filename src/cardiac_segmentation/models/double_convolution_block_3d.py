from typing import cast

from torch import Tensor, nn


class DoubleConvolutionBlock3D(nn.Module):
    """Apply two normalized 3D convolutions with LeakyReLU activations."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        """Initialize the double-convolution block."""
        super().__init__()
        self._validate_channel_count(
            in_channels,
            name="Input channels",
        )
        self._validate_channel_count(
            out_channels,
            name="Output channels",
        )
        self._block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm3d(
                out_channels,
                affine=True,
            ),
            nn.LeakyReLU(
                negative_slope=0.01,
                inplace=True,
            ),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm3d(
                out_channels,
                affine=True,
            ),
            nn.LeakyReLU(
                negative_slope=0.01,
                inplace=True,
            ),
        )

    def forward(
        self,
        input_tensor: Tensor,
    ) -> Tensor:
        """Return the result of the two-convolution block."""
        return cast(Tensor, self._block(input_tensor))

    @staticmethod
    def _validate_channel_count(
        value: int,
        *,
        name: str,
    ) -> None:
        """Validate a positive channel count."""
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
