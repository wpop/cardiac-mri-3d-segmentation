from typing import Final, cast

from torch import Tensor, nn

from cardiac_segmentation.models.decoder_block_3d import DecoderBlock3D
from cardiac_segmentation.models.double_convolution_block_3d import (
    DoubleConvolutionBlock3D,
)
from cardiac_segmentation.models.encoder_block_3d import EncoderBlock3D

_TENSOR_DIMENSION_COUNT: Final[int] = 5
_DEPTH_DIVISIBILITY: Final[int] = 4
_IN_PLANE_DIVISIBILITY: Final[int] = 8


class CompactUNet3D(nn.Module):
    """Compact anisotropic 3D U-Net for multiclass cardiac MRI segmentation."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        base_channels: int = 8,
    ) -> None:
        """Initialize encoder, bottleneck, decoder, and segmentation head."""
        super().__init__()
        self._validate_positive_integer(
            in_channels,
            name="Input channels",
        )
        self._validate_num_classes(num_classes)
        self._validate_positive_integer(
            base_channels,
            name="Base channels",
        )
        self._in_channels = in_channels
        self._encoder_level_1 = EncoderBlock3D(
            in_channels=in_channels,
            out_channels=base_channels,
            pool_size=(1, 2, 2),
        )
        self._encoder_level_2 = EncoderBlock3D(
            in_channels=base_channels,
            out_channels=2 * base_channels,
            pool_size=(2, 2, 2),
        )
        self._encoder_level_3 = EncoderBlock3D(
            in_channels=2 * base_channels,
            out_channels=4 * base_channels,
            pool_size=(2, 2, 2),
        )
        self._bottleneck = DoubleConvolutionBlock3D(
            in_channels=4 * base_channels,
            out_channels=8 * base_channels,
        )
        self._decoder_level_3 = DecoderBlock3D(
            in_channels=8 * base_channels,
            out_channels=4 * base_channels,
            upsample_scale=(2, 2, 2),
        )
        self._decoder_level_2 = DecoderBlock3D(
            in_channels=4 * base_channels,
            out_channels=2 * base_channels,
            upsample_scale=(2, 2, 2),
        )
        self._decoder_level_1 = DecoderBlock3D(
            in_channels=2 * base_channels,
            out_channels=base_channels,
            upsample_scale=(1, 2, 2),
        )
        self._segmentation_head = nn.Conv3d(
            base_channels,
            num_classes,
            kernel_size=1,
        )

    def forward(
        self,
        input_tensor: Tensor,
    ) -> Tensor:
        """Return raw multiclass logits without activation."""
        self._validate_input_tensor(input_tensor)
        skip_level_1, pooled_level_1 = self._encoder_level_1(input_tensor)
        skip_level_2, pooled_level_2 = self._encoder_level_2(pooled_level_1)
        skip_level_3, pooled_level_3 = self._encoder_level_3(pooled_level_2)
        bottleneck_tensor = self._bottleneck(pooled_level_3)
        decoded_level_3 = self._decoder_level_3(
            input_tensor=bottleneck_tensor,
            skip_tensor=skip_level_3,
        )
        decoded_level_2 = self._decoder_level_2(
            input_tensor=decoded_level_3,
            skip_tensor=skip_level_2,
        )
        decoded_level_1 = self._decoder_level_1(
            input_tensor=decoded_level_2,
            skip_tensor=skip_level_1,
        )

        return cast(Tensor, self._segmentation_head(decoded_level_1))

    def _validate_input_tensor(
        self,
        input_tensor: Tensor,
    ) -> None:
        """Validate model input shape before executing the network."""
        if input_tensor.ndim != _TENSOR_DIMENSION_COUNT:
            raise ValueError("Model input tensor must be five-dimensional [B, C, D, H, W].")

        batch_size = int(input_tensor.shape[0])
        channel_count = int(input_tensor.shape[1])
        depth = int(input_tensor.shape[2])
        height = int(input_tensor.shape[3])
        width = int(input_tensor.shape[4])

        if batch_size <= 0:
            raise ValueError("Model input batch dimension must be positive.")

        if channel_count != self._in_channels:
            raise ValueError(
                f"Model input channel count must be {self._in_channels}, "
                f"but received {channel_count}."
            )

        if depth <= 0 or height <= 0 or width <= 0:
            raise ValueError("Model input spatial dimensions must be positive.")

        if depth % _DEPTH_DIVISIBILITY != 0:
            raise ValueError(
                f"Model input depth must be divisible by {_DEPTH_DIVISIBILITY}."
            )

        if height % _IN_PLANE_DIVISIBILITY != 0:
            raise ValueError(
                f"Model input height must be divisible by {_IN_PLANE_DIVISIBILITY}."
            )

        if width % _IN_PLANE_DIVISIBILITY != 0:
            raise ValueError(
                f"Model input width must be divisible by {_IN_PLANE_DIVISIBILITY}."
            )

    @staticmethod
    def _validate_positive_integer(
        value: int,
        *,
        name: str,
    ) -> None:
        """Validate a positive integer constructor value."""
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")

    @staticmethod
    def _validate_num_classes(
        num_classes: int,
    ) -> None:
        """Validate the output class count."""
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes <= 1:
            raise ValueError("Number of classes must be an integer greater than one.")
