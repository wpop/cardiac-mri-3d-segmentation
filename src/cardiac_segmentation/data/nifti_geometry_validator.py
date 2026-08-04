from math import isclose

from cardiac_segmentation.data.nifti_volume_metadata import NiftiVolumeMetadata

_DEFAULT_RELATIVE_TOLERANCE = 1e-5
_DEFAULT_ABSOLUTE_TOLERANCE = 1e-6


class NiftiGeometryValidator:
    """Validate spatial geometry consistency between a NIfTI image and mask."""

    def __init__(
        self,
        relative_tolerance: float = _DEFAULT_RELATIVE_TOLERANCE,
        absolute_tolerance: float = _DEFAULT_ABSOLUTE_TOLERANCE,
    ) -> None:
        """Initialize the validator with numerical comparison tolerances."""
        if relative_tolerance < 0.0:
            raise ValueError("Relative tolerance must not be negative.")

        if absolute_tolerance < 0.0:
            raise ValueError("Absolute tolerance must not be negative.")

        self._relative_tolerance = relative_tolerance
        self._absolute_tolerance = absolute_tolerance

    def validate_pair(
        self,
        image_metadata: NiftiVolumeMetadata,
        mask_metadata: NiftiVolumeMetadata,
    ) -> None:
        """Validate that one MRI volume and its mask share identical geometry."""
        self._validate_shape(image_metadata, mask_metadata)
        self._validate_spacing(image_metadata, mask_metadata)
        self._validate_orientation(image_metadata, mask_metadata)
        self._validate_affine(image_metadata, mask_metadata)

    def _validate_shape(
        self,
        image_metadata: NiftiVolumeMetadata,
        mask_metadata: NiftiVolumeMetadata,
    ) -> None:
        """Validate that image and mask voxel-grid dimensions are identical."""
        if image_metadata.shape != mask_metadata.shape:
            raise ValueError(
                "NIfTI shape mismatch: "
                f"image {image_metadata.file_path} has "
                f"{image_metadata.shape}, while mask "
                f"{mask_metadata.file_path} has {mask_metadata.shape}."
            )

    def _validate_spacing(
        self,
        image_metadata: NiftiVolumeMetadata,
        mask_metadata: NiftiVolumeMetadata,
    ) -> None:
        """Validate that image and mask voxel spacing values are equivalent."""
        for axis_index, (image_spacing, mask_spacing) in enumerate(
            zip(
                image_metadata.voxel_spacing,
                mask_metadata.voxel_spacing,
                strict=True,
            )
        ):
            if not self._values_are_close(image_spacing, mask_spacing):
                raise ValueError(
                    "NIfTI voxel-spacing mismatch at axis "
                    f"{axis_index}: image {image_metadata.file_path} has "
                    f"{image_spacing}, while mask {mask_metadata.file_path} "
                    f"has {mask_spacing}."
                )

    def _validate_orientation(
        self,
        image_metadata: NiftiVolumeMetadata,
        mask_metadata: NiftiVolumeMetadata,
    ) -> None:
        """Validate that image and mask anatomical orientations are identical."""
        if image_metadata.orientation != mask_metadata.orientation:
            raise ValueError(
                "NIfTI orientation mismatch: "
                f"image {image_metadata.file_path} has "
                f"{image_metadata.orientation}, while mask "
                f"{mask_metadata.file_path} has "
                f"{mask_metadata.orientation}."
            )

    def _validate_affine(
        self,
        image_metadata: NiftiVolumeMetadata,
        mask_metadata: NiftiVolumeMetadata,
    ) -> None:
        """Validate that image and mask affine transformations are equivalent."""
        for row_index, (image_row, mask_row) in enumerate(
            zip(
                image_metadata.affine,
                mask_metadata.affine,
                strict=True,
            )
        ):
            for column_index, (image_value, mask_value) in enumerate(
                zip(image_row, mask_row, strict=True)
            ):
                if not self._values_are_close(image_value, mask_value):
                    raise ValueError(
                        "NIfTI affine mismatch at position "
                        f"[{row_index}, {column_index}]: image "
                        f"{image_metadata.file_path} has {image_value}, "
                        f"while mask {mask_metadata.file_path} has "
                        f"{mask_value}."
                    )

    def _values_are_close(
        self,
        first_value: float,
        second_value: float,
    ) -> bool:
        """Return whether two geometry values match within configured tolerances."""
        return isclose(
            first_value,
            second_value,
            rel_tol=self._relative_tolerance,
            abs_tol=self._absolute_tolerance,
        )
