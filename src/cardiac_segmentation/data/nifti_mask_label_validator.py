from cardiac_segmentation.data.nifti_mask_statistics import (
    NiftiMaskStatistics,
)


class NiftiMaskLabelValidator:
    """Validate observed mask labels against an allowed label set."""

    def __init__(self, expected_labels: tuple[int, ...]) -> None:
        """Initialize the validator with unique non-negative labels."""
        if not expected_labels:
            raise ValueError(
                "Expected mask labels must not be empty."
            )

        if any(label < 0 for label in expected_labels):
            raise ValueError(
                "Expected mask labels must be non-negative integers."
            )

        if len(set(expected_labels)) != len(expected_labels):
            raise ValueError(
                "Expected mask labels must be unique."
            )

        self._expected_labels = frozenset(expected_labels)

    def validate(self, statistics: NiftiMaskStatistics) -> None:
        """Raise an error when a mask contains unsupported labels."""
        unexpected_labels = tuple(
            label
            for label in statistics.labels
            if label not in self._expected_labels
        )

        if unexpected_labels:
            raise ValueError(
                f"NIfTI mask {statistics.file_path} contains unsupported "
                f"labels {unexpected_labels}. Expected labels are "
                f"{tuple(sorted(self._expected_labels))}."
            )
