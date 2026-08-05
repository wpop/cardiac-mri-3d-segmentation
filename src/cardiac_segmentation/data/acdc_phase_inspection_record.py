from dataclasses import dataclass
from typing import Final

from cardiac_segmentation.data.nifti_mask_statistics import (
    NiftiMaskStatistics,
)
from cardiac_segmentation.data.nifti_volume_metadata import (
    NiftiVolumeMetadata,
)

_VALID_PHASE_NAMES: Final[frozenset[str]] = frozenset({"ED", "ES"})


@dataclass(frozen=True, slots=True)
class AcdcPhaseInspectionRecord:
    """Store inspection results for one ACDC cardiac phase."""

    phase_name: str
    image_metadata: NiftiVolumeMetadata
    mask_metadata: NiftiVolumeMetadata
    mask_statistics: NiftiMaskStatistics

    def __post_init__(self) -> None:
        """Validate the phase name and relationships between stored files."""
        if self.phase_name not in _VALID_PHASE_NAMES:
            raise ValueError(
                "ACDC phase name must be either 'ED' or 'ES'."
            )

        if (
            self.mask_metadata.file_path
            != self.mask_statistics.file_path
        ):
            raise ValueError(
                "Mask metadata and mask statistics must reference "
                "the same NIfTI file."
            )

        if (
            self.image_metadata.file_path
            == self.mask_metadata.file_path
        ):
            raise ValueError(
                "Image and mask metadata must reference different files."
            )
