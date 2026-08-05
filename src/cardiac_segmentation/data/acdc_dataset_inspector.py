from pathlib import Path

from cardiac_segmentation.config.app_config import AppConfig
from cardiac_segmentation.data.acdc_dataset_inspection_report import (
    AcdcDatasetInspectionReport,
)
from cardiac_segmentation.data.acdc_indexer import AcdcDatasetIndexer
from cardiac_segmentation.data.acdc_info_parser import AcdcInfoParser
from cardiac_segmentation.data.acdc_patient_case import AcdcPatientCase
from cardiac_segmentation.data.acdc_patient_inspection_record import (
    AcdcPatientInspectionRecord,
)
from cardiac_segmentation.data.acdc_phase_inspection_record import (
    AcdcPhaseInspectionRecord,
)
from cardiac_segmentation.data.nifti_geometry_validator import (
    NiftiGeometryValidator,
)
from cardiac_segmentation.data.nifti_mask_label_validator import (
    NiftiMaskLabelValidator,
)
from cardiac_segmentation.data.nifti_mask_statistics_reader import (
    NiftiMaskStatisticsReader,
)
from cardiac_segmentation.data.nifti_metadata_reader import (
    NiftiMetadataReader,
)
from cardiac_segmentation.data.nifti_volume_metadata import (
    NiftiVolumeMetadata,
)


class AcdcDatasetInspector:
    """Inspect every real ACDC ED and ES image-mask pair."""

    def __init__(self, config: AppConfig) -> None:
        """Initialize inspection components from validated configuration."""
        self._config = config
        self._indexer = AcdcDatasetIndexer(
            dataset_root=config.dataset.root_dir,
            info_parser=AcdcInfoParser(),
        )
        self._metadata_reader = NiftiMetadataReader()
        self._geometry_validator = NiftiGeometryValidator(
            absolute_tolerance=(
                config.validation.affine_absolute_tolerance
            )
        )
        self._mask_statistics_reader = (
            NiftiMaskStatisticsReader()
        )
        self._mask_label_validator = NiftiMaskLabelValidator(
            expected_labels=config.validation.expected_labels
        )

    def inspect(self) -> AcdcDatasetInspectionReport:
        """Inspect all indexed patients and return one dataset report."""
        patient_records = tuple(
            self._inspect_patient(patient_case)
            for patient_case in self._indexer.index()
        )

        return AcdcDatasetInspectionReport(
            dataset_name=self._config.dataset.name,
            expected_labels=self._config.validation.expected_labels,
            patient_records=patient_records,
        )

    def _inspect_patient(
        self,
        patient_case: AcdcPatientCase,
    ) -> AcdcPatientInspectionRecord:
        """Inspect the ED and ES image-mask pairs for one patient."""
        phase_records = (
            self._inspect_phase(
                phase_name="ED",
                image_path=patient_case.ed_image_path,
                mask_path=patient_case.ed_mask_path,
            ),
            self._inspect_phase(
                phase_name="ES",
                image_path=patient_case.es_image_path,
                mask_path=patient_case.es_mask_path,
            ),
        )

        return AcdcPatientInspectionRecord(
            patient_id=patient_case.patient_id,
            split_name=patient_case.split_name,
            phase_records=phase_records,
        )

    def _inspect_phase(
        self,
        phase_name: str,
        image_path: Path,
        mask_path: Path,
    ) -> AcdcPhaseInspectionRecord:
        """Inspect one image-mask pair and return its validated record."""
        image_metadata = self._metadata_reader.read(image_path)
        mask_metadata = self._metadata_reader.read(mask_path)

        self._geometry_validator.validate_pair(
            image_metadata=image_metadata,
            mask_metadata=mask_metadata,
        )

        self._validate_configured_metadata_rules(
            image_metadata=image_metadata,
            mask_metadata=mask_metadata,
        )

        mask_statistics = self._mask_statistics_reader.read(
            mask_path
        )
        self._mask_label_validator.validate(mask_statistics)

        return AcdcPhaseInspectionRecord(
            phase_name=phase_name,
            image_metadata=image_metadata,
            mask_metadata=mask_metadata,
            mask_statistics=mask_statistics,
        )

    def _validate_configured_metadata_rules(
        self,
        image_metadata: NiftiVolumeMetadata,
        mask_metadata: NiftiVolumeMetadata,
    ) -> None:
        """Apply configurable finite-value and spacing requirements."""
        metadata_items = (
            image_metadata,
            mask_metadata,
        )

        for metadata in metadata_items:
            if (
                self._config.validation.require_finite_intensities
                and not metadata.has_only_finite_values
            ):
                raise ValueError(
                    "NIfTI volume contains non-finite values: "
                    f"{metadata.file_path}"
                )

            if (
                self._config.validation.require_positive_voxel_spacing
                and any(
                    spacing <= 0.0
                    for spacing in metadata.voxel_spacing
                )
            ):
                raise ValueError(
                    "NIfTI volume contains non-positive voxel spacing: "
                    f"{metadata.file_path}"
                )
