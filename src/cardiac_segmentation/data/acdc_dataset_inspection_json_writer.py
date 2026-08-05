import json
from pathlib import Path

from cardiac_segmentation.data.acdc_dataset_inspection_report import (
    AcdcDatasetInspectionReport,
)
from cardiac_segmentation.data.acdc_patient_inspection_record import (
    AcdcPatientInspectionRecord,
)
from cardiac_segmentation.data.acdc_phase_inspection_record import (
    AcdcPhaseInspectionRecord,
)
from cardiac_segmentation.data.nifti_volume_metadata import (
    NiftiVolumeMetadata,
)


class AcdcDatasetInspectionJsonWriter:
    """Write a complete ACDC dataset inspection report as JSON."""

    def write(
        self,
        report: AcdcDatasetInspectionReport,
        output_path: Path,
    ) -> Path:
        """Serialize the report and return the resolved output path."""
        resolved_output_path = output_path.expanduser().resolve(strict=False)
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

        serialized_report = json.dumps(
            self._build_report_payload(report),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )

        try:
            resolved_output_path.write_text(
                f"{serialized_report}\n",
                encoding="utf-8",
            )
        except OSError as error:
            raise OSError(
                f"Failed to write JSON inspection report: "
                f"{resolved_output_path}"
            ) from error

        return resolved_output_path

    def _build_report_payload(
        self,
        report: AcdcDatasetInspectionReport,
    ) -> dict[str, object]:
        """Build the JSON-compatible dataset-level payload."""
        return {
            "dataset_name": report.dataset_name,
            "expected_labels": list(report.expected_labels),
            "observed_labels": list(report.observed_labels),
            "patient_count": report.patient_count,
            "phase_count": report.phase_count,
            "split_patient_counts": {
                "training": report.patient_count_for_split("training"),
                "testing": report.patient_count_for_split("testing"),
            },
            "patients": [
                self._build_patient_payload(patient_record)
                for patient_record in report.patient_records
            ],
        }

    def _build_patient_payload(
        self,
        patient_record: AcdcPatientInspectionRecord,
    ) -> dict[str, object]:
        """Build the JSON-compatible payload for one patient."""
        return {
            "patient_id": patient_record.patient_id,
            "split_name": patient_record.split_name,
            "phases": [
                self._build_phase_payload(phase_record)
                for phase_record in patient_record.phase_records
            ],
        }

    def _build_phase_payload(
        self,
        phase_record: AcdcPhaseInspectionRecord,
    ) -> dict[str, object]:
        """Build the JSON-compatible payload for one cardiac phase."""
        return {
            "phase_name": phase_record.phase_name,
            "image_metadata": self._build_metadata_payload(
                phase_record.image_metadata
            ),
            "mask_metadata": self._build_metadata_payload(
                phase_record.mask_metadata
            ),
            "mask_statistics": {
                "total_voxel_count": (
                    phase_record.mask_statistics.total_voxel_count
                ),
                "label_voxel_counts": [
                    {
                        "label": label,
                        "voxel_count": voxel_count,
                    }
                    for label, voxel_count
                    in phase_record.mask_statistics.label_voxel_counts
                ],
            },
        }

    def _build_metadata_payload(
        self,
        metadata: NiftiVolumeMetadata,
    ) -> dict[str, object]:
        """Build the JSON-compatible payload for one NIfTI volume."""
        return {
            "file_path": str(metadata.file_path),
            "shape": list(metadata.shape),
            "voxel_spacing": list(metadata.voxel_spacing),
            "orientation": list(metadata.orientation),
            "affine": [
                list(row)
                for row in metadata.affine
            ],
            "data_type": metadata.data_type,
            "intensity_min": metadata.intensity_min,
            "intensity_max": metadata.intensity_max,
            "has_only_finite_values": metadata.has_only_finite_values,
        }
