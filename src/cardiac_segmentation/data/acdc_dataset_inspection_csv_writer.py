from __future__ import annotations

import csv
from pathlib import Path
from typing import Final

from cardiac_segmentation.data.acdc_dataset_inspection_report import (
    AcdcDatasetInspectionReport,
)
from cardiac_segmentation.data.acdc_patient_inspection_record import (
    AcdcPatientInspectionRecord,
)
from cardiac_segmentation.data.acdc_phase_inspection_record import (
    AcdcPhaseInspectionRecord,
)

_BASE_FIELD_NAMES: Final[tuple[str, ...]] = (
    "patient_id",
    "split_name",
    "phase_name",
    "image_path",
    "mask_path",
    "shape_x",
    "shape_y",
    "shape_z",
    "spacing_x",
    "spacing_y",
    "spacing_z",
    "orientation",
    "image_data_type",
    "mask_data_type",
    "image_intensity_min",
    "image_intensity_max",
    "mask_intensity_min",
    "mask_intensity_max",
    "mask_total_voxel_count",
)


class AcdcDatasetInspectionCsvWriter:
    """Write phase-level ACDC inspection results as CSV."""

    def write(
        self,
        report: AcdcDatasetInspectionReport,
        output_path: Path,
    ) -> Path:
        """Write one CSV row for every inspected ED and ES phase."""
        resolved_output_path = output_path.expanduser().resolve(strict=False)
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

        field_names = self._build_field_names(report)

        try:
            with resolved_output_path.open(
                "w",
                encoding="utf-8",
                newline="",
            ) as output_file:
                writer: csv.DictWriter[str] = csv.DictWriter(
                    output_file,
                    fieldnames=list(field_names),
                )
                writer.writeheader()

                for patient_record in report.patient_records:
                    for phase_record in patient_record.phase_records:
                        writer.writerow(
                            self._build_row(
                                report=report,
                                patient_record=patient_record,
                                phase_record=phase_record,
                            )
                        )
        except OSError as error:
            raise OSError(
                f"Failed to write CSV inspection summary: "
                f"{resolved_output_path}"
            ) from error

        return resolved_output_path

    def _build_field_names(
        self,
        report: AcdcDatasetInspectionReport,
    ) -> tuple[str, ...]:
        """Build fixed metadata fields and dynamic label-count fields."""
        label_field_names = tuple(
            f"label_{label}_voxel_count"
            for label in report.expected_labels
        )

        return (*_BASE_FIELD_NAMES, *label_field_names)

    def _build_row(
        self,
        report: AcdcDatasetInspectionReport,
        patient_record: AcdcPatientInspectionRecord,
        phase_record: AcdcPhaseInspectionRecord,
    ) -> dict[str, object]:
        """Build one CSV row for an inspected patient phase."""
        image_metadata = phase_record.image_metadata
        mask_metadata = phase_record.mask_metadata
        mask_statistics = phase_record.mask_statistics

        row: dict[str, object] = {
            "patient_id": patient_record.patient_id,
            "split_name": patient_record.split_name,
            "phase_name": phase_record.phase_name,
            "image_path": str(image_metadata.file_path),
            "mask_path": str(mask_metadata.file_path),
            "shape_x": image_metadata.shape[0],
            "shape_y": image_metadata.shape[1],
            "shape_z": image_metadata.shape[2],
            "spacing_x": image_metadata.voxel_spacing[0],
            "spacing_y": image_metadata.voxel_spacing[1],
            "spacing_z": image_metadata.voxel_spacing[2],
            "orientation": "".join(image_metadata.orientation),
            "image_data_type": image_metadata.data_type,
            "mask_data_type": mask_metadata.data_type,
            "image_intensity_min": image_metadata.intensity_min,
            "image_intensity_max": image_metadata.intensity_max,
            "mask_intensity_min": mask_metadata.intensity_min,
            "mask_intensity_max": mask_metadata.intensity_max,
            "mask_total_voxel_count": mask_statistics.total_voxel_count,
        }

        for label in report.expected_labels:
            row[f"label_{label}_voxel_count"] = (
                mask_statistics.voxel_count_for_label(label)
            )

        return row
