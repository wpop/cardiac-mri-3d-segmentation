from pathlib import Path

from cardiac_segmentation.data.acdc_info_parser import AcdcInfoParser
from cardiac_segmentation.data.acdc_patient_case import AcdcPatientCase


class AcdcDatasetIndexer:
    """Discover and validate all patient cases in a real ACDC dataset."""

    _SPLIT_NAMES = ("training", "testing")

    def __init__(
        self,
        dataset_root: Path,
        info_parser: AcdcInfoParser,
    ) -> None:
        """Initialize the indexer with an explicit dataset root and parser."""
        resolved_dataset_root = dataset_root.expanduser().resolve(strict=False)

        if not resolved_dataset_root.is_dir():
            raise NotADirectoryError(
                f"ACDC dataset root does not exist: {resolved_dataset_root}"
            )

        self._dataset_root = resolved_dataset_root
        self._info_parser = info_parser

    def index(self) -> tuple[AcdcPatientCase, ...]:
        """Discover and validate every patient from training and testing splits."""
        patient_cases: list[AcdcPatientCase] = []

        for split_name in self._SPLIT_NAMES:
            split_dir = self._dataset_root / split_name

            if not split_dir.is_dir():
                raise NotADirectoryError(
                    f"ACDC dataset split does not exist: {split_dir}"
                )

            patient_dirs = sorted(split_dir.glob("patient[0-9][0-9][0-9]"))

            if not patient_dirs:
                raise ValueError(
                    f"No ACDC patient directories found in split: {split_dir}"
                )

            patient_cases.extend(
                self._build_patient_case(split_name, patient_dir)
                # generator expression: pass by every patient directory in the split
                for patient_dir in patient_dirs
            )

        self._validate_unique_patient_ids(patient_cases)

        return tuple(patient_cases)

    def _build_patient_case(
        self,
        split_name: str,
        patient_dir: Path,
    ) -> AcdcPatientCase:
        """Build one validated patient case from its real dataset directory."""
        info_path = patient_dir / "Info.cfg"
        patient_info = self._info_parser.parse(info_path)
        patient_id = patient_info.patient_id

        ed_frame_name = f"{patient_id}_frame{patient_info.ed_frame:02d}"
        es_frame_name = f"{patient_id}_frame{patient_info.es_frame:02d}"

        return AcdcPatientCase(
            split_name=split_name,
            patient_dir=patient_dir,
            patient_info=patient_info,
            cine_path=patient_dir / f"{patient_id}_4d.nii.gz",
            ed_image_path=patient_dir / f"{ed_frame_name}.nii.gz",
            ed_mask_path=patient_dir / f"{ed_frame_name}_gt.nii.gz",
            es_image_path=patient_dir / f"{es_frame_name}.nii.gz",
            es_mask_path=patient_dir / f"{es_frame_name}_gt.nii.gz",
        )

    def _validate_unique_patient_ids(
        self,
        patient_cases: list[AcdcPatientCase],
    ) -> None:
        """Reject duplicate patient identifiers across dataset splits."""
        patient_ids = [patient_case.patient_id for patient_case in patient_cases]

        if len(patient_ids) != len(set(patient_ids)):
            raise ValueError(
                "Duplicate patient identifiers were found across ACDC splits."
            )
