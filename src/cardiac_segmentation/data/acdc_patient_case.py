from dataclasses import dataclass
from pathlib import Path

from cardiac_segmentation.data.acdc_patient_info import AcdcPatientInfo


@dataclass(frozen=True, slots=True)
class AcdcPatientCase:
    """Store validated file paths and metadata for one ACDC patient."""

    split_name: str
    patient_dir: Path
    patient_info: AcdcPatientInfo
    cine_path: Path
    ed_image_path: Path
    ed_mask_path: Path
    es_image_path: Path
    es_mask_path: Path

    def __post_init__(self) -> None:
        """Validate the dataset split and all required patient file paths."""
        if self.split_name not in {"training", "testing"}:
            raise ValueError(
                "Dataset split must be either 'training' or 'testing'."
            )

        if not self.patient_dir.is_dir():
            raise NotADirectoryError(
                f"ACDC patient directory does not exist: {self.patient_dir}"
            )

        required_paths = (
            self.cine_path,
            self.ed_image_path,
            self.ed_mask_path,
            self.es_image_path,
            self.es_mask_path,
        )

        for required_path in required_paths:
            if not required_path.is_file():
                raise FileNotFoundError(
                    f"Required ACDC patient file does not exist: {required_path}"
                )

    @property
    def patient_id(self) -> str:
        """Return the patient identifier stored in the parsed metadata."""
        return self.patient_info.patient_id
