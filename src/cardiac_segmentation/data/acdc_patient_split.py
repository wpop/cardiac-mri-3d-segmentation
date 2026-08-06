from dataclasses import dataclass

from cardiac_segmentation.data.acdc_patient_case import AcdcPatientCase


@dataclass(frozen=True, slots=True)
class AcdcPatientSplit:
    """Store deterministic patient-level training and validation partitions."""

    training_cases: tuple[AcdcPatientCase, ...]
    validation_cases: tuple[AcdcPatientCase, ...]

    def __post_init__(self) -> None:
        """Validate patient split membership and disjointness."""
        if not self.training_cases:
            raise ValueError("Training patient cases must not be empty.")

        if not self.validation_cases:
            raise ValueError("Validation patient cases must not be empty.")

        self._validate_training_split(self.training_cases, context="Training")
        self._validate_training_split(self.validation_cases, context="Validation")

        training_patient_ids = self._patient_ids(self.training_cases)
        validation_patient_ids = self._patient_ids(self.validation_cases)
        overlapping_patient_ids = training_patient_ids & validation_patient_ids

        if overlapping_patient_ids:
            raise ValueError(
                "Training and validation patient identifiers must be disjoint: "
                f"{tuple(sorted(overlapping_patient_ids))}."
            )

        training_object_ids = {id(patient_case) for patient_case in self.training_cases}
        validation_object_ids = {id(patient_case) for patient_case in self.validation_cases}

        if training_object_ids & validation_object_ids:
            raise ValueError(
                "The same ACDC patient case object must not appear in both splits."
            )

    def _validate_training_split(
        self,
        patient_cases: tuple[AcdcPatientCase, ...],
        *,
        context: str,
    ) -> None:
        """Validate one side of the patient split."""
        if any(patient_case.split_name != "training" for patient_case in patient_cases):
            raise ValueError(f"{context} patient cases must all come from the training split.")

        patient_ids = [patient_case.patient_id for patient_case in patient_cases]

        if len(patient_ids) != len(set(patient_ids)):
            raise ValueError(
                f"{context} patient identifiers must be unique."
            )

    @staticmethod
    def _patient_ids(
        patient_cases: tuple[AcdcPatientCase, ...],
    ) -> set[str]:
        """Return patient identifiers from one split as a set."""
        return {patient_case.patient_id for patient_case in patient_cases}
