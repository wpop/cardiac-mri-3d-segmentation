import random
from math import isfinite

from cardiac_segmentation.data.acdc_patient_case import AcdcPatientCase
from cardiac_segmentation.data.acdc_patient_split import AcdcPatientSplit


class AcdcPatientSplitter:
    """Create deterministic patient-level training and validation splits."""

    def __init__(
        self,
        validation_fraction: float,
        random_seed: int,
    ) -> None:
        """Initialize split fraction and deterministic random seed."""
        self._validate_validation_fraction(validation_fraction)
        self._validate_random_seed(random_seed)

        self._validation_fraction = validation_fraction
        self._random_seed = random_seed

    def split(
        self,
        patient_cases: tuple[AcdcPatientCase, ...],
    ) -> AcdcPatientSplit:
        """Split real ACDC training cases into patient-level partitions."""
        self._validate_patient_cases(patient_cases)
        validation_count = max(
            1,
            round(len(patient_cases) * self._validation_fraction),
        )

        if validation_count >= len(patient_cases):
            raise ValueError(
                "Validation split would leave no training patients."
            )

        random_generator = random.Random(self._random_seed)
        validation_indices = frozenset(
            random_generator.sample(
                range(len(patient_cases)),
                validation_count,
            )
        )
        training_cases = tuple(
            patient_case
            for index, patient_case in enumerate(patient_cases)
            if index not in validation_indices
        )
        validation_cases = tuple(
            patient_case
            for index, patient_case in enumerate(patient_cases)
            if index in validation_indices
        )

        return AcdcPatientSplit(
            training_cases=training_cases,
            validation_cases=validation_cases,
        )

    @staticmethod
    def _validate_validation_fraction(
        validation_fraction: float,
    ) -> None:
        """Validate the requested validation partition fraction."""
        if not isfinite(validation_fraction):
            raise ValueError("Validation fraction must be finite.")

        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("Validation fraction must satisfy 0.0 < fraction < 1.0.")

    @staticmethod
    def _validate_random_seed(
        random_seed: int,
    ) -> None:
        """Validate the deterministic random seed."""
        if isinstance(random_seed, bool) or not isinstance(random_seed, int):
            raise TypeError("Random seed must be an integer.")

    @staticmethod
    def _validate_patient_cases(
        patient_cases: tuple[AcdcPatientCase, ...],
    ) -> None:
        """Validate source patient cases before splitting."""
        if not patient_cases:
            raise ValueError("Patient cases must not be empty.")

        if any(patient_case.split_name != "training" for patient_case in patient_cases):
            raise ValueError("Only ACDC training patient cases may be split.")

        patient_ids = [patient_case.patient_id for patient_case in patient_cases]

        if len(patient_ids) != len(set(patient_ids)):
            raise ValueError("Patient identifiers must be unique before splitting.")
