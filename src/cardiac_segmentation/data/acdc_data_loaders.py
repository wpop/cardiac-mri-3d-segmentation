from collections.abc import Sized
from dataclasses import dataclass

from torch import Tensor
from torch.utils.data import DataLoader

from cardiac_segmentation.data.acdc_patient_split import AcdcPatientSplit


@dataclass(frozen=True, slots=True)
class AcdcDataLoaders:
    """Store patient split metadata with training and validation DataLoaders."""

    patient_split: AcdcPatientSplit
    training_loader: DataLoader[dict[str, Tensor | str]]
    validation_loader: DataLoader[dict[str, Tensor | str]]

    def __post_init__(self) -> None:
        """Validate DataLoader lengths and batch sizes."""
        self._validate_loader(
            loader=self.training_loader,
            expected_length=2 * len(self.patient_split.training_cases),
            context="Training",
        )
        self._validate_loader(
            loader=self.validation_loader,
            expected_length=2 * len(self.patient_split.validation_cases),
            context="Validation",
        )

    @staticmethod
    def _validate_loader(
        loader: DataLoader[dict[str, Tensor | str]],
        expected_length: int,
        *,
        context: str,
    ) -> None:
        """Validate one DataLoader's dataset length and batch size."""
        dataset = loader.dataset

        if not isinstance(dataset, Sized):
            raise TypeError(f"{context} DataLoader dataset must be sized.")

        if len(dataset) != expected_length:
            raise ValueError(
                f"{context} DataLoader dataset length must be {expected_length}, "
                f"but received {len(dataset)}."
            )

        if loader.batch_size is None or loader.batch_size <= 0:
            raise ValueError(f"{context} DataLoader must use a positive batch size.")
