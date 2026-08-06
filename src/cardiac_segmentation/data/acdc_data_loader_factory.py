import torch
from torch.utils.data import DataLoader

from cardiac_segmentation.config.preprocessing_config import PreprocessingConfig
from cardiac_segmentation.config.validation_config import ValidationConfig
from cardiac_segmentation.data.acdc_data_loaders import AcdcDataLoaders
from cardiac_segmentation.data.acdc_patient_case import AcdcPatientCase
from cardiac_segmentation.data.acdc_patient_splitter import AcdcPatientSplitter
from cardiac_segmentation.data.acdc_segmentation_dataset import AcdcSegmentationDataset


class AcdcDataLoaderFactory:
    """Create deterministic ACDC training and validation DataLoaders."""

    def __init__(  # noqa: PLR0913
        self,
        preprocessing_config: PreprocessingConfig,
        validation_config: ValidationConfig,
        validation_fraction: float,
        random_seed: int,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize split, Dataset, and DataLoader configuration."""
        self._validate_batch_size(batch_size)
        self._validate_num_workers(num_workers)
        self._validate_pin_memory(pin_memory)

        self._preprocessing_config = preprocessing_config
        self._validation_config = validation_config
        self._splitter = AcdcPatientSplitter(
            validation_fraction=validation_fraction,
            random_seed=random_seed,
        )
        self._random_seed = random_seed
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._pin_memory = pin_memory

    def create(
        self,
        patient_cases: tuple[AcdcPatientCase, ...],
    ) -> AcdcDataLoaders:
        """Create deterministic patient split and matching DataLoaders."""
        patient_split = self._splitter.split(patient_cases)
        training_dataset = AcdcSegmentationDataset(
            patient_cases=patient_split.training_cases,
            preprocessing_config=self._preprocessing_config,
            validation_config=self._validation_config,
        )
        validation_dataset = AcdcSegmentationDataset(
            patient_cases=patient_split.validation_cases,
            preprocessing_config=self._preprocessing_config,
            validation_config=self._validation_config,
        )
        training_generator = torch.Generator()
        training_generator.manual_seed(self._random_seed)

        return AcdcDataLoaders(
            patient_split=patient_split,
            training_loader=DataLoader(
                training_dataset,
                batch_size=self._batch_size,
                shuffle=True,
                num_workers=self._num_workers,
                pin_memory=self._pin_memory,
                drop_last=False,
                generator=training_generator,
                persistent_workers=self._num_workers > 0,
            ),
            validation_loader=DataLoader(
                validation_dataset,
                batch_size=self._batch_size,
                shuffle=False,
                num_workers=self._num_workers,
                pin_memory=self._pin_memory,
                drop_last=False,
                persistent_workers=self._num_workers > 0,
            ),
        )

    @staticmethod
    def _validate_batch_size(
        batch_size: int,
    ) -> None:
        """Validate the configured DataLoader batch size."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("Batch size must be a positive integer.")

    @staticmethod
    def _validate_num_workers(
        num_workers: int,
    ) -> None:
        """Validate the configured DataLoader worker count."""
        if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 0:
            raise ValueError("Number of workers must be a non-negative integer.")

    @staticmethod
    def _validate_pin_memory(
        pin_memory: bool,
    ) -> None:
        """Validate the configured pinned-memory option."""
        if not isinstance(pin_memory, bool):
            raise TypeError("pin_memory must be a boolean.")
