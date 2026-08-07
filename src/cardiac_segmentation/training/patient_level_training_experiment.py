from __future__ import annotations

import random
from collections.abc import Sized

import torch

from cardiac_segmentation.config import AppConfig, PatientLevelTrainingConfig
from cardiac_segmentation.data import (
    AcdcDataLoaderFactory,
    AcdcDataLoaders,
    AcdcDatasetIndexer,
    AcdcInfoParser,
    AcdcPatientCase,
)
from cardiac_segmentation.losses import CrossEntropyDiceLoss3D
from cardiac_segmentation.models import CompactUNet3D
from cardiac_segmentation.training.segmentation_epoch_runner import (
    SegmentationEpochRunner,
)
from cardiac_segmentation.training.segmentation_trainer import SegmentationTrainer
from cardiac_segmentation.training.segmentation_training_history import (
    SegmentationTrainingHistory,
)


class PatientLevelTrainingExperiment:
    """Run deterministic patient-level training on real ACDC training cases."""

    def __init__(
        self,
        app_config: AppConfig,
        training_config: PatientLevelTrainingConfig,
    ) -> None:
        """Initialize device, selected cases, split summary, and volume counts."""
        self._app_config = app_config
        self._training_config = training_config
        self._device = self._resolve_device()
        self._selected_cases = self._select_training_cases()
        data_loaders = self._create_data_loaders()
        self._training_patient_ids = tuple(
            patient_case.patient_id
            for patient_case in data_loaders.patient_split.training_cases
        )
        self._validation_patient_ids = tuple(
            patient_case.patient_id
            for patient_case in data_loaders.patient_split.validation_cases
        )
        self._validate_disjoint_patient_ids()
        self._training_volume_count = self._dataset_length(
            data_loaders.training_loader.dataset,
            context="Training",
        )
        self._validation_volume_count = self._dataset_length(
            data_loaders.validation_loader.dataset,
            context="Validation",
        )

    @property
    def device(self) -> torch.device:
        """Return the resolved execution device."""
        return self._device

    @property
    def training_patient_ids(self) -> tuple[str, ...]:
        """Return the deterministic training patient identifiers."""
        return self._training_patient_ids

    @property
    def validation_patient_ids(self) -> tuple[str, ...]:
        """Return the deterministic validation patient identifiers."""
        return self._validation_patient_ids

    @property
    def training_volume_count(self) -> int:
        """Return the number of training ED/ES volumes."""
        return self._training_volume_count

    @property
    def validation_volume_count(self) -> int:
        """Return the number of validation ED/ES volumes."""
        return self._validation_volume_count

    def run(self) -> SegmentationTrainingHistory:
        """Train and validate on disjoint real ACDC training patients."""
        self._seed_random_generators()
        data_loaders = self._create_data_loaders()
        self._validate_fresh_data_loaders(data_loaders)
        class_count = len(self._app_config.validation.expected_labels)
        model = CompactUNet3D(
            in_channels=1,
            num_classes=class_count,
            base_channels=self._training_config.base_channels,
        ).to(self._device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self._training_config.learning_rate,
            weight_decay=self._training_config.weight_decay,
        )
        epoch_runner = SegmentationEpochRunner(
            model=model,
            loss_function=CrossEntropyDiceLoss3D(
                num_classes=class_count,
                cross_entropy_weight=0.5,
                dice_weight=0.5,
                include_background_in_dice=False,
            ),
            optimizer=optimizer,
            num_classes=class_count,
            device=self._device,
            include_background_in_dice=False,
        )
        trainer = SegmentationTrainer(
            model=model,
            optimizer=optimizer,
            epoch_runner=epoch_runner,
        )

        return trainer.fit(
            training_loader=data_loaders.training_loader,
            validation_loader=data_loaders.validation_loader,
            epoch_count=self._training_config.epoch_count,
            checkpoint_path=self._training_config.checkpoint_path,
        )

    def _resolve_device(self) -> torch.device:
        """Resolve the configured execution device."""
        if self._training_config.device == "cpu":
            return torch.device("cpu")

        if self._training_config.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is not available.")

            return torch.device("cuda", torch.cuda.current_device())

        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())

        return torch.device("cpu")

    def _select_training_cases(self) -> tuple[AcdcPatientCase, ...]:
        """Select the first configured count of real ACDC training cases."""
        patient_cases = AcdcDatasetIndexer(
            dataset_root=self._app_config.dataset.root_dir,
            info_parser=AcdcInfoParser(),
        ).index()
        training_cases = tuple(
            patient_case
            for patient_case in patient_cases
            if patient_case.split_name == "training"
        )
        selected_cases = training_cases[: self._training_config.patient_count]

        if len(selected_cases) != self._training_config.patient_count:
            raise ValueError(
                "ACDC training patient selection produced "
                f"{len(selected_cases)} cases, but "
                f"{self._training_config.patient_count} were required."
            )

        return selected_cases

    def _create_data_loaders(self) -> AcdcDataLoaders:
        """Create fresh deterministic patient-level DataLoaders."""
        return AcdcDataLoaderFactory(
            preprocessing_config=self._app_config.preprocessing,
            validation_config=self._app_config.validation,
            validation_fraction=self._training_config.validation_fraction,
            random_seed=self._training_config.random_seed,
            batch_size=self._training_config.batch_size,
            num_workers=self._training_config.num_workers,
            pin_memory=self._training_config.pin_memory,
        ).create(self._selected_cases)

    def _validate_disjoint_patient_ids(self) -> None:
        """Verify that training and validation patient identifiers do not overlap."""
        overlapping_patient_ids = set(self._training_patient_ids) & set(
            self._validation_patient_ids
        )

        if overlapping_patient_ids:
            raise ValueError(
                "Training and validation patient identifiers must be disjoint: "
                f"{tuple(sorted(overlapping_patient_ids))}."
            )

    def _validate_fresh_data_loaders(
        self,
        data_loaders: AcdcDataLoaders,
    ) -> None:
        """Verify fresh DataLoaders match the construction-time split summary."""
        training_patient_ids = tuple(
            patient_case.patient_id
            for patient_case in data_loaders.patient_split.training_cases
        )
        validation_patient_ids = tuple(
            patient_case.patient_id
            for patient_case in data_loaders.patient_split.validation_cases
        )

        if training_patient_ids != self._training_patient_ids:
            raise ValueError("Fresh training patient identifiers changed before run.")

        if validation_patient_ids != self._validation_patient_ids:
            raise ValueError("Fresh validation patient identifiers changed before run.")

        self._validate_disjoint_patient_ids()

        training_volume_count = self._dataset_length(
            data_loaders.training_loader.dataset,
            context="Training",
        )
        validation_volume_count = self._dataset_length(
            data_loaders.validation_loader.dataset,
            context="Validation",
        )

        if training_volume_count != self._training_volume_count:
            raise ValueError(
                "Fresh training volume count changed from "
                f"{self._training_volume_count} to {training_volume_count}."
            )

        if validation_volume_count != self._validation_volume_count:
            raise ValueError(
                "Fresh validation volume count changed from "
                f"{self._validation_volume_count} to {validation_volume_count}."
            )

    @staticmethod
    def _dataset_length(
        dataset: object,
        *,
        context: str,
    ) -> int:
        """Return the length of a sized Dataset object."""
        if not isinstance(dataset, Sized):
            raise TypeError(f"{context} Dataset must be sized.")

        return len(dataset)

    def _seed_random_generators(self) -> None:
        """Seed Python and PyTorch random number generators."""
        random.seed(self._training_config.random_seed)
        torch.manual_seed(self._training_config.random_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self._training_config.random_seed)
