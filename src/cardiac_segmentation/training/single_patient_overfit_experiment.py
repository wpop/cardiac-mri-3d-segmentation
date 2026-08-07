from __future__ import annotations

import random

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from cardiac_segmentation.config import AppConfig, SinglePatientOverfitConfig
from cardiac_segmentation.data import (
    AcdcDatasetIndexer,
    AcdcInfoParser,
    AcdcPatientCase,
    AcdcSegmentationDataset,
)
from cardiac_segmentation.losses import CrossEntropyDiceLoss3D
from cardiac_segmentation.models import CompactUNet3D
from cardiac_segmentation.training.early_stopping_monitor import (
    EarlyStoppingMonitor,
)
from cardiac_segmentation.training.segmentation_epoch_runner import (
    SegmentationEpochRunner,
)
from cardiac_segmentation.training.segmentation_trainer import SegmentationTrainer
from cardiac_segmentation.training.segmentation_training_history import (
    SegmentationTrainingHistory,
)

_EXPECTED_SINGLE_PATIENT_VOLUME_COUNT = 2


class SinglePatientOverfitExperiment:
    """Run a diagnostic overfit experiment on one real ACDC training patient."""

    def __init__(
        self,
        app_config: AppConfig,
        overfit_config: SinglePatientOverfitConfig,
    ) -> None:
        """Initialize configuration, selected patient, device, and volume count."""
        self._app_config = app_config
        self._overfit_config = overfit_config
        self._device = self._resolve_device()
        self._patient_case = self._select_patient_case()
        self._dataset_volume_count = self._count_patient_volumes()

    @property
    def patient_id(self) -> str:
        """Return the selected ACDC patient identifier."""
        return self._patient_case.patient_id

    @property
    def dataset_volume_count(self) -> int:
        """Return the number of real ED/ES volumes in the selected Dataset."""
        return self._dataset_volume_count

    @property
    def device(self) -> torch.device:
        """Return the resolved execution device."""
        return self._device

    def run(self) -> SegmentationTrainingHistory:
        """Train and validate on the same patient's real ED and ES volumes."""
        self._seed_random_generators()
        dataset = AcdcSegmentationDataset(
            patient_cases=(self._patient_case,),
            preprocessing_config=self._app_config.preprocessing,
            validation_config=self._app_config.validation,
        )
        self._validate_dataset_volume_count(dataset)
        training_loader = self._create_training_loader(dataset)
        validation_loader = self._create_validation_loader(dataset)
        class_count = len(self._app_config.validation.expected_labels)
        model = CompactUNet3D(
            in_channels=1,
            num_classes=class_count,
            base_channels=self._overfit_config.base_channels,
        ).to(self._device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self._overfit_config.learning_rate,
            weight_decay=self._overfit_config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=self._overfit_config.epoch_count + 1,
        )
        early_stopping_monitor = EarlyStoppingMonitor(
            patience=self._overfit_config.epoch_count + 1,
            minimum_improvement=0.001,
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
            scheduler=scheduler,
            early_stopping_monitor=early_stopping_monitor,
        )

        return trainer.fit(
            training_loader=training_loader,
            validation_loader=validation_loader,
            epoch_count=self._overfit_config.epoch_count,
            checkpoint_path=self._overfit_config.checkpoint_path,
        )

    def _resolve_device(self) -> torch.device:
        """Resolve the configured execution device."""
        if self._overfit_config.device == "cpu":
            return torch.device("cpu")

        if self._overfit_config.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is not available.")

            return torch.device("cuda", torch.cuda.current_device())

        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())

        return torch.device("cpu")

    def _select_patient_case(self) -> AcdcPatientCase:
        """Select the configured real ACDC patient from the training split."""
        patient_cases = AcdcDatasetIndexer(
            dataset_root=self._app_config.dataset.root_dir,
            info_parser=AcdcInfoParser(),
        ).index()
        matching_cases = tuple(
            patient_case
            for patient_case in patient_cases
            if patient_case.patient_id == self._overfit_config.patient_id
        )

        if not matching_cases:
            raise ValueError(
                "ACDC patient was not found in the indexed dataset: "
                f"{self._overfit_config.patient_id}"
            )

        patient_case = matching_cases[0]

        if patient_case.split_name != "training":
            raise ValueError(
                "Single-patient overfit requires a training-split patient, but "
                f"{patient_case.patient_id} is in split '{patient_case.split_name}'."
            )

        return patient_case

    def _count_patient_volumes(self) -> int:
        """Create a temporary Dataset and return its exact ED/ES volume count."""
        dataset = AcdcSegmentationDataset(
            patient_cases=(self._patient_case,),
            preprocessing_config=self._app_config.preprocessing,
            validation_config=self._app_config.validation,
        )
        volume_count = len(dataset)

        if volume_count != _EXPECTED_SINGLE_PATIENT_VOLUME_COUNT:
            raise ValueError(
                "Single-patient overfit requires exactly two real volumes "
                f"(ED and ES), but found {volume_count} for {self._patient_case.patient_id}."
            )

        return volume_count

    def _validate_dataset_volume_count(
        self,
        dataset: AcdcSegmentationDataset,
    ) -> None:
        """Verify the fresh run Dataset still matches construction-time size."""
        volume_count = len(dataset)

        if volume_count != self._dataset_volume_count:
            raise ValueError(
                "Single-patient overfit Dataset volume count changed from "
                f"{self._dataset_volume_count} to {volume_count}."
            )

    def _create_training_loader(
        self,
        dataset: AcdcSegmentationDataset,
    ) -> DataLoader[dict[str, Tensor | str]]:
        """Create the seeded shuffled training DataLoader."""
        generator = torch.Generator()
        generator.manual_seed(self._overfit_config.random_seed)

        return DataLoader(
            dataset,
            batch_size=self._overfit_config.batch_size,
            shuffle=True,
            num_workers=self._overfit_config.num_workers,
            pin_memory=self._overfit_config.pin_memory,
            drop_last=False,
            generator=generator,
            persistent_workers=self._overfit_config.num_workers > 0,
        )

    def _create_validation_loader(
        self,
        dataset: AcdcSegmentationDataset,
    ) -> DataLoader[dict[str, Tensor | str]]:
        """Create the unshuffled validation DataLoader over the same Dataset."""
        return DataLoader(
            dataset,
            batch_size=self._overfit_config.batch_size,
            shuffle=False,
            num_workers=self._overfit_config.num_workers,
            pin_memory=self._overfit_config.pin_memory,
            drop_last=False,
            persistent_workers=self._overfit_config.num_workers > 0,
        )

    def _seed_random_generators(self) -> None:
        """Seed Python and PyTorch random number generators."""
        random.seed(self._overfit_config.random_seed)
        torch.manual_seed(self._overfit_config.random_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self._overfit_config.random_seed)
