from __future__ import annotations

import csv
import json
import random
import statistics
from collections.abc import Iterable, Mapping, Sized
from math import isinf
from pathlib import Path
from typing import Any, Final, cast

import matplotlib.pyplot as plt
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap
from torch import Tensor

from cardiac_segmentation.config import AppConfig, ValidationInferenceConfig
from cardiac_segmentation.data import (
    AcdcDataLoaderFactory,
    AcdcDataLoaders,
    AcdcDatasetIndexer,
    AcdcInfoParser,
    AcdcPatientCase,
)
from cardiac_segmentation.evaluation.validation_inference_case_result import (
    ValidationInferenceCaseResult,
)
from cardiac_segmentation.evaluation.validation_inference_report import (
    ValidationInferenceReport,
)
from cardiac_segmentation.metrics import BinaryHD95Metric3D, MulticlassDiceMetric3D
from cardiac_segmentation.models import CompactUNet3D
from cardiac_segmentation.training import SegmentationTrainingCheckpointLoader

plt.switch_backend("Agg")

_CSV_FIELDNAMES: Final[tuple[str, ...]] = (
    "patient_id",
    "volume_id",
    "rv_dice",
    "myocardium_dice",
    "lv_dice",
    "mean_foreground_dice",
    "rv_hd95_mm",
    "myocardium_hd95_mm",
    "lv_hd95_mm",
    "mean_foreground_hd95_mm",
)
_FOREGROUND_LABELS: Final[tuple[int, int, int]] = (1, 2, 3)
_PREPROCESSING_TARGET_SPACING_XYZ_MM: Final[tuple[float, float, float]] = (1.5, 1.5, 5.0)
_HD95_SPACING_DHW_MM: Final[tuple[float, float, float]] = (
    _PREPROCESSING_TARGET_SPACING_XYZ_MM[2],
    _PREPROCESSING_TARGET_SPACING_XYZ_MM[1],
    _PREPROCESSING_TARGET_SPACING_XYZ_MM[0],
)


class ValidationInferenceExperiment:
    """Run deterministic validation inference on real ACDC validation volumes."""

    def __init__(
        self,
        app_config: AppConfig,
        inference_config: ValidationInferenceConfig,
    ) -> None:
        """Initialize device, selected cases, split summary, and volume counts."""
        self._app_config = app_config
        self._inference_config = inference_config
        self._device = self._resolve_device()
        self._selected_cases = self._select_training_cases()
        data_loaders = self._create_data_loaders()
        self._validation_patient_ids = tuple(
            patient_case.patient_id
            for patient_case in data_loaders.patient_split.validation_cases
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
    def validation_patient_ids(self) -> tuple[str, ...]:
        """Return the deterministic validation patient identifiers."""
        return self._validation_patient_ids

    @property
    def validation_volume_count(self) -> int:
        """Return the number of validation ED/ES volumes."""
        return self._validation_volume_count

    def run(self) -> ValidationInferenceReport:
        """Run checkpoint inference, write report artifacts, and return metadata."""
        self._seed_random_generators()
        data_loaders = self._create_data_loaders()
        self._validate_fresh_data_loaders(data_loaders)
        class_count = len(self._app_config.validation.expected_labels)
        model = CompactUNet3D(
            in_channels=1,
            num_classes=class_count,
            base_channels=self._inference_config.base_channels,
        ).to(self._device)
        optimizer = torch.optim.AdamW(model.parameters())
        checkpoint = SegmentationTrainingCheckpointLoader().load_into(
            checkpoint_path=self._inference_config.checkpoint_path,
            model=model,
            optimizer=optimizer,
            device=self._device,
        )
        model.eval()

        case_payloads = self._run_validation_inference(
            model=model,
            validation_loader=data_loaders.validation_loader,
        )
        case_results = tuple(payload["case_result"] for payload in case_payloads)
        csv_path = self._write_csv(case_results)
        summary_path = self._write_summary_json(
            checkpoint_epoch_number=checkpoint.epoch_number,
            validation_patient_count=len(self._validation_patient_ids),
            case_results=case_results,
        )
        visualization_paths = self._write_visualizations(case_payloads)

        return ValidationInferenceReport(
            checkpoint_epoch_number=checkpoint.epoch_number,
            case_results=case_results,
            report_csv_path=csv_path,
            summary_json_path=summary_path,
            visualization_paths=visualization_paths,
        )

    def _run_validation_inference(
        self,
        *,
        model: torch.nn.Module,
        validation_loader: Iterable[Mapping[str, object]],
    ) -> tuple[dict[str, Any], ...]:
        """Run per-volume inference and retain tensors needed for visualization."""
        case_payloads: list[dict[str, Any]] = []

        with torch.inference_mode():
            for batch in validation_loader:
                images = cast(Tensor, batch["image"]).to(self._device)
                masks = cast(Tensor, batch["mask"]).to(self._device)
                logits = model(images)
                predictions = torch.argmax(logits, dim=1)

                for batch_index in range(int(images.shape[0])):
                    patient_id = self._extract_metadata(batch["patient_id"], batch_index)
                    phase_name = self._extract_metadata(batch["phase_name"], batch_index)
                    volume_id = f"{patient_id}_{phase_name}"
                    case_result = self._calculate_case_result(
                        logits=logits[batch_index : batch_index + 1],
                        mask=masks[batch_index : batch_index + 1],
                        prediction=predictions[batch_index],
                        patient_id=patient_id,
                        volume_id=volume_id,
                    )
                    case_payloads.append(
                        {
                            "case_result": case_result,
                            "image": images[batch_index, 0].detach().cpu(),
                            "mask": masks[batch_index].detach().cpu(),
                            "prediction": predictions[batch_index].detach().cpu(),
                        }
                    )

        if not case_payloads:
            raise RuntimeError("Validation inference produced no case results.")

        return tuple(case_payloads)

    def _calculate_case_result(
        self,
        *,
        logits: Tensor,
        mask: Tensor,
        prediction: Tensor,
        patient_id: str,
        volume_id: str,
    ) -> ValidationInferenceCaseResult:
        """Calculate foreground Dice and HD95 values for one validation volume."""
        metric = MulticlassDiceMetric3D(
            num_classes=len(self._app_config.validation.expected_labels),
            include_background=False,
        )
        metric.update(logits, mask)
        dice_result = metric.compute()
        rv_dice, myocardium_dice, lv_dice = dice_result.per_class_dice
        hd95_result = self._calculate_foreground_hd95(
            prediction=prediction,
            mask=mask[0],
        )
        rv_hd95_mm, myocardium_hd95_mm, lv_hd95_mm = hd95_result

        return ValidationInferenceCaseResult(
            patient_id=patient_id,
            volume_id=volume_id,
            rv_dice=rv_dice,
            myocardium_dice=myocardium_dice,
            lv_dice=lv_dice,
            mean_foreground_dice=dice_result.mean_dice,
            rv_hd95_mm=rv_hd95_mm,
            myocardium_hd95_mm=myocardium_hd95_mm,
            lv_hd95_mm=lv_hd95_mm,
            mean_foreground_hd95_mm=self._calculate_mean_foreground_hd95(hd95_result),
        )

    @staticmethod
    def _calculate_foreground_hd95(
        *,
        prediction: Tensor,
        mask: Tensor,
    ) -> tuple[float, float, float]:
        """Calculate per-class HD95 using tensor [D,H,W] spacing in millimeters."""
        hd95_metric = BinaryHD95Metric3D(spacing_mm_dhw=_HD95_SPACING_DHW_MM)
        prediction_array = prediction.detach().cpu().numpy()
        mask_array = mask.detach().cpu().numpy()
        hd95_values = [
            hd95_metric.compute(
                ground_truth_mask=mask_array == label,
                prediction_mask=prediction_array == label,
            )
            for label in _FOREGROUND_LABELS
        ]

        return (
            hd95_values[0],
            hd95_values[1],
            hd95_values[2],
        )

    @staticmethod
    def _calculate_mean_foreground_hd95(
        hd95_values_mm: tuple[float, float, float],
    ) -> float:
        """Return mean foreground HD95, preserving infinity if any class is infinite."""
        if any(isinf(value) for value in hd95_values_mm):
            return float("inf")

        return statistics.fmean(hd95_values_mm)

    def _write_csv(
        self,
        case_results: tuple[ValidationInferenceCaseResult, ...],
    ) -> Path:
        """Write one CSV row per validation volume."""
        csv_path = self._inference_config.csv_report_path
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=_CSV_FIELDNAMES)
            writer.writeheader()

            for result in case_results:
                writer.writerow(
                    {
                        "patient_id": result.patient_id,
                        "volume_id": result.volume_id,
                        "rv_dice": f"{result.rv_dice:.12f}",
                        "myocardium_dice": f"{result.myocardium_dice:.12f}",
                        "lv_dice": f"{result.lv_dice:.12f}",
                        "mean_foreground_dice": f"{result.mean_foreground_dice:.12f}",
                        "rv_hd95_mm": f"{result.rv_hd95_mm:.12f}",
                        "myocardium_hd95_mm": f"{result.myocardium_hd95_mm:.12f}",
                        "lv_hd95_mm": f"{result.lv_hd95_mm:.12f}",
                        "mean_foreground_hd95_mm": (
                            f"{result.mean_foreground_hd95_mm:.12f}"
                        ),
                    }
                )

        return csv_path

    def _write_summary_json(
        self,
        *,
        checkpoint_epoch_number: int,
        validation_patient_count: int,
        case_results: tuple[ValidationInferenceCaseResult, ...],
    ) -> Path:
        """Write aggregate validation inference statistics as JSON."""
        sorted_results = self._rank_case_results(case_results)
        middle_result = sorted_results[len(sorted_results) // 2]
        summary = {
            "checkpoint_epoch": checkpoint_epoch_number,
            "validation_patient_count": validation_patient_count,
            "validation_volume_count": len(case_results),
            "per_class": {
                "rv": self._aggregate(result.rv_dice for result in case_results),
                "myocardium": self._aggregate(
                    result.myocardium_dice for result in case_results
                ),
                "lv": self._aggregate(result.lv_dice for result in case_results),
            },
            "mean_foreground_dice": self._aggregate(
                result.mean_foreground_dice for result in case_results
            ),
            "per_class_hd95_mm": {
                "rv_hd95_mm": self._aggregate(
                    result.rv_hd95_mm for result in case_results
                ),
                "myocardium_hd95_mm": self._aggregate(
                    result.myocardium_hd95_mm for result in case_results
                ),
                "lv_hd95_mm": self._aggregate(
                    result.lv_hd95_mm for result in case_results
                ),
            },
            "mean_foreground_hd95_mm": self._aggregate(
                result.mean_foreground_hd95_mm for result in case_results
            ),
            "worst_volume_identifier": sorted_results[0].volume_id,
            "middle_volume_identifier": middle_result.volume_id,
            "best_volume_identifier": sorted_results[-1].volume_id,
        }
        json_path = self._inference_config.json_report_path
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        return json_path

    def _write_visualizations(
        self,
        case_payloads: tuple[dict[str, Any], ...],
    ) -> tuple[Path, Path, Path]:
        """Write worst, middle, and best validation case PNG visualizations."""
        output_dir = self._inference_config.visualization_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        ranked_payloads = sorted(
            case_payloads,
            key=lambda payload: (
                cast(
                    ValidationInferenceCaseResult,
                    payload["case_result"],
                ).mean_foreground_dice,
                cast(ValidationInferenceCaseResult, payload["case_result"]).volume_id,
            ),
        )
        selected_payloads = (
            ("worst", ranked_payloads[0]),
            ("middle", ranked_payloads[len(ranked_payloads) // 2]),
            ("best", ranked_payloads[-1]),
        )
        paths: list[Path] = []

        for label, payload in selected_payloads:
            case_result = cast(ValidationInferenceCaseResult, payload["case_result"])
            path = output_dir / f"validation_inference_{label}_{case_result.volume_id}.png"
            self._write_case_visualization(
                path=path,
                case_result=case_result,
                image=cast(Tensor, payload["image"]),
                mask=cast(Tensor, payload["mask"]),
                prediction=cast(Tensor, payload["prediction"]),
            )
            paths.append(path)

        return (
            paths[0],
            paths[1],
            paths[2],
        )

    def _write_case_visualization(
        self,
        *,
        path: Path,
        case_result: ValidationInferenceCaseResult,
        image: Tensor,
        mask: Tensor,
        prediction: Tensor,
    ) -> None:
        """Save a three-panel MRI, ground-truth, and prediction slice figure."""
        slice_index = self._largest_foreground_slice(mask)
        image_slice = image[slice_index].numpy()
        mask_slice = mask[slice_index].numpy()
        prediction_slice = prediction[slice_index].numpy()
        cmap = ListedColormap(("#000000", "#d95f02", "#1b9e77", "#7570b3"))
        norm = BoundaryNorm(boundaries=(-0.5, 0.5, 1.5, 2.5, 3.5), ncolors=cmap.N)
        figure, axes = plt.subplots(1, 3, figsize=(9, 3.2), constrained_layout=True)

        axes[0].imshow(image_slice, cmap="gray")
        axes[0].set_title("MRI")
        axes[1].imshow(mask_slice, cmap=cmap, norm=norm, interpolation="nearest")
        axes[1].set_title("Ground Truth")
        axes[2].imshow(prediction_slice, cmap=cmap, norm=norm, interpolation="nearest")
        axes[2].set_title("Prediction")

        for axis in axes:
            axis.axis("off")

        figure.suptitle(
            f"{case_result.volume_id} | mean foreground Dice "
            f"{case_result.mean_foreground_dice:.6f}"
        )
        figure.savefig(path, dpi=150)
        plt.close(figure)

    def _resolve_device(self) -> torch.device:
        """Resolve the configured execution device."""
        if self._inference_config.device == "cpu":
            return torch.device("cpu")

        if self._inference_config.device == "cuda":
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
        selected_cases = training_cases[: self._inference_config.patient_count]

        if len(selected_cases) != self._inference_config.patient_count:
            raise ValueError(
                "ACDC training patient selection produced "
                f"{len(selected_cases)} cases, but "
                f"{self._inference_config.patient_count} were required."
            )

        return selected_cases

    def _create_data_loaders(self) -> AcdcDataLoaders:
        """Create fresh deterministic patient-level DataLoaders."""
        return AcdcDataLoaderFactory(
            preprocessing_config=self._app_config.preprocessing,
            validation_config=self._app_config.validation,
            validation_fraction=self._inference_config.validation_fraction,
            random_seed=self._inference_config.random_seed,
            batch_size=self._inference_config.batch_size,
            num_workers=self._inference_config.num_workers,
            pin_memory=self._inference_config.pin_memory,
        ).create(self._selected_cases)

    def _validate_fresh_data_loaders(
        self,
        data_loaders: AcdcDataLoaders,
    ) -> None:
        """Verify fresh DataLoaders match the construction-time split summary."""
        validation_patient_ids = tuple(
            patient_case.patient_id
            for patient_case in data_loaders.patient_split.validation_cases
        )

        if validation_patient_ids != self._validation_patient_ids:
            raise ValueError("Fresh validation patient identifiers changed before run.")

        validation_volume_count = self._dataset_length(
            data_loaders.validation_loader.dataset,
            context="Validation",
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
        random.seed(self._inference_config.random_seed)
        torch.manual_seed(self._inference_config.random_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self._inference_config.random_seed)

    @staticmethod
    def _extract_metadata(
        value: object,
        batch_index: int,
    ) -> str:
        """Extract a string metadata value from a collated DataLoader batch."""
        if isinstance(value, str):
            return value

        if isinstance(value, tuple | list):
            item = value[batch_index]

            if not isinstance(item, str):
                raise TypeError("Batch metadata items must be strings.")

            return item

        raise TypeError("Batch metadata must be a string or sequence of strings.")

    @staticmethod
    def _aggregate(
        values: Iterable[float],
    ) -> dict[str, float]:
        """Return mean, median, minimum, and maximum for a metric."""
        value_tuple = tuple(values)

        if not value_tuple:
            raise ValueError("Cannot aggregate an empty metric sequence.")

        return {
            "mean": statistics.fmean(value_tuple),
            "median": statistics.median(value_tuple),
            "minimum": min(value_tuple),
            "maximum": max(value_tuple),
        }

    @staticmethod
    def _rank_case_results(
        case_results: tuple[ValidationInferenceCaseResult, ...],
    ) -> tuple[ValidationInferenceCaseResult, ...]:
        """Sort case results from worst to best with deterministic tie-breaking."""
        return tuple(
            sorted(
                case_results,
                key=lambda result: (
                    result.mean_foreground_dice,
                    result.volume_id,
                ),
            )
        )

    @staticmethod
    def _largest_foreground_slice(
        mask: Tensor,
    ) -> int:
        """Return axial slice index with the largest foreground mask area."""
        foreground_area = torch.isin(
            mask,
            torch.tensor(_FOREGROUND_LABELS, dtype=mask.dtype),
        ).sum(dim=(1, 2))

        return int(torch.argmax(foreground_area).item())
