from __future__ import annotations

import json
import random
from collections.abc import Iterable, Mapping, Sized
from pathlib import Path
from typing import Final, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm, ListedColormap
from numpy.typing import NDArray
from torch import Tensor

from cardiac_segmentation.config import AppConfig, ValidationVisualizationConfig
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
from cardiac_segmentation.evaluation.validation_visualization_case_result import (
    ValidationVisualizationCaseResult,
)
from cardiac_segmentation.metrics import BinaryHD95Metric3D, MulticlassDiceMetric3D
from cardiac_segmentation.models import CompactUNet3D
from cardiac_segmentation.training import SegmentationTrainingCheckpointLoader

plt.switch_backend("Agg")

_FOREGROUND_LABELS: Final[tuple[int, int, int]] = (1, 2, 3)
_SPATIAL_DIMENSION_COUNT: Final[int] = 3
_PREPROCESSING_TARGET_SPACING_XYZ_MM: Final[tuple[float, float, float]] = (1.5, 1.5, 5.0)
_HD95_SPACING_DHW_MM: Final[tuple[float, float, float]] = (
    _PREPROCESSING_TARGET_SPACING_XYZ_MM[2],
    _PREPROCESSING_TARGET_SPACING_XYZ_MM[1],
    _PREPROCESSING_TARGET_SPACING_XYZ_MM[0],
)
_MASK_CMAP = ListedColormap(("#000000", "#d62728", "#2ca02c", "#1f77b4"))
_OVERLAY_CMAP = ListedColormap(("#d62728", "#2ca02c", "#1f77b4"))
_MASK_NORM = BoundaryNorm(boundaries=(-0.5, 0.5, 1.5, 2.5, 3.5), ncolors=_MASK_CMAP.N)
_OVERLAY_NORM = BoundaryNorm(boundaries=(0.5, 1.5, 2.5, 3.5), ncolors=_OVERLAY_CMAP.N)


class ValidationVisualizationExperiment:
    """Run validation inference and save ranked multi-slice review figures."""

    def __init__(
        self,
        app_config: AppConfig,
        visualization_config: ValidationVisualizationConfig,
    ) -> None:
        """Initialize device, deterministic split, and output configuration."""
        self._app_config = app_config
        self._visualization_config = visualization_config
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

    def run(self) -> Path:
        """Run validation inference, write review PNG files, and return manifest."""
        self._seed_random_generators()
        data_loaders = self._create_data_loaders()
        self._validate_fresh_data_loaders(data_loaders)
        class_count = len(self._app_config.validation.expected_labels)
        model = CompactUNet3D(
            in_channels=1,
            num_classes=class_count,
            base_channels=self._visualization_config.base_channels,
        ).to(self._device)
        optimizer = torch.optim.AdamW(model.parameters())
        checkpoint = SegmentationTrainingCheckpointLoader().load_into(
            checkpoint_path=self._visualization_config.checkpoint_path,
            model=model,
            optimizer=optimizer,
            device=self._device,
        )
        model.eval()
        case_payloads = self._run_validation_inference(
            model=model,
            validation_loader=data_loaders.validation_loader,
        )
        selected_payloads = self._select_ranked_payloads(case_payloads)
        case_results = tuple(
            self._write_case_review(rank_name=rank_name, payload=payload)
            for rank_name, payload in selected_payloads
        )

        return self._write_manifest(
            checkpoint_epoch_number=checkpoint.epoch_number,
            case_results=case_results,
        )

    def _run_validation_inference(
        self,
        *,
        model: torch.nn.Module,
        validation_loader: Iterable[Mapping[str, object]],
    ) -> tuple[dict[str, object], ...]:
        """Run inference and retain tensors needed for slice-review figures."""
        case_payloads: list[dict[str, object]] = []

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
            raise RuntimeError("Validation visualization produced no case payloads.")

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
        """Calculate Dice and HD95 values for one validation volume."""
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
            mean_foreground_hd95_mm=self._mean_hd95(hd95_result),
        )

    def _select_ranked_payloads(
        self,
        case_payloads: tuple[dict[str, object], ...],
    ) -> tuple[tuple[str, dict[str, object]], ...]:
        """Select worst, middle, and best payloads by mean foreground Dice."""
        ranked_payloads = tuple(
            sorted(
                case_payloads,
                key=lambda payload: (
                    cast(
                        ValidationInferenceCaseResult,
                        payload["case_result"],
                    ).mean_foreground_dice,
                    cast(ValidationInferenceCaseResult, payload["case_result"]).volume_id,
                ),
            )
        )
        middle_index = len(ranked_payloads) // 2
        selected = (
            ("worst", ranked_payloads[0]),
            ("middle", ranked_payloads[middle_index]),
            ("best", ranked_payloads[-1]),
        )

        return selected[: self._visualization_config.export_case_count]

    def _write_case_review(
        self,
        *,
        rank_name: str,
        payload: dict[str, object],
    ) -> ValidationVisualizationCaseResult:
        """Write all selected slice figures for one ranked validation case."""
        output_dir = self._visualization_config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        case_result = cast(ValidationInferenceCaseResult, payload["case_result"])
        image = cast(Tensor, payload["image"])
        mask = cast(Tensor, payload["mask"])
        prediction = cast(Tensor, payload["prediction"])
        slice_indices = self.select_slice_indices(
            mask=mask,
            prediction=prediction,
            slices_per_case=self._visualization_config.slices_per_case,
        )
        file_paths: list[Path] = []

        for slice_index in slice_indices:
            path = output_dir / f"{rank_name}_{case_result.volume_id}_slice_{slice_index:02d}.png"
            self._write_slice_figure(
                path=path,
                title=self._build_figure_title(
                    rank_name=rank_name,
                    case_result=case_result,
                    slice_index=slice_index,
                ),
                slice_data=(
                    image[slice_index].numpy(),
                    mask[slice_index].numpy(),
                    prediction[slice_index].numpy(),
                ),
            )
            file_paths.append(path)

        return ValidationVisualizationCaseResult(
            rank_name=rank_name,
            volume_id=case_result.volume_id,
            slice_indices=slice_indices,
            file_paths=tuple(file_paths),
        )

    @staticmethod
    def select_slice_indices(
        *,
        mask: Tensor,
        prediction: Tensor,
        slices_per_case: int,
    ) -> tuple[int, ...]:
        """Select central and foreground-extent slices without duplicates."""
        if mask.ndim != _SPATIAL_DIMENSION_COUNT or prediction.ndim != _SPATIAL_DIMENSION_COUNT:
            raise ValueError("Mask and prediction must be three-dimensional [D,H,W].")

        if tuple(mask.shape) != tuple(prediction.shape):
            raise ValueError("Mask and prediction shapes must match.")

        if slices_per_case <= 0:
            raise ValueError("Slices per case must be positive.")

        depth = int(mask.shape[0])
        central_index = depth // 2
        foreground = (mask > 0) | (prediction > 0)
        foreground_area = foreground.sum(dim=(1, 2))
        foreground_indices = torch.nonzero(foreground_area > 0, as_tuple=False).flatten()
        candidates: list[int] = [central_index]

        if int(foreground_indices.numel()) > 0:
            candidates.extend(
                (
                    int(foreground_indices[0].item()),
                    int(foreground_indices[-1].item()),
                )
            )

        offset = 1

        while len(dict.fromkeys(candidates)) < min(slices_per_case, depth):
            lower_index = central_index - offset
            upper_index = central_index + offset

            if lower_index >= 0:
                candidates.append(lower_index)

            if upper_index < depth:
                candidates.append(upper_index)

            offset += 1

        unique_indices = tuple(
            sorted(
                index
                for index in dict.fromkeys(candidates)
                if 0 <= index < depth
            )
        )

        return unique_indices[: min(slices_per_case, depth)]

    def _write_slice_figure(
        self,
        *,
        path: Path,
        title: str,
        slice_data: tuple[NDArray[np.float32], NDArray[np.int64], NDArray[np.int64]],
    ) -> None:
        """Save one multi-panel validation slice-review figure."""
        image_slice, mask_slice, prediction_slice = slice_data
        figure, axes = plt.subplots(
            1,
            6,
            figsize=(18, 3.8),
            constrained_layout=True,
        )
        figure.suptitle(title, fontsize=10)
        self._show_mri(axes[0], image_slice, title="MRI")
        self._show_mask(axes[1], mask_slice, title="Ground Truth")
        self._show_mask(axes[2], prediction_slice, title="Prediction")
        self._show_overlay(axes[3], image_slice, mask_slice, title="MRI + GT")
        self._show_overlay(axes[4], image_slice, prediction_slice, title="MRI + Prediction")
        self._show_contours(
            axes[5],
            image_slice,
            mask_slice,
            prediction_slice,
            title="GT + Prediction Contours",
        )

        for axis in axes:
            axis.axis("off")

        figure.savefig(path, dpi=150)
        plt.close(figure)

    @staticmethod
    def _build_figure_title(
        *,
        rank_name: str,
        case_result: ValidationInferenceCaseResult,
        slice_index: int,
    ) -> str:
        """Build a compact figure title with Dice and HD95 values."""
        return (
            f"{rank_name.upper()} | {case_result.volume_id} | slice {slice_index} | "
            f"Dice RV/Myo/LV/Mean: {case_result.rv_dice:.3f}/"
            f"{case_result.myocardium_dice:.3f}/{case_result.lv_dice:.3f}/"
            f"{case_result.mean_foreground_dice:.3f}\n"
            f"HD95 mm RV/Myo/LV/Mean: {case_result.rv_hd95_mm:.2f}/"
            f"{case_result.myocardium_hd95_mm:.2f}/{case_result.lv_hd95_mm:.2f}/"
            f"{case_result.mean_foreground_hd95_mm:.2f}"
        )

    @staticmethod
    def _show_mri(
        axis: Axes,
        image_slice: NDArray[np.float32],
        *,
        title: str,
    ) -> None:
        """Draw a grayscale MRI panel."""
        axis.imshow(image_slice, cmap="gray")
        axis.set_title(title)

    @staticmethod
    def _show_mask(
        axis: Axes,
        label_slice: NDArray[np.int64],
        *,
        title: str,
    ) -> None:
        """Draw a label-only panel."""
        axis.imshow(label_slice, cmap=_MASK_CMAP, norm=_MASK_NORM, interpolation="nearest")
        axis.set_title(title)

    @staticmethod
    def _show_overlay(
        axis: Axes,
        image_slice: NDArray[np.float32],
        label_slice: NDArray[np.int64],
        *,
        title: str,
    ) -> None:
        """Draw an MRI panel with semi-transparent labels."""
        masked_labels = np.ma.masked_where(label_slice == 0, label_slice)
        axis.imshow(image_slice, cmap="gray")
        axis.imshow(
            masked_labels,
            cmap=_OVERLAY_CMAP,
            norm=_OVERLAY_NORM,
            alpha=0.45,
            interpolation="nearest",
        )
        axis.set_title(title)

    @staticmethod
    def _show_contours(
        axis: Axes,
        image_slice: NDArray[np.float32],
        mask_slice: NDArray[np.int64],
        prediction_slice: NDArray[np.int64],
        *,
        title: str,
    ) -> None:
        """Draw GT and prediction contours together on MRI."""
        axis.imshow(image_slice, cmap="gray")

        for label, color in ((1, "#d62728"), (2, "#2ca02c"), (3, "#1f77b4")):
            if np.any(mask_slice == label):
                axis.contour(mask_slice == label, levels=(0.5,), colors=color, linewidths=1.4)

            if np.any(prediction_slice == label):
                axis.contour(
                    prediction_slice == label,
                    levels=(0.5,),
                    colors=color,
                    linewidths=0.8,
                    linestyles="dashed",
                )

        axis.set_title(title)

    def _write_manifest(
        self,
        *,
        checkpoint_epoch_number: int,
        case_results: tuple[ValidationVisualizationCaseResult, ...],
    ) -> Path:
        """Write a compact JSON manifest for generated review figures."""
        manifest_path = self._visualization_config.output_dir / "visualization_manifest.json"
        manifest = {
            "checkpoint_epoch": checkpoint_epoch_number,
            "output_directory": str(self._visualization_config.output_dir),
            "report_csv_path": str(self._visualization_config.report_csv_path),
            "report_json_path": str(self._visualization_config.report_json_path),
            "selected_cases": {
                case_result.rank_name: {
                    "volume_id": case_result.volume_id,
                    "slice_indices": case_result.slice_indices,
                    "file_paths": [str(path) for path in case_result.file_paths],
                }
                for case_result in case_results
            },
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        return manifest_path

    def _resolve_device(self) -> torch.device:
        """Resolve the configured execution device."""
        if self._visualization_config.device == "cpu":
            return torch.device("cpu")

        if self._visualization_config.device == "cuda":
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
        selected_cases = training_cases[: self._visualization_config.patient_count]

        if len(selected_cases) != self._visualization_config.patient_count:
            raise ValueError(
                "ACDC training patient selection produced "
                f"{len(selected_cases)} cases, but "
                f"{self._visualization_config.patient_count} were required."
            )

        return selected_cases

    def _create_data_loaders(self) -> AcdcDataLoaders:
        """Create fresh deterministic validation DataLoaders."""
        return AcdcDataLoaderFactory(
            preprocessing_config=self._app_config.preprocessing,
            validation_config=self._app_config.validation,
            validation_fraction=self._visualization_config.validation_fraction,
            random_seed=self._visualization_config.random_seed,
            batch_size=1,
            num_workers=0,
            pin_memory=False,
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
        random.seed(self._visualization_config.random_seed)
        torch.manual_seed(self._visualization_config.random_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self._visualization_config.random_seed)

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
    def _mean_hd95(
        hd95_values_mm: tuple[float, float, float],
    ) -> float:
        """Return mean foreground HD95, preserving infinity if present."""
        if any(np.isinf(value) for value in hd95_values_mm):
            return float("inf")

        return float(np.mean(np.asarray(hd95_values_mm, dtype=np.float64)))
