from pathlib import Path
from typing import Final

import pytest
import torch
from torch import Tensor

from cardiac_segmentation.config.app_config import AppConfig
from cardiac_segmentation.config.loader import AppConfigLoader
from cardiac_segmentation.data import (
    AcdcDatasetIndexer,
    AcdcInfoParser,
    AcdcPatientCase,
    AcdcSegmentationDataset,
)
from cardiac_segmentation.metrics import MulticlassDiceMetric3D
from cardiac_segmentation.metrics.multiclass_dice_metric_result import (
    MulticlassDiceMetricResult,
)
from cardiac_segmentation.models import CompactUNet3D

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_CONFIG_PATH: Final[Path] = Path("configs/data.yaml")
_TEST_BASE_CHANNELS: Final[int] = 2
_RANDOM_SEED: Final[int] = 42
_DICE_SMOOTH: Final[float] = 1e-5
_FLOAT64_ABSOLUTE_TOLERANCE: Final[float] = 1e-10


@pytest.mark.acdc
@pytest.mark.integration
def test_multiclass_dice_metric_3d_accumulates_real_acdc_ed_and_es_predictions() -> None:
    """Accumulate foreground Dice for real ED and ES model predictions."""
    torch.manual_seed(_RANDOM_SEED)
    config = AppConfigLoader(
        project_root=_PROJECT_ROOT,
    ).load(_CONFIG_PATH)
    training_case = _select_training_case(config)
    dataset = AcdcSegmentationDataset(
        patient_cases=(training_case,),
        preprocessing_config=config.preprocessing,
        validation_config=config.validation,
    )
    model = CompactUNet3D(
        in_channels=1,
        num_classes=len(config.validation.expected_labels),
        base_channels=_TEST_BASE_CHANNELS,
    )
    model.eval()
    metric = MulticlassDiceMetric3D(
        num_classes=len(config.validation.expected_labels),
        include_background=False,
        smooth=_DICE_SMOOTH,
    )
    reference_confusion_matrix = torch.zeros(
        (
            len(config.validation.expected_labels),
            len(config.validation.expected_labels),
        ),
        dtype=torch.float64,
    )

    assert metric.volume_count == 0

    with pytest.raises(RuntimeError):
        metric.compute()

    ed_input_shape, es_input_shape = _update_metric_with_real_ed_and_es(
        dataset=dataset,
        model=model,
        metric=metric,
        reference_confusion_matrix=reference_confusion_matrix,
        num_classes=len(config.validation.expected_labels),
    )
    result = metric.compute()
    reference_result = _calculate_reference_result(
        reference_confusion_matrix=reference_confusion_matrix,
    )

    assert ed_input_shape == (1, 1, 24, 192, 192)
    assert es_input_shape == (1, 1, 24, 192, 192)
    _assert_metric_result_matches_contract(result)
    _assert_metric_result_matches_reference(
        result=result,
        reference_result=reference_result,
    )
    metric.reset()
    assert metric.volume_count == 0

    with pytest.raises(RuntimeError):
        metric.compute()


def _select_training_case(
    config: AppConfig,
) -> AcdcPatientCase:
    """Return the first real training patient case."""
    patient_cases = AcdcDatasetIndexer(
        dataset_root=config.dataset.root_dir,
        info_parser=AcdcInfoParser(),
    ).index()

    return next(
        patient_case
        for patient_case in patient_cases
        if patient_case.split_name == "training"
    )


def _update_metric_with_real_ed_and_es(
    dataset: AcdcSegmentationDataset,
    model: CompactUNet3D,
    metric: MulticlassDiceMetric3D,
    reference_confusion_matrix: Tensor,
    num_classes: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Run real ED and ES samples through the model and metric."""
    input_shapes: list[tuple[int, ...]] = []

    with torch.inference_mode():
        for dataset_index in (0, 1):
            item = dataset[dataset_index]
            image_tensor = _require_tensor(item["image"]).unsqueeze(0)
            target = _require_tensor(item["mask"]).unsqueeze(0)
            logits = model(image_tensor)
            metric.update(
                logits=logits,
                target=target,
            )
            prediction = torch.argmax(
                logits,
                dim=1,
            )
            reference_confusion_matrix += _build_reference_confusion_matrix(
                prediction=prediction,
                target=target,
                num_classes=num_classes,
            )
            input_shapes.append(tuple(int(dimension) for dimension in image_tensor.shape))

    return (
        input_shapes[0],
        input_shapes[1],
    )


def _build_reference_confusion_matrix(
    prediction: Tensor,
    target: Tensor,
    num_classes: int,
) -> Tensor:
    """Build an independent confusion matrix from real predictions and targets."""
    encoded_pairs = target.reshape(-1) * num_classes + prediction.reshape(-1)

    return torch.bincount(
        encoded_pairs,
        minlength=num_classes * num_classes,
    ).reshape(
        num_classes,
        num_classes,
    ).to(dtype=torch.float64)


def _calculate_reference_result(
    reference_confusion_matrix: Tensor,
) -> MulticlassDiceMetricResult:
    """Calculate reference foreground Dice from an independently accumulated matrix."""
    intersection = torch.diag(reference_confusion_matrix)
    target_count = torch.sum(
        reference_confusion_matrix,
        dim=1,
    )
    prediction_count = torch.sum(
        reference_confusion_matrix,
        dim=0,
    )
    class_dice = (
        (2.0 * intersection + _DICE_SMOOTH)
        / (target_count + prediction_count + _DICE_SMOOTH)
    )
    included_class_indices = (1, 2, 3)
    included_dice = class_dice[
        torch.tensor(
            included_class_indices,
            dtype=torch.int64,
        )
    ]

    return MulticlassDiceMetricResult(
        included_class_indices=included_class_indices,
        per_class_dice=tuple(float(value) for value in included_dice.tolist()),
        mean_dice=float(torch.mean(included_dice).item()),
        volume_count=2,
    )


def _assert_metric_result_matches_contract(
    result: MulticlassDiceMetricResult,
) -> None:
    """Verify accumulated metric result values."""
    assert result.volume_count == 2
    assert result.included_class_indices == (1, 2, 3)
    assert len(result.per_class_dice) == 3
    assert all(0.0 <= dice_value <= 1.0 for dice_value in result.per_class_dice)
    assert 0.0 <= result.mean_dice <= 1.0
    assert result.mean_dice == pytest.approx(
        sum(result.per_class_dice) / len(result.per_class_dice),
        abs=_FLOAT64_ABSOLUTE_TOLERANCE,
    )


def _assert_metric_result_matches_reference(
    result: MulticlassDiceMetricResult,
    reference_result: MulticlassDiceMetricResult,
) -> None:
    """Verify the metric matches independently calculated foreground Dice."""
    assert result.included_class_indices == reference_result.included_class_indices
    assert result.included_class_indices == (1, 2, 3)
    assert result.per_class_dice == pytest.approx(
        reference_result.per_class_dice,
        abs=_FLOAT64_ABSOLUTE_TOLERANCE,
    )
    assert result.mean_dice == pytest.approx(
        reference_result.mean_dice,
        abs=_FLOAT64_ABSOLUTE_TOLERANCE,
    )


def _require_tensor(
    value: Tensor | str,
) -> Tensor:
    """Return a Dataset item value as a tensor."""
    if not isinstance(value, Tensor):
        raise TypeError("Dataset item value must be a tensor.")

    return value
