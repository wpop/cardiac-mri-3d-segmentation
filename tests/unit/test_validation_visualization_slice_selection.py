from pathlib import Path

import pytest
import torch

from cardiac_segmentation.config import ValidationVisualizationConfig
from cardiac_segmentation.evaluation import ValidationVisualizationExperiment


def test_validation_visualization_slice_selection_uses_center_and_foreground_extent() -> None:
    mask = torch.zeros((9, 4, 4), dtype=torch.int64)
    prediction = torch.zeros((9, 4, 4), dtype=torch.int64)
    mask[2, 1, 1] = 1
    prediction[7, 2, 2] = 3

    slice_indices = ValidationVisualizationExperiment.select_slice_indices(
        mask=mask,
        prediction=prediction,
        slices_per_case=3,
    )

    assert slice_indices == (2, 4, 7)


def test_validation_visualization_slice_selection_avoids_duplicates() -> None:
    mask = torch.zeros((5, 4, 4), dtype=torch.int64)
    prediction = torch.zeros((5, 4, 4), dtype=torch.int64)
    mask[2, 1, 1] = 1

    slice_indices = ValidationVisualizationExperiment.select_slice_indices(
        mask=mask,
        prediction=prediction,
        slices_per_case=3,
    )

    assert len(slice_indices) == 3
    assert len(set(slice_indices)) == 3
    assert 2 in slice_indices


def test_validation_visualization_slice_selection_deduplicates_before_limiting() -> None:
    mask = torch.zeros((2, 4, 4), dtype=torch.int64)
    prediction = torch.zeros((2, 4, 4), dtype=torch.int64)
    mask[1, 1, 1] = 1

    slice_indices = ValidationVisualizationExperiment.select_slice_indices(
        mask=mask,
        prediction=prediction,
        slices_per_case=3,
    )

    assert slice_indices == (0, 1)


def test_validation_visualization_config_rejects_more_than_three_cases() -> None:
    with pytest.raises(ValueError, match="Export case count must be between 1 and 3"):
        ValidationVisualizationConfig(
            patient_count=8,
            validation_fraction=0.25,
            random_seed=42,
            base_channels=8,
            device="cpu",
            checkpoint_path=Path("checkpoint.pt"),
            output_dir=Path("visualizations"),
            report_csv_path=Path("report.csv"),
            report_json_path=Path("summary.json"),
            export_case_count=4,
            slices_per_case=3,
        )
