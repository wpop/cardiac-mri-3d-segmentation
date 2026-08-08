import json
from pathlib import Path
from typing import Final

import matplotlib.image as mpimg

from cardiac_segmentation.config import (
    AppConfigLoader,
    ValidationVisualizationConfig,
)
from cardiac_segmentation.evaluation import ValidationVisualizationExperiment

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_APP_CONFIG_PATH: Final[Path] = Path("configs/data.yaml")
_FINAL_CHECKPOINT_PATH: Final[Path] = (
    _PROJECT_ROOT / "artifacts/checkpoints/final_patient_level_training.pt"
)


def test_validation_visualization_experiment_writes_manifest_and_review_pngs(
    tmp_path: Path,
) -> None:
    """Run real validation visualization and verify generated review artifacts."""
    app_config = AppConfigLoader(project_root=_PROJECT_ROOT).load(_APP_CONFIG_PATH)
    visualization_config = ValidationVisualizationConfig(
        patient_count=8,
        validation_fraction=0.25,
        random_seed=42,
        base_channels=8,
        device="cpu",
        checkpoint_path=_FINAL_CHECKPOINT_PATH,
        output_dir=tmp_path / "validation_slice_review",
        report_csv_path=tmp_path / "validation.csv",
        report_json_path=tmp_path / "validation_summary.json",
        export_case_count=3,
        slices_per_case=3,
    )
    experiment = ValidationVisualizationExperiment(
        app_config=app_config,
        visualization_config=visualization_config,
    )

    manifest_path = experiment.run()

    assert visualization_config.output_dir.is_dir()
    assert manifest_path.is_file()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected_cases = manifest["selected_cases"]
    assert set(selected_cases) == {"worst", "middle", "best"}
    assert selected_cases["worst"]["volume_id"]
    assert selected_cases["middle"]["volume_id"]
    assert selected_cases["best"]["volume_id"]

    png_paths = sorted(visualization_config.output_dir.glob("*.png"))
    assert len(png_paths) >= 9

    for png_path in png_paths:
        image = mpimg.imread(png_path)
        assert image.size > 0

    for case_name in ("worst", "middle", "best"):
        assert len(selected_cases[case_name]["slice_indices"]) == 3
        assert len(selected_cases[case_name]["file_paths"]) == 3
