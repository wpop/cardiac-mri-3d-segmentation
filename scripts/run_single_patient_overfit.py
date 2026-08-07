# ruff: noqa: T201
from pathlib import Path
from typing import Final

from cardiac_segmentation.config import (
    AppConfigLoader,
    SinglePatientOverfitConfigLoader,
)
from cardiac_segmentation.training import SinglePatientOverfitExperiment

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
_APP_CONFIG_PATH: Final[Path] = Path("configs/data.yaml")
_OVERFIT_CONFIG_PATH: Final[Path] = Path("configs/single_patient_overfit.yaml")


def main() -> None:
    """Run the configured diagnostic single-patient overfit experiment."""
    app_config = AppConfigLoader(
        project_root=_PROJECT_ROOT,
    ).load(_APP_CONFIG_PATH)
    overfit_config = SinglePatientOverfitConfigLoader(
        project_root=_PROJECT_ROOT,
    ).load(_OVERFIT_CONFIG_PATH)
    experiment = SinglePatientOverfitExperiment(
        app_config=app_config,
        overfit_config=overfit_config,
    )

    print("Single-patient overfit experiment")
    print(f"Patient ID: {experiment.patient_id}")
    print(f"Dataset volume count: {experiment.dataset_volume_count}")
    print(f"Execution device: {experiment.device}")
    print(f"Configured epoch count: {overfit_config.epoch_count}")
    print(f"Checkpoint path: {overfit_config.checkpoint_path}")

    history = experiment.run()

    for record in history.epoch_records:
        print(
            "Epoch "
            f"{record.epoch_number}: "
            f"training loss={record.training_result.average_loss:.6f}, "
            f"validation loss={record.validation_result.average_loss:.6f}, "
            "training mean foreground Dice="
            f"{record.training_result.dice_result.mean_dice:.6f}, "
            "validation mean foreground Dice="
            f"{record.validation_result.dice_result.mean_dice:.6f}, "
            f"training duration={record.training_duration_seconds:.2f}s, "
            f"validation duration={record.validation_duration_seconds:.2f}s"
        )

    print(f"Selected best epoch: {history.best_epoch_number}")
    print(
        "Best validation mean foreground Dice: "
        f"{history.best_record.validation_result.dice_result.mean_dice:.6f}"
    )
    print(f"Final checkpoint path: {history.checkpoint_path}")


if __name__ == "__main__":
    main()
