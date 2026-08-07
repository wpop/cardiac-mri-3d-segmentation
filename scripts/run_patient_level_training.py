# ruff: noqa: T201
from pathlib import Path
from typing import Final

from cardiac_segmentation.config import (
    AppConfigLoader,
    PatientLevelTrainingConfigLoader,
)
from cardiac_segmentation.training import (
    PatientLevelTrainingExperiment,
    SegmentationTrainingEpochRecord,
)

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
_APP_CONFIG_PATH: Final[Path] = Path("configs/data.yaml")
_TRAINING_CONFIG_PATH: Final[Path] = Path("configs/patient_level_training.yaml")


def _print_epoch_record(
    record: SegmentationTrainingEpochRecord,
) -> None:
    """Print one completed epoch immediately."""
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
        f"validation duration={record.validation_duration_seconds:.2f}s",
        flush=True,
    )


def main() -> None:
    """Run the configured patient-level cardiac MRI training experiment."""
    app_config = AppConfigLoader(
        project_root=_PROJECT_ROOT,
    ).load(_APP_CONFIG_PATH)
    training_config = PatientLevelTrainingConfigLoader(
        project_root=_PROJECT_ROOT,
    ).load(_TRAINING_CONFIG_PATH)
    experiment = PatientLevelTrainingExperiment(
        app_config=app_config,
        training_config=training_config,
    )

    print("Patient-level cardiac MRI segmentation training")
    print(f"Execution device: {experiment.device}")
    print(f"Training patient IDs: {', '.join(experiment.training_patient_ids)}")
    print(f"Validation patient IDs: {', '.join(experiment.validation_patient_ids)}")
    print(f"Training patient count: {len(experiment.training_patient_ids)}")
    print(f"Validation patient count: {len(experiment.validation_patient_ids)}")
    print(f"Training volume count: {experiment.training_volume_count}")
    print(f"Validation volume count: {experiment.validation_volume_count}")
    print(f"Configured epoch count: {training_config.epoch_count}")
    print(f"Checkpoint path: {training_config.checkpoint_path}")

    history = experiment.run(epoch_callback=_print_epoch_record)

    print(f"Best epoch: {history.best_epoch_number}")
    print(
        "Best validation mean foreground Dice: "
        f"{history.best_record.validation_result.dice_result.mean_dice:.6f}"
    )
    print(f"Checkpoint path: {history.checkpoint_path}")


if __name__ == "__main__":
    main()
