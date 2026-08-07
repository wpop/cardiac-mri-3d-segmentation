# ruff: noqa: T201
from pathlib import Path
from typing import Final

from cardiac_segmentation.config import (
    AppConfigLoader,
    PatientLevelResumeTrainingConfigLoader,
)
from cardiac_segmentation.training import (
    PatientLevelResumeTrainingExperiment,
    SegmentationTrainingEpochRecord,
)

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
_APP_CONFIG_PATH: Final[Path] = Path("configs/data.yaml")
_RESUME_CONFIG_PATH: Final[Path] = Path("configs/patient_level_resume_training.yaml")


def _print_epoch_record(
    record: SegmentationTrainingEpochRecord,
) -> None:
    """Print one completed resumed epoch immediately."""
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
    """Resume the configured patient-level cardiac MRI training experiment."""
    app_config = AppConfigLoader(
        project_root=_PROJECT_ROOT,
    ).load(_APP_CONFIG_PATH)
    resume_config = PatientLevelResumeTrainingConfigLoader(
        project_root=_PROJECT_ROOT,
    ).load(_RESUME_CONFIG_PATH)
    experiment = PatientLevelResumeTrainingExperiment(
        app_config=app_config,
        resume_config=resume_config,
    )

    print("Resume patient-level cardiac MRI segmentation training", flush=True)
    print(f"Execution device: {experiment.device}", flush=True)
    print(f"Resume checkpoint path: {experiment.resume_checkpoint_path}", flush=True)
    print(f"Training patient IDs: {', '.join(experiment.training_patient_ids)}", flush=True)
    print(f"Validation patient IDs: {', '.join(experiment.validation_patient_ids)}", flush=True)
    print(f"Training patient count: {len(experiment.training_patient_ids)}", flush=True)
    print(f"Validation patient count: {len(experiment.validation_patient_ids)}", flush=True)
    print(f"Training volume count: {experiment.training_volume_count}", flush=True)
    print(f"Validation volume count: {experiment.validation_volume_count}", flush=True)
    print(f"Target final epoch number: {experiment.final_epoch_number}", flush=True)

    history = experiment.run(epoch_callback=_print_epoch_record)

    print(f"Resumed from epoch: {history.resumed_from_epoch_number}", flush=True)
    print(f"Final completed epoch: {history.final_epoch_number}", flush=True)
    print(f"Selected best epoch: {history.best_epoch_number}", flush=True)
    print(
        f"Best validation mean foreground Dice: {history.best_validation_mean_dice:.6f}",
        flush=True,
    )
    print(f"Final checkpoint path: {history.checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
