from dataclasses import dataclass
from pathlib import Path

from cardiac_segmentation.evaluation.validation_inference_case_result import (
    ValidationInferenceCaseResult,
)

_VISUALIZATION_COUNT = 3


@dataclass(frozen=True, slots=True)
class ValidationInferenceReport:
    """Store validation inference report metadata and output paths."""

    checkpoint_epoch_number: int
    case_results: tuple[ValidationInferenceCaseResult, ...]
    report_csv_path: Path
    summary_json_path: Path
    visualization_paths: tuple[Path, Path, Path]

    def __post_init__(self) -> None:
        """Validate report content and generated output paths."""
        if self.checkpoint_epoch_number <= 0:
            raise ValueError("Checkpoint epoch number must be positive.")

        if not self.case_results:
            raise ValueError("Validation inference case results must not be empty.")

        if not self.report_csv_path.is_file():
            raise FileNotFoundError(
                f"CSV report does not exist: {self.report_csv_path}"
            )

        if not self.summary_json_path.is_file():
            raise FileNotFoundError(
                f"JSON summary does not exist: {self.summary_json_path}"
            )

        if len(self.visualization_paths) != _VISUALIZATION_COUNT:
            raise ValueError("Exactly three visualization paths are required.")

        if len(set(self.visualization_paths)) != len(self.visualization_paths):
            raise ValueError("Visualization paths must be unique.")

    @property
    def validation_volume_count(self) -> int:
        """Return the number of evaluated validation volumes."""
        return len(self.case_results)

    @property
    def mean_foreground_dice(self) -> float:
        """Return the mean foreground Dice across validation volumes."""
        return sum(result.mean_foreground_dice for result in self.case_results) / len(
            self.case_results
        )
