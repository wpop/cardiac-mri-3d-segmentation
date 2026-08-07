# ruff: noqa: T201
from pathlib import Path
from typing import Final

from cardiac_segmentation.config import (
    AppConfigLoader,
    ValidationInferenceConfigLoader,
)
from cardiac_segmentation.evaluation import (
    ValidationInferenceCaseResult,
    ValidationInferenceExperiment,
)

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
_APP_CONFIG_PATH: Final[Path] = Path("configs/data.yaml")
_INFERENCE_CONFIG_PATH: Final[Path] = Path("configs/validation_inference.yaml")


def main() -> None:
    """Run the configured validation inference report."""
    app_config = AppConfigLoader(
        project_root=_PROJECT_ROOT,
    ).load(_APP_CONFIG_PATH)
    inference_config = ValidationInferenceConfigLoader(
        project_root=_PROJECT_ROOT,
    ).load(_INFERENCE_CONFIG_PATH)
    experiment = ValidationInferenceExperiment(
        app_config=app_config,
        inference_config=inference_config,
    )

    print("Cardiac MRI validation inference report")
    print(f"Execution device: {experiment.device}")
    print(f"Validation patient count: {len(experiment.validation_patient_ids)}")
    print(f"Validation volume count: {experiment.validation_volume_count}")

    report = experiment.run()
    ranked_results = _rank_results(report.case_results)
    worst_result = ranked_results[0]
    middle_result = ranked_results[len(ranked_results) // 2]
    best_result = ranked_results[-1]

    print(f"Checkpoint epoch: {report.checkpoint_epoch_number}")
    print(f"Validation patient count: {len(experiment.validation_patient_ids)}")
    print(f"Validation volume count: {report.validation_volume_count}")
    _print_metric("RV", tuple(result.rv_dice for result in report.case_results))
    _print_metric(
        "Myocardium",
        tuple(result.myocardium_dice for result in report.case_results),
    )
    _print_metric("LV", tuple(result.lv_dice for result in report.case_results))
    _print_metric(
        "Mean foreground",
        tuple(result.mean_foreground_dice for result in report.case_results),
    )
    _print_case("Worst", worst_result)
    _print_case("Middle", middle_result)
    _print_case("Best", best_result)
    print(f"CSV report path: {report.report_csv_path}")
    print(f"JSON summary path: {report.summary_json_path}")

    for path in report.visualization_paths:
        print(f"Visualization path: {path}")


def _rank_results(
    case_results: tuple[ValidationInferenceCaseResult, ...],
) -> tuple[ValidationInferenceCaseResult, ...]:
    """Sort case results from worst to best by mean foreground Dice."""
    return tuple(
        sorted(
            case_results,
            key=lambda result: (
                result.mean_foreground_dice,
                result.volume_id,
            ),
        )
    )


def _print_metric(
    name: str,
    values: tuple[float, ...],
) -> None:
    """Print aggregate metric statistics."""
    sorted_values = tuple(sorted(values))
    count = len(sorted_values)
    midpoint = count // 2

    if count % 2 == 0:
        median = (sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2.0
    else:
        median = sorted_values[midpoint]

    print(
        f"{name} Dice: "
        f"mean={sum(values) / count:.6f}, "
        f"median={median:.6f}, "
        f"min={min(values):.6f}, "
        f"max={max(values):.6f}"
    )


def _print_case(
    label: str,
    result: ValidationInferenceCaseResult,
) -> None:
    """Print one ranked case summary."""
    print(
        f"{label} case: "
        f"{result.volume_id}, "
        f"mean foreground Dice={result.mean_foreground_dice:.6f}"
    )


if __name__ == "__main__":
    main()
