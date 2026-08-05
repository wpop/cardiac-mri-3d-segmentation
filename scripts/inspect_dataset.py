import argparse
import logging
from pathlib import Path
from typing import Final, cast

from cardiac_segmentation.config.loader import AppConfigLoader
from cardiac_segmentation.data import AcdcDatasetInspectionRunner

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG_PATH: Final[Path] = Path("configs/data.yaml")
_LOGGER = logging.getLogger(__name__)


def _parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for dataset inspection."""
    parser = argparse.ArgumentParser(
        description="Inspect the real ACDC cardiac MRI dataset."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG_PATH,
        help="Path to the dataset inspection YAML configuration.",
    )

    return parser.parse_args()


def main() -> None:
    """Run ACDC inspection and report generated artifact locations."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    arguments = _parse_arguments()
    config_path = cast(Path, arguments.config)

    config = AppConfigLoader(
        project_root=_PROJECT_ROOT
    ).load(config_path)

    report = AcdcDatasetInspectionRunner(config).run()

    _LOGGER.info(
        "Inspected %d patients and %d cardiac phases.",
        report.patient_count,
        report.phase_count,
    )
    _LOGGER.info(
        "JSON report: %s",
        config.inspection.report_path,
    )
    _LOGGER.info(
        "CSV summary: %s",
        config.inspection.summary_path,
    )


if __name__ == "__main__":
    main()
