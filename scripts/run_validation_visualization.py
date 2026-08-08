import sys
from pathlib import Path
from typing import Final

from cardiac_segmentation.config import (
    AppConfigLoader,
    ValidationVisualizationConfigLoader,
)
from cardiac_segmentation.evaluation import ValidationVisualizationExperiment

_APP_CONFIG_PATH: Final[Path] = Path("configs/data.yaml")
_VISUALIZATION_CONFIG_PATH: Final[Path] = Path("configs/validation_visualization.yaml")


def main() -> None:
    """Run validation slice-review visualization generation."""
    project_root = Path.cwd()
    app_config = AppConfigLoader(project_root=project_root).load(_APP_CONFIG_PATH)
    visualization_config = ValidationVisualizationConfigLoader(
        project_root=project_root,
    ).load(_VISUALIZATION_CONFIG_PATH)
    experiment = ValidationVisualizationExperiment(
        app_config=app_config,
        visualization_config=visualization_config,
    )
    manifest_path = experiment.run()
    sys.stdout.write(f"Validation visualization manifest: {manifest_path}\n")


if __name__ == "__main__":
    main()
