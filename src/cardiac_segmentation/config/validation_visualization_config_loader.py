from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Final

from cardiac_segmentation.config.loader import AppConfigLoader
from cardiac_segmentation.config.validation_visualization_config import (
    ValidationVisualizationConfig,
)


class ValidationVisualizationConfigLoader:
    """Load and validate a validation visualization configuration."""

    _ROOT_KEYS: Final[frozenset[str]] = frozenset({"validation_visualization"})
    _SECTION_KEYS: Final[frozenset[str]] = frozenset(
        {
            "patient_count",
            "validation_fraction",
            "random_seed",
            "base_channels",
            "device",
            "checkpoint_path",
            "output_dir",
            "report_csv_path",
            "report_json_path",
            "export_case_count",
            "slices_per_case",
        }
    )

    def __init__(
        self,
        project_root: Path,
    ) -> None:
        """Initialize the loader with an explicit project root."""
        resolved_project_root = project_root.expanduser().resolve()

        if not resolved_project_root.is_dir():
            raise NotADirectoryError(
                f"Project root directory does not exist: {resolved_project_root}"
            )

        self._project_root = resolved_project_root

    def load(
        self,
        config_path: Path,
    ) -> ValidationVisualizationConfig:
        """Load a YAML file and construct the visualization config."""
        resolved_config_path = self._resolve_config_path(config_path)
        raw_config = AppConfigLoader._read_yaml(resolved_config_path)
        root = AppConfigLoader._require_mapping(
            raw_config,
            context="Configuration root",
        )
        AppConfigLoader._validate_keys(
            root,
            expected_keys=self._ROOT_KEYS,
            context="Configuration root",
        )
        section = self._require_section(root)
        AppConfigLoader._validate_keys(
            section,
            expected_keys=self._SECTION_KEYS,
            context="Validation visualization configuration",
        )

        return ValidationVisualizationConfig(
            patient_count=self._require_integer(section, "patient_count"),
            validation_fraction=AppConfigLoader._require_float(
                section,
                "validation_fraction",
                context="Validation visualization configuration",
            ),
            random_seed=self._require_integer(section, "random_seed"),
            base_channels=self._require_integer(section, "base_channels"),
            device=AppConfigLoader._require_string(
                section,
                "device",
                context="Validation visualization configuration",
            ),
            checkpoint_path=self._resolve_project_path(
                AppConfigLoader._require_string(
                    section,
                    "checkpoint_path",
                    context="Validation visualization configuration",
                )
            ),
            output_dir=self._resolve_project_path(
                AppConfigLoader._require_string(
                    section,
                    "output_dir",
                    context="Validation visualization configuration",
                )
            ),
            report_csv_path=self._resolve_project_path(
                AppConfigLoader._require_string(
                    section,
                    "report_csv_path",
                    context="Validation visualization configuration",
                )
            ),
            report_json_path=self._resolve_project_path(
                AppConfigLoader._require_string(
                    section,
                    "report_json_path",
                    context="Validation visualization configuration",
                )
            ),
            export_case_count=self._require_integer(section, "export_case_count"),
            slices_per_case=self._require_integer(section, "slices_per_case"),
        )

    def _resolve_config_path(
        self,
        config_path: Path,
    ) -> Path:
        """Resolve a configuration path relative to the project root."""
        candidate = config_path.expanduser()

        if not candidate.is_absolute():
            candidate = self._project_root / candidate

        resolved_path = candidate.resolve(strict=False)

        if not resolved_path.is_file():
            raise FileNotFoundError(f"Configuration file does not exist: {resolved_path}")

        return resolved_path

    def _require_section(
        self,
        root: Mapping[str, object],
    ) -> dict[str, object]:
        """Read the required validation visualization section."""
        if "validation_visualization" not in root:
            raise ValueError(
                "Missing required configuration section: validation_visualization"
            )

        return AppConfigLoader._require_mapping(
            root["validation_visualization"],
            context="Configuration section 'validation_visualization'",
        )

    @staticmethod
    def _require_integer(
        section: Mapping[str, object],
        key: str,
    ) -> int:
        """Require an integer value from the visualization section."""
        value = section[key]

        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"Validation visualization configuration key '{key}' must be an integer."
            )

        return value

    def _resolve_project_path(
        self,
        path_value: str,
    ) -> Path:
        """Resolve a configured path relative to the project root."""
        candidate = Path(path_value).expanduser()

        if not candidate.is_absolute():
            candidate = self._project_root / candidate

        return candidate.resolve(strict=False)
