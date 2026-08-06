from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Final, cast

import yaml

from cardiac_segmentation.config.app_config import AppConfig
from cardiac_segmentation.config.dataset_config import DatasetConfig
from cardiac_segmentation.config.inspection_config import InspectionConfig
from cardiac_segmentation.config.preprocessing_config import (
    PreprocessingConfig,
)
from cardiac_segmentation.config.validation_config import ValidationConfig

_SPATIAL_DIMENSION_COUNT: Final[int] = 3


class AppConfigLoader:
    """Load and validate the complete application configuration from YAML."""

    _ROOT_KEYS = frozenset(
        {
            "dataset",
            "inspection",
            "preprocessing",
            "validation",
        }
    )

    _DATASET_KEYS = frozenset(
        {
            "name",
            "root_dir",
        }
    )

    _INSPECTION_KEYS = frozenset(
        {
            "output_dir",
            "report_filename",
            "summary_filename",
            "visualizations_dirname",
        }
    )

    _VALIDATION_KEYS = frozenset(
        {
            "expected_labels",
            "affine_absolute_tolerance",
            "require_finite_intensities",
            "require_positive_voxel_spacing",
        }
    )

    _PREPROCESSING_KEYS = frozenset(
        {
            "target_spacing_mm",
            "target_shape",
            "intensity_lower_percentile",
            "intensity_upper_percentile",
            "normalize_nonzero_only",
        }
    )

    def __init__(self, project_root: Path) -> None:
        """Initialize the loader with an explicit project root directory."""
        resolved_project_root = project_root.expanduser().resolve()

        if not resolved_project_root.is_dir():
            raise NotADirectoryError(
                f"Project root directory does not exist: {resolved_project_root}"
            )

        self._project_root = resolved_project_root

    def load(self, config_path: Path) -> AppConfig:
        """Load a YAML file and construct a validated application configuration."""
        resolved_config_path = self._resolve_config_path(config_path)
        raw_config = self._read_yaml(resolved_config_path)

        root = self._require_mapping(raw_config, context="Configuration root")
        self._validate_keys(
            root,
            expected_keys=self._ROOT_KEYS,
            context="Configuration root",
        )

        dataset_section = self._require_section(root, "dataset")
        inspection_section = self._require_section(root, "inspection")
        validation_section = self._require_section(root, "validation")
        preprocessing_section = self._require_section(root, "preprocessing")

        return AppConfig(
            dataset=self._build_dataset_config(dataset_section),
            inspection=self._build_inspection_config(inspection_section),
            validation=self._build_validation_config(validation_section),
            preprocessing=self._build_preprocessing_config(
                preprocessing_section
            ),
        )

    def _resolve_config_path(self, config_path: Path) -> Path:
        """Resolve a configuration path relative to the project root."""
        candidate = config_path.expanduser()

        if not candidate.is_absolute():
            candidate = self._project_root / candidate

        resolved_path = candidate.resolve(strict=False)

        if not resolved_path.is_file():
            raise FileNotFoundError(f"Configuration file does not exist: {resolved_path}")

        return resolved_path

    @staticmethod
    def _read_yaml(config_path: Path) -> object:
        """Read and parse a YAML configuration file."""
        try:
            file_content = config_path.read_text(encoding="utf-8")
        except OSError as error:
            raise OSError(f"Failed to read configuration file: {config_path}") from error

        try:
            return yaml.safe_load(file_content)
        except yaml.YAMLError as error:
            raise ValueError(f"Invalid YAML configuration file: {config_path}") from error

    @staticmethod
    def _require_mapping(value: object, *, context: str) -> dict[str, object]:
        """Require a YAML mapping with string keys."""
        if not isinstance(value, Mapping):
            raise TypeError(f"{context} must be a YAML mapping.")

        raw_mapping = cast(Mapping[object, object], value)
        result: dict[str, object] = {}

        for key, item in raw_mapping.items():
            if not isinstance(key, str):
                raise TypeError(f"{context} keys must be strings.")

            result[key] = item

        return result

    def _require_section(
        self,
        root: Mapping[str, object],
        section_name: str,
    ) -> dict[str, object]:
        """Read a required top-level configuration section."""
        if section_name not in root:
            raise ValueError(f"Missing required configuration section: {section_name}")

        return self._require_mapping(
            root[section_name],
            context=f"Configuration section '{section_name}'",
        )

    @staticmethod
    def _validate_keys(
        section: Mapping[str, object],
        *,
        expected_keys: frozenset[str],
        context: str,
    ) -> None:
        """Validate strict configuration keys for one mapping."""
        actual_keys = set(section)
        missing_keys = expected_keys - actual_keys
        unknown_keys = actual_keys - expected_keys

        if missing_keys:
            formatted_keys = ", ".join(sorted(missing_keys))
            raise ValueError(f"{context} is missing required keys: {formatted_keys}")

        if unknown_keys:
            formatted_keys = ", ".join(sorted(unknown_keys))
            raise ValueError(f"{context} contains unknown keys: {formatted_keys}")

    def _build_dataset_config(
        self,
        section: Mapping[str, object],
    ) -> DatasetConfig:
        """Build the validated dataset configuration section."""
        self._validate_keys(
            section,
            expected_keys=self._DATASET_KEYS,
            context="Dataset configuration",
        )

        name = self._require_string(section, "name", context="Dataset configuration")
        root_dir_value = self._require_string(
            section,
            "root_dir",
            context="Dataset configuration",
        )

        return DatasetConfig(
            name=name,
            root_dir=self._resolve_project_path(root_dir_value),
        )

    def _build_inspection_config(
        self,
        section: Mapping[str, object],
    ) -> InspectionConfig:
        """Build the validated inspection configuration section."""
        self._validate_keys(
            section,
            expected_keys=self._INSPECTION_KEYS,
            context="Inspection configuration",
        )

        output_dir_value = self._require_string(
            section,
            "output_dir",
            context="Inspection configuration",
        )

        return InspectionConfig(
            output_dir=self._resolve_project_path(output_dir_value),
            report_filename=self._require_string(
                section,
                "report_filename",
                context="Inspection configuration",
            ),
            summary_filename=self._require_string(
                section,
                "summary_filename",
                context="Inspection configuration",
            ),
            visualizations_dirname=self._require_string(
                section,
                "visualizations_dirname",
                context="Inspection configuration",
            ),
        )

    def _build_validation_config(
        self,
        section: Mapping[str, object],
    ) -> ValidationConfig:
        """Build the validated metadata validation configuration section."""
        self._validate_keys(
            section,
            expected_keys=self._VALIDATION_KEYS,
            context="Validation configuration",
        )

        return ValidationConfig(
            expected_labels=self._require_integer_tuple(
                section,
                "expected_labels",
                context="Validation configuration",
            ),
            affine_absolute_tolerance=self._require_float(
                section,
                "affine_absolute_tolerance",
                context="Validation configuration",
            ),
            require_finite_intensities=self._require_bool(
                section,
                "require_finite_intensities",
                context="Validation configuration",
            ),
            require_positive_voxel_spacing=self._require_bool(
                section,
                "require_positive_voxel_spacing",
                context="Validation configuration",
            ),
        )

    def _build_preprocessing_config(
        self,
        section: Mapping[str, object],
    ) -> PreprocessingConfig:
        """Build the validated preprocessing configuration section."""
        self._validate_keys(
            section,
            expected_keys=self._PREPROCESSING_KEYS,
            context="Preprocessing configuration",
        )

        return PreprocessingConfig(
            target_spacing_mm=self._require_float_triplet(
                section,
                "target_spacing_mm",
                context="Preprocessing configuration",
            ),
            target_shape=self._require_integer_triplet(
                section,
                "target_shape",
                context="Preprocessing configuration",
            ),
            intensity_lower_percentile=self._require_float(
                section,
                "intensity_lower_percentile",
                context="Preprocessing configuration",
            ),
            intensity_upper_percentile=self._require_float(
                section,
                "intensity_upper_percentile",
                context="Preprocessing configuration",
            ),
            normalize_nonzero_only=self._require_bool(
                section,
                "normalize_nonzero_only",
                context="Preprocessing configuration",
            ),
        )

    def _resolve_project_path(self, path_value: str) -> Path:
        """Resolve a configured path relative to the project root."""
        candidate = Path(path_value).expanduser()

        if not candidate.is_absolute():
            candidate = self._project_root / candidate

        return candidate.resolve(strict=False)

    @staticmethod
    def _require_string(
        section: Mapping[str, object],
        key: str,
        *,
        context: str,
    ) -> str:
        """Require a non-empty string value from a configuration section."""
        value = section[key]

        if not isinstance(value, str):
            raise TypeError(f"{context} key '{key}' must be a string.")

        normalized_value = value.strip()

        if not normalized_value:
            raise ValueError(f"{context} key '{key}' must not be empty.")

        return normalized_value

    @staticmethod
    def _require_bool(
        section: Mapping[str, object],
        key: str,
        *,
        context: str,
    ) -> bool:
        """Require a boolean value from a configuration section."""
        value = section[key]

        if not isinstance(value, bool):
            raise TypeError(f"{context} key '{key}' must be a boolean.")

        return value

    @staticmethod
    def _require_float(
        section: Mapping[str, object],
        key: str,
        *,
        context: str,
    ) -> float:
        """Require a numeric value and return it as a float."""
        value = section[key]

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{context} key '{key}' must be a number.")

        return float(value)

    @staticmethod
    def _require_integer_tuple(
        section: Mapping[str, object],
        key: str,
        *,
        context: str,
    ) -> tuple[int, ...]:
        """Require an arbitrary-length YAML list of integer values."""
        value = section[key]

        if not isinstance(value, list):
            raise TypeError(f"{context} key '{key}' must be a YAML list.")

        result: list[int] = []

        for index, item in enumerate(value):
            if isinstance(item, bool) or not isinstance(item, int):
                raise TypeError(
                    f"{context} key '{key}' item at index {index} must be an integer."
                )

            result.append(item)

        return tuple(result)

    @staticmethod
    def _require_float_triplet(
        section: Mapping[str, object],
        key: str,
        *,
        context: str,
    ) -> tuple[float, float, float]:
        """Require an exact three-item YAML list of numeric values."""
        value = section[key]

        if not isinstance(value, list):
            raise TypeError(f"{context} key '{key}' must be a YAML list.")

        if len(value) != _SPATIAL_DIMENSION_COUNT:
            raise ValueError(
                f"{context} key '{key}' must contain exactly three values."
            )

        result: list[float] = []

        for index, item in enumerate(value):
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise TypeError(
                    f"{context} key '{key}' item at index {index} "
                    "must be a number."
                )

            result.append(float(item))

        return (
            result[0],
            result[1],
            result[2],
        )

    @staticmethod
    def _require_integer_triplet(
        section: Mapping[str, object],
        key: str,
        *,
        context: str,
    ) -> tuple[int, int, int]:
        """Require an exact three-item YAML list of integer values."""
        value = section[key]

        if not isinstance(value, list):
            raise TypeError(f"{context} key '{key}' must be a YAML list.")

        if len(value) != _SPATIAL_DIMENSION_COUNT:
            raise ValueError(
                f"{context} key '{key}' must contain exactly three values."
            )

        result: list[int] = []

        for index, item in enumerate(value):
            if isinstance(item, bool) or not isinstance(item, int):
                raise TypeError(
                    f"{context} key '{key}' item at index {index} "
                    "must be an integer."
                )

            result.append(item)

        return (
            result[0],
            result[1],
            result[2],
        )
