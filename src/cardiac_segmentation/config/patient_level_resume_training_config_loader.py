from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Final

from cardiac_segmentation.config.loader import AppConfigLoader
from cardiac_segmentation.config.patient_level_resume_training_config import (
    PatientLevelResumeTrainingConfig,
)


class PatientLevelResumeTrainingConfigLoader:
    """Load and validate a patient-level resume training configuration."""

    _ROOT_KEYS: Final[frozenset[str]] = frozenset({"patient_level_resume_training"})
    _SECTION_KEYS: Final[frozenset[str]] = frozenset(
        {
            "patient_count",
            "validation_fraction",
            "final_epoch_number",
            "batch_size",
            "num_workers",
            "pin_memory",
            "random_seed",
            "base_channels",
            "learning_rate",
            "weight_decay",
            "device",
            "resume_checkpoint_path",
            "checkpoint_path",
        }
    )

    def __init__(
        self,
        project_root: Path,
    ) -> None:
        """Initialize the loader with an explicit project root directory."""
        resolved_project_root = project_root.expanduser().resolve()

        if not resolved_project_root.is_dir():
            raise NotADirectoryError(
                f"Project root directory does not exist: {resolved_project_root}"
            )

        self._project_root = resolved_project_root

    def load(
        self,
        config_path: Path,
    ) -> PatientLevelResumeTrainingConfig:
        """Load a YAML file and construct the resume training configuration."""
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
            context="Patient-level resume training configuration",
        )

        return PatientLevelResumeTrainingConfig(
            patient_count=self._require_integer(section, "patient_count"),
            validation_fraction=AppConfigLoader._require_float(
                section,
                "validation_fraction",
                context="Patient-level resume training configuration",
            ),
            final_epoch_number=self._require_integer(section, "final_epoch_number"),
            batch_size=self._require_integer(section, "batch_size"),
            num_workers=self._require_integer(section, "num_workers"),
            pin_memory=AppConfigLoader._require_bool(
                section,
                "pin_memory",
                context="Patient-level resume training configuration",
            ),
            random_seed=self._require_integer(section, "random_seed"),
            base_channels=self._require_integer(section, "base_channels"),
            learning_rate=AppConfigLoader._require_float(
                section,
                "learning_rate",
                context="Patient-level resume training configuration",
            ),
            weight_decay=AppConfigLoader._require_float(
                section,
                "weight_decay",
                context="Patient-level resume training configuration",
            ),
            device=AppConfigLoader._require_string(
                section,
                "device",
                context="Patient-level resume training configuration",
            ),
            resume_checkpoint_path=self._resolve_project_path(
                AppConfigLoader._require_string(
                    section,
                    "resume_checkpoint_path",
                    context="Patient-level resume training configuration",
                )
            ),
            checkpoint_path=self._resolve_project_path(
                AppConfigLoader._require_string(
                    section,
                    "checkpoint_path",
                    context="Patient-level resume training configuration",
                )
            ),
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
        """Read the required resume training section."""
        if "patient_level_resume_training" not in root:
            raise ValueError(
                "Missing required configuration section: patient_level_resume_training"
            )

        return AppConfigLoader._require_mapping(
            root["patient_level_resume_training"],
            context="Configuration section 'patient_level_resume_training'",
        )

    @staticmethod
    def _require_integer(
        section: Mapping[str, object],
        key: str,
    ) -> int:
        """Require an integer value from the resume configuration section."""
        value = section[key]

        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                "Patient-level resume training configuration key "
                f"'{key}' must be an integer."
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
