from dataclasses import dataclass

from cardiac_segmentation.config.dataset_config import DatasetConfig
from cardiac_segmentation.config.inspection_config import InspectionConfig
from cardiac_segmentation.config.preprocessing_config import (
    PreprocessingConfig,
)
from cardiac_segmentation.config.validation_config import ValidationConfig


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Combine all validated application configuration sections."""

    dataset: DatasetConfig
    inspection: InspectionConfig
    validation: ValidationConfig
    preprocessing: PreprocessingConfig
