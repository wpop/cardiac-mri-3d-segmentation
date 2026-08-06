from pathlib import Path
from typing import Final

from cardiac_segmentation.config.loader import AppConfigLoader

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_CONFIG_PATH: Final[Path] = Path("configs/data.yaml")


def test_load_real_preprocessing_configuration() -> None:
    """Load preprocessing parameters from the real application YAML file."""
    config = AppConfigLoader(
        project_root=_PROJECT_ROOT
    ).load(_CONFIG_PATH)

    assert config.preprocessing.target_spacing_mm == (1.5, 1.5, 5.0)
    assert config.preprocessing.target_shape == (192, 192, 24)
    assert config.preprocessing.intensity_lower_percentile == 1.0
    assert config.preprocessing.intensity_upper_percentile == 99.0
    assert config.preprocessing.normalize_nonzero_only is True
