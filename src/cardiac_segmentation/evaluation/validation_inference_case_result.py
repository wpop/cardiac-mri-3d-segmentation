from dataclasses import dataclass
from math import isfinite, isinf


@dataclass(frozen=True, slots=True)
class ValidationInferenceCaseResult:
    """Store validated per-volume validation inference Dice values."""

    patient_id: str
    volume_id: str
    rv_dice: float
    myocardium_dice: float
    lv_dice: float
    mean_foreground_dice: float
    rv_hd95_mm: float
    myocardium_hd95_mm: float
    lv_hd95_mm: float
    mean_foreground_hd95_mm: float

    def __post_init__(self) -> None:
        """Validate identifiers, normalized Dice values, and HD95 distances."""
        if not self.patient_id:
            raise ValueError("Patient identifier must not be empty.")

        if not self.volume_id:
            raise ValueError("Volume identifier must not be empty.")

        self._validate_dice(self.rv_dice, name="RV Dice")
        self._validate_dice(self.myocardium_dice, name="Myocardium Dice")
        self._validate_dice(self.lv_dice, name="LV Dice")
        self._validate_dice(
            self.mean_foreground_dice,
            name="Mean foreground Dice",
        )
        self._validate_hd95(self.rv_hd95_mm, name="RV HD95")
        self._validate_hd95(self.myocardium_hd95_mm, name="Myocardium HD95")
        self._validate_hd95(self.lv_hd95_mm, name="LV HD95")
        self._validate_hd95(
            self.mean_foreground_hd95_mm,
            name="Mean foreground HD95",
        )

    @staticmethod
    def _validate_dice(
        value: float,
        *,
        name: str,
    ) -> None:
        """Validate a finite Dice value inside [0, 1]."""
        if not isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and inside [0.0, 1.0].")

    @staticmethod
    def _validate_hd95(
        value: float,
        *,
        name: str,
    ) -> None:
        """Validate a non-negative HD95 value in millimeters or infinity."""
        if not (isfinite(value) or isinf(value)) or value < 0.0:
            raise ValueError(f"{name} must be non-negative or infinity.")
