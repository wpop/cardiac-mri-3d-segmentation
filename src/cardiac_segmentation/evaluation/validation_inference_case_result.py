from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class ValidationInferenceCaseResult:
    """Store validated per-volume validation inference Dice values."""

    patient_id: str
    volume_id: str
    rv_dice: float
    myocardium_dice: float
    lv_dice: float
    mean_foreground_dice: float

    def __post_init__(self) -> None:
        """Validate identifiers and normalized Dice values."""
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

    @staticmethod
    def _validate_dice(
        value: float,
        *,
        name: str,
    ) -> None:
        """Validate a finite Dice value inside [0, 1]."""
        if not isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and inside [0.0, 1.0].")
