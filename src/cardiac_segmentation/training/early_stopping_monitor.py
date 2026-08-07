from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any


@dataclass(slots=True)
class EarlyStoppingMonitor:
    """Track validation improvements and decide when training should stop."""

    patience: int
    minimum_improvement: float
    best_metric: float | None = None
    epochs_without_improvement: int = 0

    def __post_init__(self) -> None:
        """Validate early-stopping parameters and restored state."""
        if (
            isinstance(self.patience, bool)
            or not isinstance(self.patience, int)
            or self.patience <= 0
        ):
            raise ValueError("Early-stopping patience must be a positive integer.")

        if not isfinite(self.minimum_improvement) or self.minimum_improvement < 0.0:
            raise ValueError(
                "Early-stopping minimum improvement must be finite and non-negative."
            )

        if self.best_metric is not None and not self._is_valid_metric(self.best_metric):
            raise ValueError("Best early-stopping metric must be finite.")

        if (
            isinstance(self.epochs_without_improvement, bool)
            or not isinstance(self.epochs_without_improvement, int)
            or self.epochs_without_improvement < 0
        ):
            raise ValueError(
                "Epochs without improvement must be a non-negative integer."
            )

    def step(
        self,
        metric: float,
    ) -> bool:
        """Update the monitor and return whether patience has been exhausted."""
        if not self._is_valid_metric(metric):
            raise ValueError("Early-stopping metric must be finite.")

        if self.best_metric is None or metric > (
            self.best_metric + self.minimum_improvement
        ):
            self.best_metric = metric
            self.epochs_without_improvement = 0
            return False

        self.epochs_without_improvement += 1
        return self.epochs_without_improvement >= self.patience

    def state_dict(self) -> dict[str, int | float | None]:
        """Return serializable early-stopping state."""
        return {
            "patience": self.patience,
            "minimum_improvement": self.minimum_improvement,
            "best_metric": self.best_metric,
            "epochs_without_improvement": self.epochs_without_improvement,
        }

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
    ) -> None:
        """Restore early-stopping state from a checkpoint mapping."""
        required_keys = {
            "patience",
            "minimum_improvement",
            "best_metric",
            "epochs_without_improvement",
        }
        actual_keys = set(state_dict)

        if actual_keys != required_keys:
            missing_keys = required_keys - actual_keys
            unknown_keys = actual_keys - required_keys
            raise ValueError(
                "Early-stopping state keys must match exactly; "
                f"missing={tuple(sorted(missing_keys))}, "
                f"unknown={tuple(sorted(unknown_keys))}."
            )

        patience = state_dict["patience"]
        minimum_improvement = state_dict["minimum_improvement"]
        best_metric = state_dict["best_metric"]
        epochs_without_improvement = state_dict["epochs_without_improvement"]

        if isinstance(patience, bool) or not isinstance(patience, int):
            raise TypeError("Early-stopping state patience must be an integer.")

        if isinstance(minimum_improvement, bool) or not isinstance(
            minimum_improvement,
            (int, float),
        ):
            raise TypeError(
                "Early-stopping state minimum improvement must be numeric."
            )

        if best_metric is not None and (
            isinstance(best_metric, bool) or not isinstance(best_metric, (int, float))
        ):
            raise TypeError("Early-stopping state best metric must be numeric or None.")

        if isinstance(epochs_without_improvement, bool) or not isinstance(
            epochs_without_improvement,
            int,
        ):
            raise TypeError(
                "Early-stopping state epochs without improvement must be an integer."
            )

        self.patience = patience
        self.minimum_improvement = float(minimum_improvement)
        self.best_metric = None if best_metric is None else float(best_metric)
        self.epochs_without_improvement = epochs_without_improvement
        self.__post_init__()

    @staticmethod
    def _is_valid_metric(
        metric: float,
    ) -> bool:
        """Return whether a monitored metric is finite."""
        return isfinite(metric)
