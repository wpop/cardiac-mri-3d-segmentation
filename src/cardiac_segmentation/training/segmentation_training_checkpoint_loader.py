from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, cast

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from cardiac_segmentation.training.segmentation_training_checkpoint import (
    SegmentationTrainingCheckpoint,
)


class SegmentationTrainingCheckpointLoader:
    """Load a checkpoint into a model and optimizer with strict validation."""

    _CHECKPOINT_KEYS: Final[frozenset[str]] = frozenset(
        {
            "format_version",
            "epoch_number",
            "model_state_dict",
            "optimizer_state_dict",
            "training_average_loss",
            "validation_average_loss",
            "training_mean_dice",
            "validation_mean_dice",
            "included_class_indices",
            "validation_per_class_dice",
        }
    )

    def load_into(
        self,
        checkpoint_path: Path,
        model: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
    ) -> SegmentationTrainingCheckpoint:
        """Load checkpoint state into the supplied model and optimizer."""
        resolved_checkpoint_path = checkpoint_path.expanduser().resolve(strict=False)

        if not resolved_checkpoint_path.is_file():
            raise FileNotFoundError(
                "Checkpoint path must exist and be a regular file: "
                f"{resolved_checkpoint_path}"
            )

        checkpoint_payload = self._load_checkpoint_payload(
            checkpoint_path=resolved_checkpoint_path,
            device=device,
        )
        model_state_dict = self._require_mapping(
            checkpoint_payload["model_state_dict"],
            key="model_state_dict",
        )
        optimizer_state_dict = self._require_mapping(
            checkpoint_payload["optimizer_state_dict"],
            key="optimizer_state_dict",
        )
        model.load_state_dict(
            model_state_dict,
            strict=True,
        )
        optimizer.load_state_dict(dict(optimizer_state_dict))
        self._validate_model_parameters(
            model=model,
            device=device,
        )
        self._validate_optimizer_state(optimizer)

        return SegmentationTrainingCheckpoint(
            checkpoint_path=resolved_checkpoint_path,
            format_version=self._require_integer(checkpoint_payload, "format_version"),
            epoch_number=self._require_integer(checkpoint_payload, "epoch_number"),
            training_average_loss=self._require_float(
                checkpoint_payload,
                "training_average_loss",
            ),
            validation_average_loss=self._require_float(
                checkpoint_payload,
                "validation_average_loss",
            ),
            training_mean_dice=self._require_float(
                checkpoint_payload,
                "training_mean_dice",
            ),
            validation_mean_dice=self._require_float(
                checkpoint_payload,
                "validation_mean_dice",
            ),
            included_class_indices=self._require_integer_tuple(
                checkpoint_payload,
                "included_class_indices",
            ),
            validation_per_class_dice=self._require_float_tuple(
                checkpoint_payload,
                "validation_per_class_dice",
            ),
        )

    def _load_checkpoint_payload(
        self,
        checkpoint_path: Path,
        device: torch.device,
    ) -> dict[str, object]:
        """Load and validate the raw checkpoint mapping and its keys."""
        raw_checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        checkpoint_payload = self._require_mapping(
            raw_checkpoint,
            key="checkpoint root",
        )
        actual_keys = set(checkpoint_payload)
        missing_keys = self._CHECKPOINT_KEYS - actual_keys
        unknown_keys = actual_keys - self._CHECKPOINT_KEYS

        if missing_keys:
            formatted_keys = ", ".join(sorted(missing_keys))
            raise ValueError(f"Checkpoint is missing required keys: {formatted_keys}")

        if unknown_keys:
            formatted_keys = ", ".join(sorted(unknown_keys))
            raise ValueError(f"Checkpoint contains unknown keys: {formatted_keys}")

        return dict(checkpoint_payload)

    @staticmethod
    def _require_mapping(
        value: object,
        *,
        key: str,
    ) -> Mapping[str, Any]:
        """Require a mapping with string keys."""
        if not isinstance(value, Mapping):
            raise TypeError(f"Checkpoint key '{key}' must be a mapping.")

        mapping = cast(Mapping[object, Any], value)

        if any(not isinstance(mapping_key, str) for mapping_key in mapping):
            raise TypeError(f"Checkpoint key '{key}' mapping keys must be strings.")

        return cast(Mapping[str, Any], mapping)

    @staticmethod
    def _require_integer(
        checkpoint_payload: Mapping[str, object],
        key: str,
    ) -> int:
        """Require an integer checkpoint metadata field."""
        value = checkpoint_payload[key]

        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"Checkpoint key '{key}' must be an integer.")

        return value

    @staticmethod
    def _require_float(
        checkpoint_payload: Mapping[str, object],
        key: str,
    ) -> float:
        """Require a numeric checkpoint metadata field."""
        value = checkpoint_payload[key]

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"Checkpoint key '{key}' must be a number.")

        return float(value)

    @staticmethod
    def _require_integer_tuple(
        checkpoint_payload: Mapping[str, object],
        key: str,
    ) -> tuple[int, ...]:
        """Require a tuple of integer checkpoint metadata values."""
        value = checkpoint_payload[key]

        if not isinstance(value, tuple):
            raise TypeError(f"Checkpoint key '{key}' must be a tuple.")

        if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
            raise TypeError(f"Checkpoint key '{key}' must contain only integers.")

        return value

    @staticmethod
    def _require_float_tuple(
        checkpoint_payload: Mapping[str, object],
        key: str,
    ) -> tuple[float, ...]:
        """Require a tuple of numeric checkpoint metadata values."""
        value = checkpoint_payload[key]

        if not isinstance(value, tuple):
            raise TypeError(f"Checkpoint key '{key}' must be a tuple.")

        result: list[float] = []

        for item in value:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise TypeError(f"Checkpoint key '{key}' must contain only numbers.")

            result.append(float(item))

        return tuple(result)

    @staticmethod
    def _validate_model_parameters(
        model: nn.Module,
        device: torch.device,
    ) -> None:
        """Verify model parameters are on the device and finite."""
        for parameter in model.parameters():
            if parameter.device != device:
                raise ValueError("Every model parameter must be on the configured device.")

            if not bool(torch.isfinite(parameter).all().item()):
                raise ValueError("Every model parameter must be finite.")

    @staticmethod
    def _validate_optimizer_state(
        optimizer: Optimizer,
    ) -> None:
        """Verify all optimizer tensor state values are finite."""
        for state in optimizer.state.values():
            for value in state.values():
                if isinstance(value, Tensor) and not bool(torch.isfinite(value).all().item()):
                    raise ValueError("Optimizer tensor state values must be finite.")
