from collections.abc import Sized
from math import isfinite
from typing import Final, cast

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from cardiac_segmentation.metrics import MulticlassDiceMetric3D
from cardiac_segmentation.training.segmentation_epoch_result import (
    SegmentationEpochResult,
)

_IMAGE_DIMENSION_COUNT: Final[int] = 5
_TARGET_DIMENSION_COUNT: Final[int] = 4


class SegmentationEpochRunner:
    """Execute one training or validation epoch for 3D segmentation."""

    def __init__(  # noqa: PLR0913
        self,
        model: nn.Module,
        loss_function: nn.Module,
        optimizer: Optimizer,
        num_classes: int,
        device: torch.device,
        include_background_in_dice: bool = False,
        dice_smooth: float = 1e-5,
    ) -> None:
        """Initialize model, loss, optimizer, device, and metric policy."""
        self._validate_num_classes(num_classes)
        self._validate_device(device)
        self._validate_include_background_in_dice(include_background_in_dice)
        self._validate_dice_smooth(dice_smooth)
        self._validate_model_parameters(
            model=model,
            device=device,
        )

        self._model = model
        self._loss_function = loss_function
        self._optimizer = optimizer
        self._num_classes = num_classes
        self._device = device
        self._include_background_in_dice = include_background_in_dice
        self._dice_smooth = dice_smooth

    def train_epoch(
        self,
        data_loader: DataLoader[dict[str, Tensor | str]],
    ) -> SegmentationEpochResult:
        """Run one optimization epoch and return aggregated results."""
        self._validate_data_loader(data_loader)
        self._model.train()
        metric = self._create_metric()
        total_loss = 0.0
        batch_count = 0
        volume_count = 0

        for batch in data_loader:
            image_tensor, target = self._prepare_batch(batch)
            self._optimizer.zero_grad(set_to_none=True)
            logits = cast(
                Tensor,
                self._model(image_tensor),
            )
            loss = cast(
                Tensor,
                self._loss_function(
                    logits,
                    target,
                ),
            )
            self._validate_loss(loss)
            torch.autograd.backward(loss)
            self._validate_finite_gradients()
            self._optimizer.step()
            metric.update(
                logits=logits.detach(),
                target=target.detach(),
            )
            batch_volume_count = int(image_tensor.shape[0])
            total_loss += float(loss.detach().item()) * batch_volume_count
            volume_count += batch_volume_count
            batch_count += 1

        return self._build_epoch_result(
            metric=metric,
            total_loss=total_loss,
            batch_count=batch_count,
            volume_count=volume_count,
        )

    def validate_epoch(
        self,
        data_loader: DataLoader[dict[str, Tensor | str]],
    ) -> SegmentationEpochResult:
        """Run one validation epoch and return aggregated results."""
        self._validate_data_loader(data_loader)
        self._model.eval()
        metric = self._create_metric()
        total_loss = 0.0
        batch_count = 0
        volume_count = 0

        with torch.inference_mode():
            for batch in data_loader:
                image_tensor, target = self._prepare_batch(batch)
                logits = cast(
                    Tensor,
                    self._model(image_tensor),
                )
                loss = cast(
                    Tensor,
                    self._loss_function(
                        logits,
                        target,
                    ),
                )
                self._validate_loss(loss)
                metric.update(
                    logits=logits,
                    target=target,
                )
                batch_volume_count = int(image_tensor.shape[0])
                total_loss += float(loss.item()) * batch_volume_count
                volume_count += batch_volume_count
                batch_count += 1

        return self._build_epoch_result(
            metric=metric,
            total_loss=total_loss,
            batch_count=batch_count,
            volume_count=volume_count,
        )

    def _prepare_batch(
        self,
        batch: dict[str, Tensor | str],
    ) -> tuple[Tensor, Tensor]:
        """Extract, validate, and move image and target tensors to the device."""
        image_tensor = self._require_tensor(
            batch.get("image"),
            name="Image",
        )
        target = self._require_tensor(
            batch.get("mask"),
            name="Target",
        )
        self._validate_batch_tensors(
            image_tensor=image_tensor,
            target=target,
        )
        image_tensor = image_tensor.to(
            self._device,
            non_blocking=image_tensor.is_pinned(),
        )
        target = target.to(
            self._device,
            non_blocking=target.is_pinned(),
        )

        return image_tensor, target

    def _validate_batch_tensors(
        self,
        image_tensor: Tensor,
        target: Tensor,
    ) -> None:
        """Validate image and target tensors before model forwarding."""
        if image_tensor.ndim != _IMAGE_DIMENSION_COUNT:
            raise ValueError("Image batch must be five-dimensional [B, C, D, H, W].")

        if target.ndim != _TARGET_DIMENSION_COUNT:
            raise ValueError("Target batch must be four-dimensional [B, D, H, W].")

        batch_size = int(image_tensor.shape[0])

        if batch_size <= 0:
            raise ValueError("Image batch dimension must be positive.")

        if int(target.shape[0]) != batch_size:
            raise ValueError("Image and target batch dimensions must match.")

        if tuple(image_tensor.shape[2:]) != tuple(target.shape[1:]):
            raise ValueError("Image and target spatial dimensions must match.")

        if int(image_tensor.shape[1]) <= 0:
            raise ValueError("Image channel count must be positive.")

        if not image_tensor.is_floating_point():
            raise TypeError("Image batch must use a floating-point dtype.")

        if target.dtype != torch.int64:
            raise TypeError("Target batch must have dtype torch.int64.")

        if not bool(torch.isfinite(image_tensor).all().item()):
            raise ValueError("Image batch contains non-finite values.")

        if not bool(torch.isfinite(target).all().item()):
            raise ValueError("Target batch contains non-finite values.")

    def _validate_loss(
        self,
        loss: Tensor,
    ) -> None:
        """Validate a scalar finite loss tensor."""
        if loss.ndim != 0:
            raise ValueError("Segmentation loss must be a scalar tensor.")

        if not bool(torch.isfinite(loss).item()):
            raise ValueError("Segmentation loss must be finite.")

    def _validate_finite_gradients(self) -> None:
        """Validate that every existing model gradient is finite."""
        for parameter in self._model.parameters():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item()):
                raise ValueError("Model gradients must contain only finite values.")

    def _build_epoch_result(
        self,
        metric: MulticlassDiceMetric3D,
        total_loss: float,
        batch_count: int,
        volume_count: int,
    ) -> SegmentationEpochResult:
        """Create a validated epoch result from accumulated totals."""
        if batch_count <= 0:
            raise ValueError("Epoch must contain at least one batch.")

        if volume_count <= 0:
            raise ValueError("Epoch must contain at least one volume.")

        return SegmentationEpochResult(
            average_loss=total_loss / volume_count,
            dice_result=metric.compute(),
            batch_count=batch_count,
            volume_count=volume_count,
        )

    def _create_metric(self) -> MulticlassDiceMetric3D:
        """Create a fresh Dice metric accumulator for one epoch."""
        return MulticlassDiceMetric3D(
            num_classes=self._num_classes,
            include_background=self._include_background_in_dice,
            smooth=self._dice_smooth,
        )

    @staticmethod
    def _require_tensor(
        value: Tensor | str | None,
        *,
        name: str,
    ) -> Tensor:
        """Return a collated batch field as a tensor."""
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} batch field must be a tensor.")

        return value

    @staticmethod
    def _validate_data_loader(
        data_loader: DataLoader[dict[str, Tensor | str]],
    ) -> None:
        """Reject empty DataLoaders."""
        dataset = data_loader.dataset

        if not isinstance(dataset, Sized):
            raise TypeError("DataLoader dataset must be sized.")

        if len(dataset) == 0:
            raise ValueError("DataLoader must not be empty.")

    @staticmethod
    def _validate_num_classes(
        num_classes: int,
    ) -> None:
        """Validate the number of segmentation classes."""
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes <= 1:
            raise ValueError("Number of classes must be an integer greater than one.")

    @staticmethod
    def _validate_device(
        device: torch.device,
    ) -> None:
        """Validate the configured execution device."""
        if not isinstance(device, torch.device):
            raise TypeError("Device must be a torch.device.")

    @staticmethod
    def _validate_include_background_in_dice(
        include_background_in_dice: bool,
    ) -> None:
        """Validate whether background contributes to Dice."""
        if not isinstance(include_background_in_dice, bool):
            raise TypeError("include_background_in_dice must be a boolean.")

    @staticmethod
    def _validate_dice_smooth(
        dice_smooth: float,
    ) -> None:
        """Validate the Dice smoothing value."""
        if not isfinite(dice_smooth) or dice_smooth <= 0.0:
            raise ValueError("Dice smooth value must be finite and strictly positive.")

    @staticmethod
    def _validate_model_parameters(
        model: nn.Module,
        device: torch.device,
    ) -> None:
        """Validate trainable parameters and model device placement."""
        parameters = tuple(model.parameters())
        trainable_parameters = tuple(
            parameter
            for parameter in parameters
            if parameter.requires_grad
        )

        if not trainable_parameters:
            raise ValueError("Model must expose at least one trainable parameter.")

        if any(parameter.device != device for parameter in parameters):
            raise ValueError("Every model parameter must already be on the configured device.")
