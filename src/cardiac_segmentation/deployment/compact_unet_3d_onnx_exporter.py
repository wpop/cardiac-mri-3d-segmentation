from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import onnx
import onnxruntime as ort  # type: ignore[import-untyped]
import torch
from numpy.typing import NDArray
from onnx import TensorProto, ValueInfoProto
from torch import Tensor

from cardiac_segmentation.models import CompactUNet3D

_IN_CHANNELS: Final[int] = 1
_NUM_CLASSES: Final[int] = 4
_BASE_CHANNELS: Final[int] = 8
_INPUT_SHAPE: Final[tuple[int, int, int, int, int]] = (1, 1, 24, 192, 192)
_OUTPUT_SHAPE: Final[tuple[int, int, int, int, int]] = (1, 4, 24, 192, 192)
_INPUT_NAME: Final[str] = "image"
_OUTPUT_NAME: Final[str] = "logits"
_MINIMUM_OPSET_VERSION: Final[int] = 18
_DEFAULT_OPSET_VERSION: Final[int] = 18
_PARITY_MAX_ABSOLUTE_TOLERANCE: Final[float] = 5e-3
_PARITY_MEAN_ABSOLUTE_TOLERANCE: Final[float] = 5e-4


class CompactUNet3DOnnxExporter:
    """Export final CompactUNet3D checkpoints and validate ONNX Runtime parity."""

    def __init__(
        self,
        opset_version: int = _DEFAULT_OPSET_VERSION,
    ) -> None:
        """Initialize the fixed deployment contract and ONNX opset."""
        self._validate_opset_version(opset_version)
        self._opset_version = opset_version

    @property
    def opset_version(self) -> int:
        """Return the configured ONNX opset version."""
        return self._opset_version

    @property
    def input_shape(self) -> tuple[int, int, int, int, int]:
        """Return the fixed deployment input shape."""
        return _INPUT_SHAPE

    @property
    def output_shape(self) -> tuple[int, int, int, int, int]:
        """Return the fixed deployment output shape."""
        return _OUTPUT_SHAPE

    @property
    def parity_tolerance(self) -> dict[str, float]:
        """Return the absolute-difference tolerances used for parity validation."""
        return {
            "maximum_absolute_difference": _PARITY_MAX_ABSOLUTE_TOLERANCE,
            "mean_absolute_difference": _PARITY_MEAN_ABSOLUTE_TOLERANCE,
        }

    def load_model(
        self,
        checkpoint_path: Path,
    ) -> tuple[CompactUNet3D, int]:
        """Construct CompactUNet3D, load checkpoint weights, and switch to eval."""
        checkpoint_payload = self._load_checkpoint_payload(checkpoint_path)
        model_state_dict = self._require_mapping(
            checkpoint_payload["model_state_dict"],
            key="model_state_dict",
        )
        model = CompactUNet3D(
            in_channels=_IN_CHANNELS,
            num_classes=_NUM_CLASSES,
            base_channels=_BASE_CHANNELS,
        )
        model.load_state_dict(
            model_state_dict,
            strict=True,
        )
        model.eval()

        return (
            model,
            self._require_integer(checkpoint_payload, "epoch_number"),
        )

    def export_and_validate(
        self,
        *,
        checkpoint_path: Path,
        onnx_path: Path,
    ) -> dict[str, object]:
        """Export a checkpoint to ONNX and return validation/parity metadata."""
        model, checkpoint_epoch = self.load_model(checkpoint_path)
        deterministic_input = self.create_deterministic_input()
        self.export(
            model=model,
            sample_input=deterministic_input,
            onnx_path=onnx_path,
        )

        return self.validate(
            model=model,
            sample_input=deterministic_input,
            onnx_path=onnx_path,
            checkpoint_epoch=checkpoint_epoch,
        )

    def export(
        self,
        *,
        model: CompactUNet3D,
        sample_input: Tensor,
        onnx_path: Path,
    ) -> None:
        """Export raw-logit CompactUNet3D inference to a fixed-shape ONNX file."""
        self._validate_sample_input(sample_input)
        onnx_path.parent.mkdir(parents=True, exist_ok=True)

        with torch.inference_mode():
            torch.onnx.export(
                model,
                (sample_input,),
                str(onnx_path),
                export_params=True,
                opset_version=self._opset_version,
                do_constant_folding=True,
                input_names=[_INPUT_NAME],
                output_names=[_OUTPUT_NAME],
                dynamic_axes=None,
                dynamo=False,
            )

    def validate(
        self,
        *,
        model: CompactUNet3D,
        sample_input: Tensor,
        onnx_path: Path,
        checkpoint_epoch: int,
    ) -> dict[str, object]:
        """Validate ONNX graph contract and PyTorch-vs-ONNX numerical parity."""
        self._validate_sample_input(sample_input)

        if not onnx_path.is_file():
            raise FileNotFoundError(f"ONNX export did not create a file: {onnx_path}")

        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        graph_input = onnx_model.graph.input[0]
        graph_output = onnx_model.graph.output[0]
        input_shape = self._tensor_value_shape(graph_input)
        output_shape = self._tensor_value_shape(graph_output)
        input_dtype = self._tensor_value_dtype(graph_input)
        output_dtype = self._tensor_value_dtype(graph_output)

        if graph_input.name != _INPUT_NAME:
            raise ValueError(f"ONNX input name must be {_INPUT_NAME}.")

        if graph_output.name != _OUTPUT_NAME:
            raise ValueError(f"ONNX output name must be {_OUTPUT_NAME}.")

        if input_shape != _INPUT_SHAPE:
            raise ValueError(f"ONNX input shape must be {_INPUT_SHAPE}, got {input_shape}.")

        if output_shape != _OUTPUT_SHAPE:
            raise ValueError(
                f"ONNX output shape must be {_OUTPUT_SHAPE}, got {output_shape}."
            )

        if input_dtype != "float32" or output_dtype != "float32":
            raise TypeError("ONNX input and output tensors must use float32.")

        session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        session_input = session.get_inputs()[0]
        session_output = session.get_outputs()[0]

        if session_input.name != _INPUT_NAME:
            raise ValueError("ONNX Runtime input name does not match the contract.")

        if session_output.name != _OUTPUT_NAME:
            raise ValueError("ONNX Runtime output name does not match the contract.")

        with torch.inference_mode():
            torch_output = model(sample_input).detach().cpu().numpy()

        onnx_output = self._run_onnx_session(
            session=session,
            sample_input=sample_input.detach().cpu().numpy(),
        )
        absolute_difference = np.abs(torch_output - onnx_output)
        maximum_absolute_difference = float(np.max(absolute_difference))
        mean_absolute_difference = float(np.mean(absolute_difference))
        parity_passed = (
            maximum_absolute_difference <= _PARITY_MAX_ABSOLUTE_TOLERANCE
            and mean_absolute_difference <= _PARITY_MEAN_ABSOLUTE_TOLERANCE
        )

        return {
            "checkpoint_epoch": checkpoint_epoch,
            "opset_version": self._opset_version,
            "onnx_path": str(onnx_path),
            "onnx_file_size_bytes": onnx_path.stat().st_size,
            "input_name": graph_input.name,
            "input_dtype": input_dtype,
            "input_shape": input_shape,
            "output_name": graph_output.name,
            "output_dtype": output_dtype,
            "output_shape": output_shape,
            "pytorch_output_shape": tuple(int(value) for value in torch_output.shape),
            "onnx_output_shape": tuple(int(value) for value in onnx_output.shape),
            "maximum_absolute_difference": maximum_absolute_difference,
            "mean_absolute_difference": mean_absolute_difference,
            "parity_tolerance": self.parity_tolerance,
            "parity_passed": parity_passed,
            "onnx_graph_valid": True,
            "onnxruntime_session_created": True,
        }

    def create_deterministic_input(self) -> Tensor:
        """Create a deterministic float32 input tensor for parity validation."""
        value_count = int(np.prod(np.asarray(_INPUT_SHAPE, dtype=np.int64)))
        input_tensor = torch.linspace(
            -1.0,
            1.0,
            steps=value_count,
            dtype=torch.float32,
        ).reshape(_INPUT_SHAPE)

        return input_tensor.contiguous()

    @staticmethod
    def _run_onnx_session(
        *,
        session: ort.InferenceSession,
        sample_input: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Run ONNX Runtime and return the single logits output."""
        outputs = session.run(
            [_OUTPUT_NAME],
            {_INPUT_NAME: sample_input},
        )

        if len(outputs) != 1:
            raise RuntimeError("ONNX Runtime should return exactly one output.")

        return cast(NDArray[np.float32], outputs[0])

    @staticmethod
    def _load_checkpoint_payload(
        checkpoint_path: Path,
    ) -> dict[str, object]:
        """Load the raw checkpoint mapping from disk."""
        resolved_checkpoint_path = checkpoint_path.expanduser().resolve(strict=False)

        if not resolved_checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint path must exist: {resolved_checkpoint_path}"
            )

        raw_checkpoint = torch.load(
            resolved_checkpoint_path,
            map_location=torch.device("cpu"),
            weights_only=False,
        )

        if not isinstance(raw_checkpoint, Mapping):
            raise TypeError("Checkpoint root must be a mapping.")

        return dict(cast(Mapping[str, object], raw_checkpoint))

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
    def _tensor_value_shape(
        value_info: ValueInfoProto,
    ) -> tuple[int, ...]:
        """Read a fixed tensor shape from an ONNX value info object."""
        dimensions = value_info.type.tensor_type.shape.dim

        return tuple(int(dimension.dim_value) for dimension in dimensions)

    @staticmethod
    def _tensor_value_dtype(
        value_info: ValueInfoProto,
    ) -> str:
        """Read a tensor dtype from an ONNX value info object."""
        element_type = int(value_info.type.tensor_type.elem_type)

        if element_type == TensorProto.FLOAT:
            return "float32"

        return f"onnx_tensor_type_{element_type}"

    @staticmethod
    def _validate_sample_input(
        sample_input: Tensor,
    ) -> None:
        """Validate the deterministic deployment sample input."""
        if tuple(int(value) for value in sample_input.shape) != _INPUT_SHAPE:
            raise ValueError(f"Sample input shape must be {_INPUT_SHAPE}.")

        if sample_input.dtype != torch.float32:
            raise TypeError("Sample input must use torch.float32 dtype.")

        if not sample_input.is_contiguous():
            raise ValueError("Sample input must be contiguous.")

    @staticmethod
    def _validate_opset_version(
        opset_version: int,
    ) -> None:
        """Validate the ONNX opset version."""
        if isinstance(opset_version, bool) or not isinstance(opset_version, int):
            raise TypeError("ONNX opset version must be an integer.")

        if opset_version < _MINIMUM_OPSET_VERSION:
            raise ValueError("ONNX opset version must be at least 18.")
