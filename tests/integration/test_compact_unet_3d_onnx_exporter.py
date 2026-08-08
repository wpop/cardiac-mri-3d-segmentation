from pathlib import Path
from typing import Final

import onnx
import onnxruntime as ort

from cardiac_segmentation.deployment import CompactUNet3DOnnxExporter

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_FINAL_CHECKPOINT_PATH: Final[Path] = (
    _PROJECT_ROOT / "artifacts/checkpoints/final_patient_level_training.pt"
)
_EXPECTED_INPUT_SHAPE: Final[tuple[int, int, int, int, int]] = (1, 1, 24, 192, 192)
_EXPECTED_OUTPUT_SHAPE: Final[tuple[int, int, int, int, int]] = (1, 4, 24, 192, 192)


def test_compact_unet_3d_onnx_exporter_loads_final_checkpoint() -> None:
    """Verify the final checkpoint loads into the deployment model."""
    model, checkpoint_epoch = CompactUNet3DOnnxExporter().load_model(
        _FINAL_CHECKPOINT_PATH
    )

    assert checkpoint_epoch == 48
    assert not model.training


def test_compact_unet_3d_onnx_exporter_exports_valid_contract_and_parity(
    tmp_path: Path,
) -> None:
    """Export ONNX and verify graph, runtime session, shapes, names, and parity."""
    onnx_path = tmp_path / "cardiac_segmentation.onnx"
    result = CompactUNet3DOnnxExporter().export_and_validate(
        checkpoint_path=_FINAL_CHECKPOINT_PATH,
        onnx_path=onnx_path,
    )

    assert onnx_path.is_file()
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    assert session.get_inputs()[0].name == "image"
    assert session.get_outputs()[0].name == "logits"
    assert result["input_name"] == "image"
    assert result["output_name"] == "logits"
    assert result["input_shape"] == _EXPECTED_INPUT_SHAPE
    assert result["output_shape"] == _EXPECTED_OUTPUT_SHAPE
    assert result["pytorch_output_shape"] == _EXPECTED_OUTPUT_SHAPE
    assert result["onnx_output_shape"] == _EXPECTED_OUTPUT_SHAPE
    assert result["parity_passed"] is True
    assert result["onnx_graph_valid"] is True
    assert result["onnxruntime_session_created"] is True
