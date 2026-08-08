from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Final

from cardiac_segmentation.deployment import CompactUNet3DOnnxExporter

_CHECKPOINT_PATH: Final[Path] = Path("artifacts/checkpoints/final_patient_level_training.pt")
_ONNX_PATH: Final[Path] = Path("artifacts/models/cardiac_segmentation.onnx")


def main() -> None:
    """Export final CompactUNet3D checkpoint to ONNX and print parity metadata."""
    result = CompactUNet3DOnnxExporter().export_and_validate(
        checkpoint_path=_CHECKPOINT_PATH,
        onnx_path=_ONNX_PATH,
    )
    sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
