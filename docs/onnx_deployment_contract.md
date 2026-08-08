# ONNX Deployment Contract

The exported ONNX model emits raw segmentation logits. It does not include
argmax or postprocessing.

Input:

- name: `image`
- dtype: `float32`
- shape: `[1, 1, 24, 192, 192]`

Output:

- name: `logits`
- dtype: `float32`
- shape: `[1, 4, 24, 192, 192]`

Class mapping:

- `0`: background
- `1`: RV cavity
- `2`: myocardium
- `3`: LV cavity

Export metadata:

- exporter: `scripts/export_onnx_model.py`
- checkpoint: `artifacts/checkpoints/final_patient_level_training.pt`
- checkpoint epoch: `48`
- ONNX path: `artifacts/models/cardiac_segmentation.onnx`
- opset: `18`
- model size: about `1.42 MB`

Validated parity:

- ONNX graph validation: passed
- ONNX Runtime session creation: passed
- PyTorch/ONNX parity: passed
- max absolute difference: `0.0022666454`
- mean absolute difference: `0.0001460407`
- max absolute tolerance: `0.005`
- mean absolute tolerance: `0.0005`

Inference consumers should apply `argmax` over the class channel of `logits`
outside the ONNX graph to obtain label IDs.
