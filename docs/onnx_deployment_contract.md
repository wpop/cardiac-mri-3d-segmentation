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
