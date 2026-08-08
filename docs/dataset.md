# ACDC Dataset Notes

The project expects the ACDC dataset under `data/raw/acdc/`. The code indexes
ACDC training patients, reads patient metadata, and creates deterministic
patient-level training and validation splits over ED/ES volumes.

The segmentation labels are:

| Label | Class |
| --- | --- |
| 0 | background |
| 1 | RV cavity |
| 2 | myocardium |
| 3 | LV cavity |

Preprocessing is configured in `configs/data.yaml`:

- target spacing: `(1.5, 1.5, 5.0)` mm
- target shape: `(192, 192, 24)` in NIfTI spatial order
- intensity clipping: 1st to 99th percentile
- nonzero-only intensity normalization
- geometry validation for labels, affine consistency, finite intensities, and
  positive voxel spacing

Generated dataset inspection reports are written under
`artifacts/dataset_inspection/` and are not committed.
