# Predicted reference annotations

These files contain complete 60-position predictions for four images chosen to
cover the supplied calibration regimes:

| Image | Split | Calibration profile |
| --- | --- | --- |
| `tray_000001.jpg` | Training | `view_001` |
| `tray_000122.jpg` | Training | `view_122` |
| `tray_000250.jpg` | Testing | `view_250_dark_upper` |
| `tray_000271.jpg` | Testing | `view_271_bright_lower` |

The modified participant labeler generated these annotations in unattended
prediction mode. Legacy YOLO defect boxes are mapped to calibrated fixture
positions. Unboxed good-tray positions use cap appearance, while scrap-tray
occupancy uses annotated column anchors and the fixed five-position fill order.

All 240 slots have a canonical class and `review_status: confirmed`. Each slot
also retains `prediction_confidence`, `prediction_source`, and an `auto:` note.
These are machine-generated reference predictions, not human-certified ground
truth. Do not use these four images to measure the predictor that created them.
Reserve separate real images for evaluation.

The `.json` files retain full geometry and provenance. The `.csv` files provide
a compact position-level view for analysis and training-data audits. Source
images remain in the ignored `project/tray-inspection-detection-dataset` folder.
