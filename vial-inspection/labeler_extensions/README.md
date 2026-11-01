# Vial labeler extensions

This folder extends the supplied participant labeler without modifying it.

## Expanded calibration

`calibration/calibration.json` contains the two original profiles plus profiles
for the later dark/upper and bright/lower camera regimes. Reference images live
beside it in `calibration/examples/`, as required by the original labeler.

Run the original tool with the external calibration:

```powershell
python ..\vial-labeler-participants-20261014\vial-labeler-participants-v1\labeler.py run `
  --images "..\..\project\tray-inspection-detection-dataset\images\Testing" `
  --files tray_000250.jpg `
  --calibration calibration\calibration.json `
  --output review-250
```

Automatic alignment remains fail-closed. Visually inspect all target rings in
every generated review before confirming labels. The added grids were calibrated
for the supplied 1920 x 1080 fixed-camera images, not arbitrary camera views.

## Training export

`export_training.py` converts an imported, fully confirmed review into either:

- `crops`: position-classification crops arranged by split and class.
- `yolo`: full images plus one normalized box per reviewed fixture position.
- `both`: both representations.

The canonical position classes are:

1. `pass`
2. `missing_cap`
3. `crooked_cap`
4. `damaged_vial`
5. `no_vial`

The exporter deliberately does not mix fixture-level legacy classes such as
`complete_order` and `incomplete_order` into position labels. See
`class_mapping.json` for the explicit mapping policy.

```powershell
python export_training.py `
  --review review-250-confirmed\labels.json `
  --split-root "..\..\project\tray-inspection-detection-dataset\images" `
  --output training-export `
  --format both
```

Safety checks reject partial reviews, changed source images, unknown classes,
invalid geometry, duplicate images/slots, and output-directory reuse.

