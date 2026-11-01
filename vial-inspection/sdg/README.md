# Vial synthetic-data generation

This component adds read-only stage discovery and, next, session-layer defect
overrides around the supplied digital twin. It does not edit the packaged USDC.

## Discover stage structure

Run with Isaac Sim's Python:

```bash
./python.sh /workspace/vial-inspection/sdg/discover_stage.py \
  --stage /workspace/vial-inspection/csi_digital_twin_v1/stations/tray_inspection/tray_inspection.usdc \
  --output /workspace/vial-inspection/sdg/stage_inventory.json
```

The generated inventory is an audit artifact and is ignored when it is produced
under a temporary/output directory. Review it before binding generator logic to
prim paths.

## Generate the deterministic proof scene

```bash
./python.sh /workspace/vial-inspection/sdg/generate_proof_scene.py \
  --stage /workspace/vial-inspection/csi_digital_twin_v1/stations/tray_inspection/tray_inspection.usdc \
  --camera-config /workspace/vial-inspection/csi_digital_twin_v1/cameras/CapSealCam.json \
  --apply-lens /workspace/vial-inspection/csi_digital_twin_v1/tools/apply_lens.py \
  --output /workspace/vial-inspection/sdg/output/proof_scene
```

The proof scene places `pass`, `missing_cap`, `crooked_cap`, `damaged_vial`, and
`no_vial` in G01-G05, empties the scrap fixture, renders through CapSealCam, adds
the supplied lens distortion, and exports 60 fixed-position crops and labels.

## Generate a seeded dataset

`generate_dataset.py` creates scene-level train/validation splits. Every good
tray contains exactly six examples of each class, while all 30 scrap positions
are labeled `no_vial`. Defect severity, brightness, sensor noise, and a bounded
image-space camera shift are derived from each scene seed and recorded in its
metadata.

```bash
./python.sh /workspace/vial-inspection/sdg/generate_dataset.py \
  --stage /workspace/vial-inspection/csi_digital_twin_v1/stations/tray_inspection/tray_inspection.usdc \
  --camera-config /workspace/vial-inspection/csi_digital_twin_v1/cameras/CapSealCam.json \
  --apply-lens /workspace/vial-inspection/csi_digital_twin_v1/tools/apply_lens.py \
  --output /workspace/vial-inspection/sdg/output/dataset-v1 \
  --scenes 50 --seed 2026 --val-fraction 0.2
```

Splitting is performed by scene, never by crop, to prevent near-identical crops
from the same render leaking between training and validation. Generated output
is excluded by the repository `.gitignore`.

## Export crops for classifier training

After generation completes, validate the dataset and create a standard
`split/class/image.png` ImageFolder layout:

```bash
python export_dataset.py \
  --dataset-root output/dataset-v1 \
  --output output/classification-v1 \
  --areas good
```

The default `good` policy exports only G01-G30, which are balanced at six
examples per class per scene. Use `--areas all` only when the training plan
intentionally includes the scrap tray's 30 additional `no_vial` positions.
The exporter verifies every scene-label hash and records each crop hash in
`manifest.csv`.

## Train the first baseline

The dependency-light baseline uses 64-pixel grayscale HOG features and an
OpenCV linear multiclass SVM. It writes a model plus accuracy, macro-F1,
per-class scores, and a confusion matrix:

```bash
python train_baseline.py \
  --dataset-root output/classification-v1 \
  --output output/baseline-v1 \
  --seed 2026
```

This baseline is an early data/pipeline diagnostic. The intended production
model remains a transfer-learned image classifier evaluated against real,
held-out inspection images.
