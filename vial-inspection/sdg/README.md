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

Pass multiple compatible roots to train a hybrid baseline without duplicating
the crop files:

```bash
python train_baseline.py \
  --dataset-root output/classification-v1 output/real-pseudo-v1 \
  --output output/hybrid-baseline-v1 --seed 2026
```

Use `--pass-weight` to tune the false-defect cost on a real validation split.
Choose the value using Validation only, then report Testing once.

For a direct false-alarm control, train a pass/defect gate followed by a defect
classifier:

```bash
python train_two_stage.py \
  --dataset-root output/classification-v1 output/real-pseudo-v1 \
  --output output/two-stage-v1 --seed 2026
```

Run `evaluate_real_dataset.py --model output/two-stage-v1 --tune-threshold`
on Validation, then pass the selected `--defect-threshold` to one Testing run.
The selected project profile is stored in `two_stage_profile.json`; use
`--threshold-profile two_stage_profile.json` for repeatable inference.

## Run auditable tray inference

`infer_tray.py` performs leakage-free inference without dataset YOLO labels. It
uses the two-stage model on G01-G30, the existing appearance/fill fallback for
S01-S30, and the supplied three-valued inspection rules for the tray verdict:

```bash
python infer_tray.py \
  --labeler ../vial-labeler-participants-20261014/vial-labeler-participants-v1/labeler.py \
  --images ../../project/tray-inspection-detection-dataset/images/Testing \
  --files tray_000012.jpg tray_000013.jpg \
  --calibration ../labeler_extensions/calibration/calibration.json \
  --model output/two-stage-v1 \
  --tray-model output/tray-classifier-v1 \
  --threshold-profile two_stage_profile.json \
  --output output/inference-demo-v1
```

The command exports `results.json`, a position-level CSV, and a tray summary
CSV. When `--tray-model` is present, both the learned tray verdict and the
conservative inspection-rule verdict are retained. Scrap routing is
intentionally left unresolved unless an upstream
position-level verdict is available; a visually passing scrap vial alone does
not prove a routing error.

Because a tray fails when any of 30 good positions is defective, calibrate a
separate tray threshold against Validation-level `complete_order` and
`incomplete_order` labels to audit whether score aggregation is viable:

```bash
python calibrate_tray_verdict.py \
  --model output/two-stage-v1 \
  --labeler ../vial-labeler-participants-20261014/vial-labeler-participants-v1/labeler.py \
  --images ../../project/tray-inspection-detection-dataset/images/Validation \
  --verdicts ../../project/tray-inspection-detection-dataset/tray_verdicts.csv \
  --calibration ../labeler_extensions/calibration/calibration.json \
  --output output/tray-threshold-v1.json --tune
```

For the current HOG/SVM model this audit fails: the Validation-selected maximum
score threshold has only 15.2% incomplete-order recall on Testing. Do not use
that experimental threshold for deployment. `two_stage_profile.json` remains
the supported position-level profile, and its tray verdict is conservative.

Train a genuine tray classifier from all 30 ordered gate scores, their sorted
distribution, summary statistics, defect-candidate counts, and calibration
profile. Candidate SVM configurations are selected on Validation; only the
chosen configuration is evaluated on Testing:

```bash
python train_tray_classifier.py \
  --position-model output/two-stage-v1 \
  --labeler ../vial-labeler-participants-20261014/vial-labeler-participants-v1/labeler.py \
  --dataset-root ../../project/tray-inspection-detection-dataset/images \
  --verdicts ../../project/tray-inspection-detection-dataset/tray_verdicts.csv \
  --calibration ../labeler_extensions/calibration/calibration.json \
  --output output/tray-classifier-v1 --feature-set distribution
```

`distribution` is the safer default: it excludes ordered slot identity and
calibration-profile identity to reduce acquisition-sequence leakage. Use
`--feature-set full` only as an explicitly labeled ablation.

## Check the real-image domain gap

`evaluate_real.py` applies the synthetic-trained model to calibrated crops from
the four supplied real-image regimes and records agreement by image, class, and
position. These annotations are machine-generated pseudo-labels, so agreement
is a consistency diagnostic rather than ground-truth accuracy:

```bash
python evaluate_real.py \
  --model output/baseline-v1/model.yml \
  --annotations ../reference_annotations \
  --images ../labeler_extensions/calibration/examples \
  --output output/real-agreement-v1.json \
  --areas good
```

## Build a balanced real pseudo-label dataset

`build_real_pseudo_dataset.py` runs the unattended labeler across a real image
split, filters low-confidence positions, excludes named holdout images, splits
by source image, and downsamples each class to a balanced ImageFolder dataset:

```bash
python build_real_pseudo_dataset.py \
  --labeler ../vial-labeler-participants-20261014/vial-labeler-participants-v1/labeler.py \
  --images ../../project/tray-inspection-detection-dataset/images/Training \
  --detections-root ../../project/tray-inspection-detection-dataset/labels \
  --calibration ../labeler_extensions/calibration/calibration.json \
  --output output/real-pseudo-v1 \
  --exclude tray_000001.jpg tray_000122.jpg tray_000250.jpg tray_000271.jpg \
  --min-confidence 0.90 --seed 2026
```

The output remains pseudo-labeled data, not certified ground truth. Balancing
and image-level splitting prevent the dominant pass class and neighboring crops
from making the validation result look artificially strong.

## Evaluate untouched real splits

`evaluate_real_dataset.py` runs the same calibrated mapping over an entire
unused Validation or Testing image directory and reports both overall and
macro-per-class agreement. Use the hybrid model and exclude any calibration
images used by earlier diagnostics:

```bash
python evaluate_real_dataset.py \
  --model output/hybrid-baseline-v1/model.yml \
  --labeler ../vial-labeler-participants-20261014/vial-labeler-participants-v1/labeler.py \
  --images ../../project/tray-inspection-detection-dataset/images/Testing \
  --detections-root ../../project/tray-inspection-detection-dataset/labels \
  --calibration ../labeler_extensions/calibration/calibration.json \
  --output output/testing-agreement-v1.json \
  --exclude tray_000250.jpg tray_000271.jpg --min-confidence 0.90
```
