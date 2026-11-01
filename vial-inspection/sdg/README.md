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
