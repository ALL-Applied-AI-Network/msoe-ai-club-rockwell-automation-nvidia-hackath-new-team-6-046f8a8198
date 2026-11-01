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
