# Fail-Safe Vial Inspection with Isaac Sim and Edge Inference

Rockwell Automation + NVIDIA Innovation Lab Hackathon · Fall 2026

This project turns an overhead image of a pharmaceutical vial fixture into an
auditable 60-position inspection and a fail-safe tray disposition. It combines
an NVIDIA Isaac Sim digital twin, synthetic and real-image training, calibrated
position mapping, a two-stage defect classifier, tray-level aggregation, and a
persistent inspection service deployed on an NVIDIA DGX Spark.

The system produces three operational outcomes:

- `accept`: a complete tray clears the confidence and multi-frame gates;
- `reject`: any frame is confidently incomplete;
- `hold`: evidence is insufficient or conflicting.

> This is a hackathon prototype, not a validated production quality system.
> Testing labels include machine-generated pseudo-labels, and independent
> physical-tray validation is still required.

## Challenge

Each fixture contains 30 expected good-tray positions and 30 scrap-tray
positions. The solution must locate every position, recognize pass, missing-cap,
crooked-cap, damaged-vial, and no-vial conditions, and decide whether the order
is complete without allowing uncertain trays to pass silently.

## Solution

```text
Isaac Sim digital twin ──> synthetic scenes and crops ─┐
                                                       ├─> two-stage position model
Real images ──> calibration ──> pseudo-label export ──┘
                                                             │
30 good-position score distribution ──> tray classifier ──> confidence gate
                                                             │
camera frames ──> tray-session consensus ──> accept / reject / hold
                                                             │
                                      JSON, CSV, overlays, HTML dashboard
```

Key design choices:

- Four calibration profiles cover the supplied camera/lighting regimes.
- The supplied labeler remains usable while prediction, calibration, and export
  capabilities live in separate project components.
- The position model uses a pass/defect gate followed by four-way defect
  classification.
- The tray classifier uses score distributions rather than ordered slot or
  camera-profile identity, reducing direct acquisition-sequence leakage.
- A Validation-selected confidence gate converts weak complete predictions to
  `unresolved`.
- Multi-frame confirmation never lowers the single-frame gate: any incomplete
  frame rejects, while acceptance requires three consistent frames and at least
  one gated-complete frame.
- Deployment bundles contain SHA-256 hashes for every model, scaler,
  calibration, and reference image.

## Results

| Evaluation | Result |
| --- | ---: |
| Synthetic scenes | 50 scenes, 3,000 mapped positions |
| Balanced good-tray synthetic crops | 1,500 |
| Real Testing position pseudo-label agreement | 71.6% overall, 86.1% macro-class |
| Ungated tray classifier | 94.3% accuracy, 97.0% balanced accuracy |
| Confidence-gated Testing false accepts | **0 of 66 incomplete trays** |
| Confidence-gated Testing routing | 1 complete, 62 incomplete, 7 unresolved |
| Controlled multi-frame replay | complete → accept; incomplete → reject; former false accepts → hold |
| Automated tests | **49 tests + 7 calibration subtests passing** |

The zero-false-accept gate is intentionally conservative: three of four genuine
complete Testing frames become unresolved. Multi-frame confirmation recovers the
known complete acquisition sequence without accepting the known false-accept
sequence. This replay validates the state machine, not independent production
accuracy. Full evidence and caveats are in
[REAL_DOMAIN_REPORT.md](vial-inspection/sdg/REAL_DOMAIN_REPORT.md).

## Demo

The deployed Spark service loads the signed bundle once, watches for new camera
frames, persists state across restarts, and atomically updates:

- `latest.json` for a machine-facing route decision;
- `health.json` for service and bundle status;
- session and position JSON/CSVs;
- annotated 60-position images;
- an offline HTML report.

With the VPN connected, forward the Spark's localhost-only report server:

```powershell
ssh -N -L 8765:127.0.0.1:8765 yt-hamstersc2e8@172.30.102.122
```

Then open <http://127.0.0.1:8765/report.html>. See the
[Spark runbook](vial-inspection/deployment/spark/README.md) for runtime and
camera-adapter commands. 

## Reproduce locally

### Required external artifacts

Large datasets and model weights are deliberately excluded by `.gitignore`.

1. Obtain the event-provided tray inspection dataset from the
   [challenge library page](https://msoe-maic.com/library/?article=Rockwell-NVIDIA-Fall-2026-Vial-Inspection)
   and place it at `project/tray-inspection-detection-dataset/`.
2. Use the supplied digital-twin assets under
   `vial-inspection/csi_digital_twin_v1/`. Large binary USD geometry distributed
   with the event is not duplicated in Git.
3. Reproduce the model using [the SDG and training runbook](vial-inspection/sdg/README.md),
   or obtain the team's signed `deployment-bundle-v1` and place it at
   `vial-inspection/sdg/output/deployment-bundle-v1/`.

The bundle is approximately 30.6 MB and contains two position SVMs, the tray
SVM, scaler, thresholds, calibration configuration, and calibration reference
images. Its `manifest.json` is verified before any model is loaded.

### Batch inference

Using a Python environment with Pillow, NumPy, and OpenCV contrib:

```bash
python vial-inspection/sdg/infer_tray.py \
  --labeler vial-inspection/vial-labeler-participants-20261014/vial-labeler-participants-v1/labeler.py \
  --images project/tray-inspection-detection-dataset/images/Testing \
  --files tray_000035.jpg tray_000112.jpg \
  --bundle vial-inspection/sdg/output/deployment-bundle-v1 \
  --output vial-inspection/sdg/output/demo
```

### Tests

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.uv-cache'
uv run --with pytest --with opencv-python-headless --with pillow pytest -q `
  vial-inspection/vial-labeler-participants-20261014/vial-labeler-participants-v1/tests `
  vial-inspection/labeler_extensions/tests `
  vial-inspection/sdg/tests
```

## Repository map

| Path | Purpose |
| --- | --- |
| `vial-inspection/csi_digital_twin_v1/` | Supplied USD digital twin and camera definitions |
| `vial-inspection/labeler_extensions/` | Calibration profiles and clean training exports |
| `vial-inspection/reference_annotations/` | Auditable 60-position reference annotations |
| `vial-inspection/sdg/` | Synthetic generation, training, evaluation, inference, and service code |
| `vial-inspection/deployment/spark/` | DGX Spark launcher and GUI-access runbook |

## Limitations and responsible use

- Only 21 complete-order trays are supplied; Validation contains two.
- Neighboring acquisition frames cross the predefined splits, so reported tray
  metrics remain provisional.
- Position evaluation uses high-confidence fixture pseudo-labels rather than
  certified human ground truth.
- Multi-frame replay uses known supplied sequences and is not an independent
  physical-tray evaluation.
- `hold` must remain a non-accepting route. Do not reinterpret it as pass.
- Camera changes require a new calibration and an independent acceptance test.

## Credits and references

- Rockwell Automation and NVIDIA Innovation Lab: challenge definition and
  event-provided inspection assets.
- MSOE AI Club and ALL Applied AI Network: hackathon organization and repository
  hosting.
- NVIDIA Isaac Sim / Omniverse and OpenUSD: digital-twin composition and
  synthetic rendering.
- OpenCV, NumPy, and Pillow: feature extraction, SVM inference, image processing,
  and report rendering.
- The event-supplied participant labeler was inspected and extended; its package
  does not grant a new redistribution license. See its
  [README](vial-inspection/vial-labeler-participants-20261014/vial-labeler-participants-v1/README.md).
- OpenAI Codex was used as a collaborative engineering assistant for code,
  testing, documentation, and deployment automation; technical decisions and
  reported limitations remain the team's responsibility.

Primary references:

- [Vial Inspection challenge library page](https://msoe-maic.com/library/?article=Rockwell-NVIDIA-Fall-2026-Vial-Inspection)
- [NVIDIA Isaac Sim documentation](https://docs.isaacsim.omniverse.nvidia.com/)
- [OpenCV documentation](https://docs.opencv.org/)
- [OpenUSD documentation](https://openusd.org/release/)
