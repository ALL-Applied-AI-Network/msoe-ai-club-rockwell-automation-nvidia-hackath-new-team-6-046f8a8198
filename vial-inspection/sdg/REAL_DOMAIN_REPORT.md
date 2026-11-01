# Real-domain agreement report

## Result

The synthetic-only HOG/SVM baseline was evaluated on the 30 good-tray
positions in each of four supplied real images. It agreed with the existing
machine-generated reference annotations on **6 of 120 positions (5.0%)**.

This is an agreement diagnostic, not ground-truth accuracy. The reference
annotations were produced by the legacy YOLO plus appearance heuristic and
were not manually certified.

| Calibration profile | Agreement | Pseudo-label distribution |
| --- | ---: | --- |
| `view_001` | 3.3% | 30 pass |
| `view_122` | 13.3% | 21 pass, 5 no-vial, 4 crooked-cap |
| `view_250_dark_upper` | 3.3% | 29 pass, 1 no-vial |
| `view_271_bright_lower` | 0.0% | 29 pass, 1 no-vial |

The baseline recognized 5 of 7 pseudo-labeled `no_vial` examples, but only 1
of 109 pseudo-labeled `pass` examples. Most real pass crops collapsed into
`no_vial` or `crooked_cap`. This failure occurs in every lighting profile, so
brightness augmentation alone will not close the gap.

## Interpretation

The 91.7% synthetic validation result measures separation inside the rendered
domain. The real images differ in vial-rim geometry, transparent-tube
reflections, crop scale, fixture texture, lens response, and surrounding
occlusion. The current synthetic renderer does not cover those appearance
features closely enough for a hand-crafted HOG representation to transfer.

## Recommended next experiment

Use the unattended labeler to pseudo-label the complete real training split,
retain only high-confidence positions, and train a hybrid real-plus-synthetic
baseline. Split by source image, never by crop. Keep the four calibration
images out of training so the same agreement check can measure whether hybrid
training improves transfer. Synthetic generation should then be targeted at
the remaining disagreement modes, especially real pass/crooked-cap separation.

Reproduce the diagnostic with `evaluate_real.py`; its JSON output retains every
position-level disagreement and reference provenance.
