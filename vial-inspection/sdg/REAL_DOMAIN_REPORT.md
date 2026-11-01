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

## Hybrid follow-up

The full real training split was pseudo-labeled at a minimum confidence of
0.90, excluding all four calibration images. All 244 eligible images aligned;
balancing the limiting class produced 40 training and 10 validation examples
per class. Combining those 250 real crops with the 1,500 synthetic crops gave:

| Metric | Synthetic only | Hybrid |
| --- | ---: | ---: |
| Balanced validation accuracy | 91.7% | 90.9% |
| Balanced validation macro-F1 | 91.7% | 90.9% |
| Held-out real pseudo-label agreement | 5.0% | **70.0%** |

Held-out agreement by profile was 73.3% for `view_001`, 66.7% for `view_122`,
80.0% for `view_250_dark_upper`, and 60.0% for
`view_271_bright_lower`. Real `no_vial` agreement was 100%, `crooked_cap` was
75%, and `pass` was 67.9%. No held-out good-tray examples were pseudo-labeled
`missing_cap` or `damaged_vial`, so real transfer for those two classes remains
unmeasured.

The bright-lower regime is now the clearest residual gap. The next data cycle
should emphasize bright-lower pass examples and independently held-out real
missing-cap and damaged-vial examples. These figures still measure agreement
with pseudo-labels, not certified ground-truth accuracy.

## Untouched split evaluation

The hybrid model was subsequently evaluated on the original real Validation
and Testing splits. The two Testing calibration images were excluded, and only
fixture mappings with confidence of at least 0.90 were retained.

| Split | Images | Positions | Overall agreement | Macro class agreement |
| --- | ---: | ---: | ---: | ---: |
| Validation | 35 | 866 | 63.5% | 83.0% |
| Testing | 68 | 1,704 | 66.3% | 86.0% |

| Class | Validation | Testing |
| --- | ---: | ---: |
| pass | 59% (737) | 64% (1,565) |
| missing_cap | 98% (51) | 96% (51) |
| crooked_cap | 76% (58) | 77% (53) |
| damaged_vial | 100% (9) | 100% (9) |
| no_vial | 82% (11) | 92% (26) |

The primary remaining error is false defect alarms on real pass positions. The
balanced training distribution is intentionally unlike the highly pass-heavy
operating distribution. Pass-class cost or decision-threshold calibration must
therefore be selected on Validation before a final Testing measurement. The
small damaged-vial and no-vial supports should still be treated cautiously.

Pass-class SVM weights of 2 and 3 were tried using Validation only. Both
produced the same 63.5% overall and 83.0% macro agreement as weight 1, so the
unweighted hybrid remains the selected baseline. No additional Testing tuning
was performed. The false-positive reduction needs more varied real pass crops
or a two-stage pass/defect model rather than a class-cost change.

## Two-stage result

A binary pass/defect SVM followed by a four-class defect SVM was trained on the
same hybrid crops. Validation-only threshold selection chose a defect score of
`0.399706184864044`. The locked threshold was then evaluated once on Testing.

| Metric | One-stage Testing | Two-stage Testing |
| --- | ---: | ---: |
| Overall agreement | 66.3% | **71.6%** |
| Macro class agreement | 86.0% | **86.1%** |
| pass | 64% | **70%** |
| missing_cap | 96% | 96% |
| crooked_cap | **77%** | 72% |
| damaged_vial | 100% | 100% |
| no_vial | 92% | 92% |

The two-stage gate reduces false defect alarms while retaining most defect
performance, at the cost of five percentage points of crooked-cap agreement.
The selected threshold and evaluation provenance are versioned in
`two_stage_profile.json`.

## Tray-level aggregation audit

Applying “any predicted defect means incomplete order” compounds position-level
false alarms across 30 slots. A maximum-gate-score threshold was therefore
selected on the 35 Validation trays. Only two Validation trays were complete;
the selected threshold achieved 65.2% balanced accuracy but detected just
30.3% of incomplete trays.

The locked threshold failed on the 70 Testing trays: 17.1% overall accuracy,
32.6% balanced accuracy, 50% complete-order recall, and only 15.2%
incomplete-order recall. It is not a deployable threshold and is deliberately
not included as a selected project profile. The auditable inference command
keeps the conservative inspection-rule verdict until a genuine tray-level
model can be trained. A future tray model needs aggregated score-distribution
features and more complete-order training examples; maximum score alone is not
separable on this dataset.

## Learned tray classifier

A tray classifier was trained from the distribution of all 30 good-position
gate scores, score quantiles and summary statistics, and predicted defect-type
counts. The selected version deliberately excludes ordered slot identity and
calibration-profile identity. An RBF SVM (`C=0.1`, `gamma=0.01`) was selected
on Validation. The saved model remains fit on Training only so Validation can
calibrate a confidence gate without contaminating the model it evaluates.

Before confidence gating, Testing results were 94.3% accuracy and 97.0%
balanced accuracy: all four complete trays were classified as complete, but
four of 66 incomplete trays were false accepts. A complete-order margin gate
was then selected exclusively on Validation. Its `0.809390` threshold admits
no incomplete Validation tray; lower-confidence nominal-complete decisions
become `unresolved` rather than being promoted or silently rejected.

On the Testing regression, the gate produced zero false accepts, one accepted
complete tray, 62 incomplete decisions, and seven unresolved decisions. Three
of the unresolved trays were genuinely complete. This is a deliberate
fail-safe tradeoff, not an accuracy improvement: uncertain trays must remain on
the reject/review path until more complete-order calibration data are available.

These results remain provisional. Only 21 complete trays exist, and neighboring
frames from the complete-order acquisition run occur across the predefined
splits. The distribution-only ablation reduces direct view/position leakage,
but only two complete trays are available to calibrate the Validation gate and
neither clears the zero-false-accept threshold. A genuinely independent
acquisition is still required before deployment.
The selected configuration and caveat are versioned in
`tray_classifier_profile.json`.

## Multi-frame runtime mitigation

The inspection service can group camera frames by physical tray ID without
lowering the locked single-frame confidence gate. Any incomplete frame rejects
the session immediately. A session is accepted only after at least three
frames, when every base tray prediction is complete-like and at least one frame
clears the confidence gate. All other sessions remain on hold.

A controlled replay exercised all three routes. Frames `tray_000108`,
`tray_000109`, and `tray_000112` formed a complete sequence and were accepted
on frame three; the additional `tray_000117` frame was ignored after session
finalization. A known incomplete frame was rejected immediately. The former
false-accept frames `tray_000063`, `tray_000092`, and `tray_000194` remained on
hold because none cleared the gate. This replay verifies the state machine, not
a new independent accuracy result: production acceptance still requires tray
IDs from the capture system and an independent multi-tray session evaluation.
