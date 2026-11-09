# Causal radar replay pilot

Scene: `scene-0553`. Split: **train / development only**. Run: `08ce61efb51817c6`.

Measured report-level simulation on recorded radar data. No learned model or RF validation.

| Branch | Mean GOSPA ↓ | Matched RMSE (m) | False track-frames | Missed target-frames |
|---|---:|---:|---:|---:|
| clean_baseline | 56.515 | 1.888 | 2351 | 570 |
| clean_protected | 54.082 | 1.853 | 2104 | 607 |
| attacked_baseline | 56.552 | 1.981 | 2351 | 570 |
| attacked_protected | 54.099 | 1.943 | 2104 | 607 |

RMSE includes only matches within 4m and must be read with false/missed counts. GOSPA includes all emitted hypotheses and eligible annotations; reflection centroids are not object centers.

## Detector measurements

| Branch | TP | FP | FN | Precision | Recall | First event delay (s) |
|---|---:|---:|---:|---:|---:|---:|
| clean_baseline | 0 | 3084 | 0 | 0.000 | n/a | n/a |
| clean_protected | 0 | 1866 | 0 | 0.000 | n/a | n/a |
| attacked_baseline | 4 | 3088 | 316 | 0.001 | 0.013 | 3.776 |
| attacked_protected | 41 | 2007 | 279 | 0.020 | 0.128 | 0.725 |

The baseline branches run the detector in monitor-only mode. Protected branches reject flagged updates and inflate covariance using heuristic temporal trust. No detection means a null delay, never zero.

False-alert counts are report decisions, not independent attack events. All rows come from one development scene and one injection; there is no held-out performance claim.

Compare each attacked branch with its own clean branch. Protection also changes clean tracking, so a lower protected GOSPA alone is not evidence of successful attack mitigation.

## Limits and next work

- Pilot heuristic thresholds; no validation calibration or trained model.
- No hardware/RF validation. Report-level injection after clustering.
- Reflection centroids do not equal object centers; full scene clutter retained.
- Three-radar recovery not established. Shared pose errors unmodeled.
- New/unmatched reports are unknown; sustained corruption may initialize new tracks.
- No visibility-aware suppression detector or persistent fused IDs yet.
- Local-update timing excludes decoding, feature extraction, fusion and display.

Next: inspect episodes and clutter, calibrate frontend/residual thresholds on training/validation recordings, add benign stresses and additional attack families, then compare GRU and temporal graph detection. See HANDOFF.md.

Open `demo.html` for playback, scrub controls, reference overlays, and clean/attacked/protected comparisons. All display annotations are offline overlays; inference cannot access them.