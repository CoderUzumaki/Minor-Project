# nuScenes mini radar coverage audit

This is a measured offline feasibility audit, not attack detection or mitigation validation.

Loaded 10 scenes, 404 keyframes and 13263 radar files.
Primary denominator: 13015 vehicle/pedestrian object-keyframes within 60 m of ego; 4921 have radar support.

| Supporting sensors | Object-keyframes |
|---|---:|
| 0 | 8094 |
| 1 | 3525 |
| 2 | 1372 |
| 3 | 24 |
| 4 | 0 |
| 5 | 0 |

Two-plus support: 1396/13015 (10.73% of eligible; 28.37% of radar-observed).
Three-plus support: 24/13015 (0.18% of eligible).

Fixed-pair candidate episodes (>=3 consecutive keyframes, >=0.9 s endpoint span): 149 in 10 scenes.
Fixed-triple candidate episodes under the same rule: 2 in 2 scenes.
Moving objects (offline speed >=0.5 m/s): 5422 object-keyframes; 475 with 2+ support; 3 with 3+ support.

## Interpretation

Pair overlap supplies candidate sequences for two-stream corruption. A fixed triple may supply two corrupted streams plus a third observing stream; it does not guarantee the third is independent or sufficient to identify truth.
A few candidate episodes justify a pilot only. Sustained support across independent scenes and held-out data is needed before a publication-scale claim. Inspect candidates visually and rerun stricter matching before selecting attacks.

## Sensitivity

| Configuration | Eligible | 2+ | 3+ |
|---|---:|---:|---:|
| age50_margin0_points1 | 13015 | 479 | 4 |
| age50_margin0_points2 | 13015 | 113 | 1 |
| age50_margin0.5_points1 | 13015 | 588 | 5 |
| age50_margin0.5_points2 | 13015 | 150 | 1 |
| age100_margin0_points1 | 13015 | 1148 | 22 |
| age100_margin0_points2 | 13015 | 292 | 5 |
| age100_margin0.5_points1 | 13015 | 1396 | 24 |
| age100_margin0.5_points2 | 13015 | 386 | 6 |
| age200_margin0_points1 | 13015 | 1163 | 23 |
| age200_margin0_points2 | 13015 | 295 | 6 |
| age200_margin0.5_points1 | 13015 | 1412 | 25 |
| age200_margin0.5_points2 | 13015 | 389 | 7 |
| age100_margin0.5_points1_static_boxes | 13015 | 1395 | 24 |

## Method and limitations

- Offline annotation-assisted coverage proxy, not a detector or exact point identity.
- Mini is a feasibility pilot, not training or security validation.
- Velocity can use future annotations for offline matching only.
- BEV padding can overcount clutter; strict/two-point sensitivities are provided.
- Observed overlap is not a guarantee of independence, visibility, or recoverability.
- Episodes are keyframe observations; continuity between keyframes is not established.
- Primary: latest sweep at or before annotation timestamp, age <=100 ms; >=1 valid return in an oriented BEV box padded by 0.5 m per side.
- A point in multiple annotated boxes is excluded from every box. No stacked multisweep cloud is treated as simultaneous evidence.
- Annotations and annotation-derived velocity are confined to this offline audit. They must never become online detector inputs.
- Official default radar validity filters are explicit in summary.json; unfiltered radar results are not claimed.
- Ground-truth category/range restriction determines the audit denominator; lack of support is not proof of lack of field of view.

Source: https://github.com/nutonomy/nuscenes-devkit and https://www.nuscenes.org/nuscenes
Reproduce with scripts/audit_radar_overlap.py. CSVs, hashes, package versions and parameters accompany this report.
