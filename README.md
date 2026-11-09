# Coordinated radar attack research

Implemented milestones: **nuScenes mini ingestion, observed radar-overlap audit,
and causal clean/attacked replay with a residual-detector baseline and offline display**.
The research specification is [Research_Plan.md](Research_Plan.md); no separate `Plan.md`
was present when implementation began. No learned attack detector has been trained yet.
Current implementation status, evidence and remaining work are in [HANDOFF.md](HANDOFF.md).

## Run and display the working pilot

Open `outputs/replay_pilot/demo.html` in a desktop browser. It is self-contained;
no server is required. Use Play or the timeline to compare clean, attacked and
protected tracks. Reference boxes and attack labels are explicitly offline overlays.
The initial measured baseline has many false alarms and low attack recall; it is
a working experimental pipeline, **not a validated defense**.

To reproduce locally after dataset preparation and the overlap audit:

```powershell
.venv\Scripts\python -m pytest -q --basetemp=outputs/test_tmp
.venv\Scripts\python -u scripts/run_replay.py
.venv\Scripts\python -u scripts/verify_replay.py
```

For Colab, upload `notebooks/02_causal_radar_replay.ipynb` and Run all. It embeds
the source, prepares the data, runs the audit and replay, checks isolated inference,
and produces a downloadable results ZIP containing the presentation.

`outputs/replay_pilot/REPORT.md` and `summary.json` contain measured results.
`inputs/` contains annotation-free clean and attacked reports; `evaluation/`
contains labels and references; `runs/` contains causal outputs. `provenance.json`
records parameters, hashes and package versions. `data_manifests/mini_splits.json`
groups recordings before new features or attacks are created. These are pilot splits;
the ten mini scenes are not a sufficient publication benchmark.

The online entry point can run without nuScenes annotations or attack labels:

```powershell
.venv\Scripts\python scripts/infer_replay.py outputs/replay_pilot/inputs/attacked.json outputs/replay_pilot/standalone.json --protected
```

## Google Colab (recommended)

Open `notebooks/01_nuscenes_mini_radar_audit.ipynb` with Colab's **Upload notebook**.
Use a **CPU** runtime and **Run all**. The notebook is self-contained; it embeds the
same source scripts and tests as this repository. It creates an isolated Python 3.12
environment, checks dependencies, downloads the official mini archive, extracts the
radar subset, runs the audit, and displays measured results. It does not mount Drive.

Live notebook created for this project:
https://colab.research.google.com/drive/10yvF-r2LUNI3LQzEJMReiqtBpwGw1HVu

Temporary data is stored under `/content/radar-project/data/nuscenes`. Before ending
the runtime, download `/content/radar-project/mini_audit_results.zip` using the final
cell's optional `files.download(archive)` command or the Colab Files panel.
The notebook itself is saved in Drive, but runtime data is not automatically persistent.

The initial archive is 4,167,696,325 bytes (~4.17 GB). It contains every modality;
selective extraction saves disk space but not bandwidth. Reserve at least 8 GB free.
Download is resumable. The official source is https://www.nuscenes.org/data/v1.0-mini.tgz.
Dataset use is subject to https://www.nuscenes.org/terms-of-use. Do not commit or
redistribute raw data. No unofficial mirrors or credentials are embedded.

## Local setup

Use Python **3.12**, the target recommended by the current devkit documentation.
This workspace has `.venv`; to recreate on Windows with Python 3.12 installed:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pytest -q
.venv\Scripts\python -u scripts/prepare_nuscenes.py
.venv\Scripts\python -u scripts/audit_radar_overlap.py
```

Linux/Colab equivalents use `.venv/bin/python`. There is no GPU dependency at this
stage. `requirements.txt` pins direct dependencies; each audit records installed
transitive versions in `environment_freeze.txt`.

## What overlap means

For each annotated keyframe, retrieve the latest sweep from each of the five radar
streams **at or before** the annotation time. Transform radar points with that
sweep's calibration and ego pose into the global frame. Use the annotated object's
oriented BEV box to count supporting returns. Do not combine historical sweeps into
one apparently simultaneous observation.

Primary settings, fixed before examining results:

- Vehicle/pedestrian annotations within 60 m; bicycle racks excluded.
- Latest causal sweep no older than 100 ms; no future radar measurements.
- Default devkit validity filters, explicitly specified in the script.
- At least one valid return per supporting radar in a box padded by 0.5 m per side.
- Returns falling into multiple annotation boxes are excluded from all of them.
- A candidate episode has the **same pair or triple** supporting the **same target**
  for at least three consecutive keyframes, spanning at least 0.9 seconds.

Sensitivity: 50/100/200 ms age limits, 0/0.5 m padding, 1/2 returns per sensor, and a
static-box comparison. Annotation-derived velocity back-translates the box center to
the sweep timestamp; it may use future annotations, is an offline association aid,
and must NEVER become an online model feature. Missing velocities use static boxes
and are counted. Box orientation is held fixed within the short window.

This measures **observed geometric support**, not theoretical field of view, exact
radar point identity, independent evidence, or attack detectability. Empty support
can reflect occlusion, radar sparsity, or filtering. Padding can include clutter.
Keyframe continuity does not establish uninterrupted high-frequency support.

## Outputs

`outputs/mini_audit/` contains:

| File | Purpose |
|---|---|
| `REPORT.md`, `summary.json` | Measured results, method, configuration and limitations |
| `object_coverage.csv` | Every eligible object-keyframe under every configuration |
| `sweep_inventory.csv` | Actual selected sweep tokens, ages and valid return counts |
| `sensitivity.csv` | How timing, padding and support requirements affect counts |
| `per_scene.csv`, `per_category.csv`, `sensor_pairs.csv` | Disaggregated evidence |
| `overlap_episodes.csv`, `candidate_episodes.csv` | Exact fixed-pair/triple episode IDs |
| `coverage.png`, `example_overlap.png` | Plots derived from measured data |
| `example_manifest.json` | Scene, object and sweep IDs behind the example |
| `environment_freeze.txt` | Installed package versions |

The dataset preparation manifest records archive and extracted-file SHA-256 hashes.
The audit records metadata hashes and its source hash. Reuse checks sizes; use hashes
when independently verifying a transferred dataset.

## Feasibility decision

Two-radar episodes support experiments that manipulate two streams. Three-radar
episodes may support two manipulated streams plus a remaining observer, subject to
manual inspection and association checks. Neither guarantees recoverability.

Mini's ten scenes are a **pilot**, not a sufficient training/test benchmark. Do not
make a decision from raw totals alone: inspect strict-box/two-return sensitivity,
moving-target subsets, sustained episodes, and diversity across scenes. If third-
observer support is scarce, narrow the radar-only claims or evaluate a larger subset
before committing to a learned defense. Never generate fake overlap to rescue scope.

## Development

After source changes, rebuild the portable notebook:

```powershell
python scripts/build_notebook.py
```

Tests cover causal timestamp selection, sensor-to-global transformations, rotated
box geometry, ambiguous-point exclusion, fixed-sensor episode continuity and archive
path safety. The notebook executes these tests before downloading data.

Primary references:
- https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/installation.md
- https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py
- https://www.nuscenes.org/tutorials/nuscenes_tutorial.html
