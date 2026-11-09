# Presenting the radar replay prototype

Open `outputs/replay_pilot/demo.html` in a desktop browser. It needs no server or
internet connection. Keep the results folder together for linked evidence files.
The HTML still needs a human browser check; automated visual verification was blocked.

## Five-minute walkthrough

1. **Problem (30s):** two radar-report streams may be manipulated together. These
   are recorded benign nuScenes measurements with software-injected modifications,
   not an over-the-air hardware attack.
2. **Clean history (45s):** press Play. Dots are clustered reflection centers,
   circles are fused estimates, ellipses are modeled uncertainty. Reference boxes
   are optional offline overlays.
3. **Attack (60s):** pause in the highlighted window. The middle panel receives a
   shared 4m shift in selected rear-radar reports. The right panel receives exactly
   the same modified input with baseline protection. Scrub to compare the same time.
4. **Response and limits (60s):** explain residuals, sequential anomaly scores,
   rejection and coasting. Tracks can expire or restart. Coordinated agreement can
   evade detection, and absent evidence is not proof of malicious activity.
5. **Evidence (60s):** read all four table rows, including clean protected. The first
   detector has many false alerts and low recall. Lower GOSPA alone does not establish
   successful mitigation; protection also changes clean tracking.
6. **Research next step (45s):** frontend calibration, more attack/benign-stress
   cases, GRU baseline, then temporal graph learning and held-out comparison.

## Accurate answers

- **Live radar?** No, causal replay of real recorded radar data.
- **GNN running?** No, the running detector is innovation/CUSUM.
- **Does inference see the boxes?** No, reference boxes and labels are offline overlays.
- **Reliable coordinated-attack detection?** Not yet; the measured baseline misses
  most altered reports and has many false alarms.
- **Recover truth when all observers agree on a plausible false target?** Not in
  general; independent evidence is needed for otherwise indistinguishable cases.
- **Reproducible?** Sources, inputs, parameters, hashes and decision logs are saved.
  Isolated inference checks frames/decisions without annotations or attack labels.

Do not call heuristic trust a probability, one event's delay an average, or this
development pilot a held-out accuracy result. No hardware-safety claim is supported.
