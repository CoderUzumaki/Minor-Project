"""Prepare and evaluate a real-data report-level attack pilot; write a portable demo.

Annotations are used only by the offline injector and evaluator. The online API
is scripts/replay_core.py and can be run from exported inputs without nuScenes.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
from pyquaternion import Quaternion
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN

from replay_core import Config, replay

ROOT = Path(__file__).resolve().parents[1]
SENSORS = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_splits(nusc, path):
    """Group whole recordings before features, attack generation or fitting."""
    scenes = sorted(nusc.scene, key=lambda s: s['name'])
    groups = sorted({s['log_token'] for s in scenes}, key=lambda s: hashlib.sha256(('20261109:' + s).encode()).hexdigest())
    n = len(groups)
    if n < 3:
        raise ValueError('At least three recordings are required for grouped pilot splits')
    ntest = max(1, n // 5)
    split = {g: ('test' if i < ntest else 'validation' if i < 2 * ntest else 'train') for i, g in enumerate(groups)}
    manifest = dict(schema_version=1, dataset='v1.0-mini', purpose='pilot only; not a publication benchmark',
                    grouping='nuScenes log_token; nearby recordings may still be correlated', seed=20261109,
                    scenes=[dict(scene=s['name'], scene_token=s['token'], log_token=s['log_token'], split=split[s['log_token']]) for s in scenes])
    path = Path(path)
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != manifest:
            raise ValueError('Existing split manifest differs; refusing to silently replace it')
    else:
        write_json(path, manifest)
    return manifest


def extract(nusc, scene):
    samples = []
    token = scene['first_sample_token']
    while token:
        sample = nusc.get('sample', token)
        samples.append(sample)
        token = sample['next']
    start_us = samples[0]['timestamp']
    first_sd = nusc.get('sample_data', samples[0]['data']['RADAR_FRONT'])
    origin = np.asarray(nusc.get('ego_pose', first_sd['ego_pose_token'])['translation'])
    last_us = samples[-1]['timestamp']
    records = [sd for sd in nusc.sample_data if sd['channel'] in SENSORS
               and nusc.get('sample', sd['sample_token'])['scene_token'] == scene['token']
               and sd['timestamp'] <= last_us]
    records.sort(key=lambda sd: (sd['timestamp'], sd['channel']))
    events, sources = [], []
    for sd in records:
        path = Path(nusc.dataroot) / sd['filename']
        points = RadarPointCloud.from_file(str(path), invalid_states=[0], dynprop_states=list(range(7)), ambig_states=[3]).points
        valid = np.isfinite(points[:3]).all(axis=0) & (np.linalg.norm(points[:2], axis=0) <= 60.)
        points = points[:, valid]
        cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        pose = nusc.get('ego_pose', sd['ego_pose_token'])
        xyz = Quaternion(cs['rotation']).rotation_matrix @ points[:3] + np.asarray(cs['translation'])[:, None]
        xyz = Quaternion(pose['rotation']).rotation_matrix @ xyz + np.asarray(pose['translation'])[:, None] - origin[:, None]
        xy = xyz[:2].T
        reports = []
        if len(xy):
            labels = DBSCAN(eps=2., min_samples=1).fit_predict(xy)
            for label in sorted(set(labels)):
                mask = labels == label
                cluster = xy[mask]
                # Centroids are radar reflection centers, not annotated box centers.
                scatter = np.cov(cluster.T) / len(cluster) if len(cluster) > 1 else np.zeros((2, 2))
                r = scatter + np.eye(2) * 1.5 ** 2
                rcs = points[5, mask]
                reports.append(dict(id=f'{sd["token"]}:{label}', position=cluster.mean(axis=0).tolist(),
                                    covariance=r.tolist(), point_count=len(cluster),
                                    rcs_mean=float(np.mean(rcs[np.isfinite(rcs)])) if np.isfinite(rcs).any() else None))
        events.append(dict(sensor=sd['channel'], time_s=(sd['timestamp'] - start_us) / 1e6,
                           acquisition_timestamp_us=sd['timestamp'], reception_timestamp_us=None,
                           sweep_token=sd['token'], reports=reports))
        sources.append(dict(file=sd['filename'], sha256=sha(path)))
    references = []
    for sample in samples:
        sd = nusc.get('sample_data', sample['data']['RADAR_FRONT'])
        ego = np.asarray(nusc.get('ego_pose', sd['ego_pose_token'])['translation'])
        objects = []
        for token in sample['anns']:
            a = nusc.get('sample_annotation', token)
            category = a['category_name']
            eligible = ((category.startswith('vehicle.') and category != 'vehicle.bicycle_rack')
                        or category.startswith('human.pedestrian.')) and np.linalg.norm(np.asarray(a['translation'])[:2] - ego[:2]) <= 60
            objects.append(dict(instance=a['instance_token'], position=(np.asarray(a['translation']) - origin)[:2].tolist(),
                                size=a['size'], yaw=Quaternion(a['rotation']).yaw_pitch_roll[0],
                                category=category, eligible=bool(eligible)))
        references.append(dict(time_s=(sample['timestamp'] - start_us) / 1e6, sample_token=sample['token'],
                               ego=(ego - origin)[:2].tolist(), objects=objects))
    return events, references, dict(origin_global=origin.tolist(), start_timestamp_us=start_us, radar_files=sources)


def inside(position, obj, padding=.5):
    delta = np.asarray(position) - obj['position']
    c, s = np.cos(obj['yaw']), np.sin(obj['yaw'])
    local = np.array([[c, s], [-s, c]]) @ delta
    return abs(local[0]) <= obj['size'][1] / 2 + padding and abs(local[1]) <= obj['size'][0] / 2 + padding


def inject(events, references, attack):
    """Offline annotation-assisted target selection, never an inference feature.

    Use the latest annotation box for a transparent pilot association proxy.
    Shared shifts at equal times implement a coordinated displacement process.
    """
    altered = json.loads(json.dumps(events))
    labels = []
    times = [r['time_s'] for r in references]
    for event in altered:
        t = event['time_s']
        idx = int(np.searchsorted(times, t, side='right')) - 1
        objects = references[idx]['objects'] if idx >= 0 else []
        for report in event['reports']:
            matches = [o for o in objects if inside(report['position'], o)]
            active = (attack['start_s'] <= t <= attack['end_s'] and event['sensor'] in attack['sensors']
                      and len(matches) == 1 and matches[0]['instance'] == attack['target_instance'])
            if active:
                scale = min(1., (t - attack['start_s']) / attack['ramp_s']) if attack['ramp_s'] > 0 else 1.
                shift = np.asarray(attack['displacement_m']) * scale
                report['position'] = (np.asarray(report['position']) + shift).tolist()
                active = bool(np.linalg.norm(shift) > 0)
            labels.append(dict(report_id=report['id'], time_s=t, sensor=event['sensor'], attacked=bool(active)))
    return altered, labels


def frame_metrics(estimates, truth, cutoff=10., match_gate=4.):
    """GOSPA p=2, alpha=2; independent 4m matching for RMSE and counts."""
    a, b = np.asarray(estimates).reshape(-1, 2), np.asarray(truth).reshape(-1, 2)
    n, m = len(a), len(b)
    distances = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    if n and m:
        ii, jj = linear_sum_assignment(np.minimum(distances ** 2, cutoff ** 2))
        gospa2 = float(np.minimum(distances[ii, jj] ** 2, cutoff ** 2).sum()) + cutoff ** 2 / 2 * abs(n - m)
        # Maximize number of gated matches first, then minimize distance.
        cost = np.where(distances <= match_gate, distances / max(n, m, 1), 1e6)
        ii, jj = linear_sum_assignment(cost)
        errors = distances[ii, jj][distances[ii, jj] <= match_gate]
    else:
        gospa2, errors = cutoff ** 2 / 2 * (n + m), np.array([])
    return dict(gospa=float(np.sqrt(gospa2)), matched=len(errors), false_tracks=n - len(errors),
                missed_targets=m - len(errors), squared_error_sum=float((errors ** 2).sum()))


def evaluate(result, references, labels, attack):
    rows = []
    for frame, ref in zip(result['frames'], references):
        # Count all emitted hypotheses, including clutter, against all eligible GT.
        truth = [o['position'] for o in ref['objects'] if o['eligible']]
        row = frame_metrics([r['state'][:2] for r in frame['fused']], truth)
        row.update(time_s=frame['time_s'])
        rows.append(row)
    label_map = {r['report_id']: r['attacked'] for r in labels}
    scored = [dict(**r, attacked=label_map[r['report_id']]) for r in result['decisions']]
    tp = sum(r['alert'] and r['attacked'] for r in scored)
    fp = sum(r['alert'] and not r['attacked'] for r in scored)
    fn = sum(not r['alert'] and r['attacked'] for r in scored)
    tn = sum(not r['alert'] and not r['attacked'] for r in scored)
    matched = sum(r['matched'] for r in rows)
    actual = [r['time_s'] for r in scored if r['attacked']]
    hits = [r['time_s'] for r in scored if r['attacked'] and r['alert']]
    clean_minutes = (references[-1]['time_s'] - references[0]['time_s']) / 60
    summary = dict(mean_gospa=float(np.mean([r['gospa'] for r in rows])),
                   matched_position_rmse_m=float(np.sqrt(sum(r['squared_error_sum'] for r in rows) / matched)) if matched else None,
                   matched_object_frames=matched, false_track_frames=sum(r['false_tracks'] for r in rows),
                   missed_target_frames=sum(r['missed_targets'] for r in rows),
                   detection=dict(tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
                                  precision=tp / (tp + fp) if tp + fp else None,
                                  recall=tp / (tp + fn) if tp + fn else None,
                                  f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                                  attacked_reports=len(actual), event_detected=bool(hits) if actual else None,
                                  event_delay_s=min(hits) - min(actual) if hits else None,
                                  # Per-report alerts, not deduplicated attack events.
                                  false_alert_reports_per_minute=fp / clean_minutes if clean_minutes else None),
                   timing=result['timing'])
    return summary, rows, scored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataroot', default=str(ROOT / 'data/nuscenes'))
    parser.add_argument('--output', default=str(ROOT / 'outputs/replay_pilot'))
    parser.add_argument('--scene', help='Training scene only; default is longest audited training episode')
    parser.add_argument('--displacement', type=float, default=4.)
    parser.add_argument('--ramp', type=float, default=0.)
    args = parser.parse_args()
    if not np.isfinite(args.displacement) or not np.isfinite(args.ramp) or args.ramp < 0 or args.displacement == 0:
        parser.error('Displacement must be finite and nonzero; ramp must be finite and nonnegative')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    nusc = NuScenes(version='v1.0-mini', dataroot=args.dataroot, verbose=False)
    manifest_path = ROOT / 'data_manifests/mini_splits.json'
    splits = freeze_splits(nusc, manifest_path)
    training = {s['scene'] for s in splits['scenes'] if s['split'] == 'train'}
    candidates_path = ROOT / 'outputs/mini_audit/candidate_episodes.csv'
    candidates = pd.read_csv(candidates_path)
    candidates = candidates[(candidates.sensor_count == 2) & candidates.scene.isin(training) & (candidates.duration_s >= 4)]
    if args.scene:
        candidates = candidates[candidates.scene == args.scene]
    if candidates.empty:
        raise ValueError('No >=4s training episode found. Run the overlap audit first; do not use held-out scenes for the demo.')
    episode = candidates.sort_values(['duration_s', 'scene'], ascending=[False, True]).iloc[0]
    scene = next(s for s in nusc.scene if s['name'] == episode.scene)
    print(f'Preparing {scene["name"]}: training-only pilot, target {episode.instance_token}', flush=True)
    events, references, provenance = extract(nusc, scene)
    time_lookup = {r['sample_token']: r['time_s'] for r in references}
    start = time_lookup[episode.start_sample] + 2.
    end = min(start + 6., time_lookup[episode.end_sample])
    attack = dict(type='coordinated_position_displacement', injection_boundary='clustered position reports before tracking',
                  target_instance=episode.instance_token, sensors=episode.sensors.split('|'),
                  start_s=start, end_s=end, displacement_m=[args.displacement, 0.], ramp_s=args.ramp,
                  target_selection='exclusive membership in latest offline annotation box +0.5m; association proxy',
                  rf_validated=False)
    attacked_events, labels = inject(events, references, attack)
    counts = {s: sum(r['attacked'] and r['sensor'] == s for r in labels) for s in attack['sensors']}
    if not all(counts.values()):
        raise RuntimeError(f'No actual corruption in one of the selected streams: {counts}')
    attack['actual_corrupted_report_counts'] = counts
    frame_times = [r['time_s'] for r in references]
    # These files are sufficient for inference. No annotation/label fields inside.
    write_json(out / 'inputs/clean.json', dict(events=events, frame_times=frame_times))
    write_json(out / 'inputs/attacked.json', dict(events=attacked_events, frame_times=frame_times))
    write_json(out / 'evaluation/reference.json', references)
    write_json(out / 'evaluation/attack.json', attack)
    write_json(out / 'evaluation/labels.json', labels)
    cfg = Config()
    results, summaries, metric_rows = {}, {}, {}
    clean_labels = [dict(r, attacked=False) for r in labels]
    for name, input_events, protected, input_labels in [
        ('clean_baseline', events, False, clean_labels), ('clean_protected', events, True, clean_labels),
        ('attacked_baseline', attacked_events, False, labels), ('attacked_protected', attacked_events, True, labels)]:
        print(f'Running {name} ...', flush=True)
        result = replay(input_events, frame_times, protected, cfg)
        summary, rows, scored = evaluate(result, references, input_labels, attack)
        write_json(out / f'runs/{name}.json', result)
        pd.DataFrame(rows).to_csv(out / f'{name}_tracking.csv', index=False)
        pd.DataFrame(scored).to_csv(out / f'{name}_decisions.csv', index=False)
        results[name], summaries[name], metric_rows[name] = result, summary, rows
    source_paths = [ROOT / 'scripts' / name for name in ('replay_core.py', 'run_replay.py', 'render_replay.py', 'infer_replay.py', 'verify_replay.py')]
    provenance.update(scene=scene['name'], scene_token=scene['token'], split='train',
                      manifest_sha256=sha(manifest_path), config=asdict(cfg),
                      preprocessing=dict(range_m=60, dbscan_eps_m=2, min_samples=1, centroid_std_floor_m=1.5),
                      sources={p.name: sha(p) for p in source_paths},
                      metadata_sha256={p.name: sha(p) for p in sorted((Path(args.dataroot) / 'v1.0-mini').glob('*.json'))},
                      inputs_sha256={name: sha(out / f'inputs/{name}.json') for name in ('clean', 'attacked')},
                      evaluation_sha256={name: sha(out / f'evaluation/{name}.json') for name in ('attack', 'labels', 'reference')},
                      candidate_manifest_sha256=sha(candidates_path), requirements_sha256=sha(ROOT / 'requirements.txt'),
                      environment=dict(python=platform.python_version(), platform=platform.platform(),
                                       packages={p: importlib.metadata.version(p) for p in ['numpy', 'scipy', 'scikit-learn', 'nuscenes-devkit', 'pandas']}),
                      elapsed_before_render_s=time.perf_counter() - started,
                      limitations=['Pilot heuristic thresholds; no validation calibration or trained model.',
                                   'No hardware/RF validation. Report-level injection after clustering.',
                                   'Reflection centroids do not equal object centers; full scene clutter retained.',
                                   'Three-radar recovery not established. Shared pose errors unmodeled.',
                                   'New/unmatched reports are unknown; sustained corruption may initialize new tracks.',
                                   'No visibility-aware suppression detector or persistent fused IDs yet.',
                                   'Local-update timing excludes decoding, feature extraction, fusion and display.'])
    identity = {key: provenance[key] for key in ('sources', 'inputs_sha256', 'evaluation_sha256', 'config', 'manifest_sha256')}
    provenance['run_id'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    write_json(out / 'provenance.json', provenance)
    write_json(out / 'summary.json', summaries)
    from render_replay import render
    render(out, results, summaries, metric_rows, references, attack, provenance)
    print(f'Finished. Open {out / "demo.html"}', flush=True)


if __name__ == '__main__':
    main()
