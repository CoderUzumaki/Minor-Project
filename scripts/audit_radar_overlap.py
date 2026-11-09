"""Offline annotation-assisted radar coverage audit; NOT an online detector.

Only the latest causal sweep per radar is used. Annotation box velocities may
use future annotations: they are offline matching aids, never model features.
BEV box membership is a coverage proxy, not ground-truth point association.
"""
from __future__ import annotations
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import importlib.metadata
import itertools
import json
import platform
from pathlib import Path
import subprocess
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud

CHANNELS = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']


def latest_causal(times, timestamp, max_age_us):
    idx = bisect_right(times, timestamp) - 1
    return idx if idx >= 0 and timestamp - times[idx] <= max_age_us else None


def transform_points(points, calibration, pose):
    p = Quaternion(calibration['rotation']).rotation_matrix @ points[:3]
    p += np.asarray(calibration['translation'])[:, None]
    p = Quaternion(pose['rotation']).rotation_matrix @ p
    return p + np.asarray(pose['translation'])[:, None]


def box_membership(points, boxes, margin):
    """Oriented horizontal boxes; dimensions are nuScenes width,length,height."""
    hit = np.zeros((len(boxes), points.shape[1]), dtype=bool)
    for i, box in enumerate(boxes):
        yaw = Quaternion(box['rotation']).yaw_pitch_roll[0]
        c, s = np.cos(yaw), np.sin(yaw)
        delta = points[:2] - np.asarray(box['translation'])[:2, None]
        local_x = c * delta[0] + s * delta[1]
        local_y = -s * delta[0] + c * delta[1]
        width, length, _ = box['size']
        hit[i] = (np.abs(local_x) <= length / 2 + margin) & (np.abs(local_y) <= width / 2 + margin)
    return hit


def exclusive_counts(hit):
    """Exclude returns inside multiple boxes, rather than double-counting them."""
    unique = hit.sum(axis=0) == 1
    return (hit & unique).sum(axis=1), int((hit.sum(axis=0) > 1).sum())


def sustained_episodes(rows, threshold, max_gap_s=0.75):
    """Consecutive keyframes for a FIXED sensor pair/triple and target."""
    groups = defaultdict(list)
    for row in rows:
        for combo in itertools.combinations(row['sensors'], threshold):
            groups[(row['scene'], row['instance_token'], combo)].append(row)
    episodes = []
    for (scene, instance, combo), records in groups.items():
        records.sort(key=lambda r: r['timestamp_us'])
        chunks = []
        chunk = []
        for r in records:
            if chunk and (r['frame_index'] != chunk[-1]['frame_index'] + 1 or (r['timestamp_us'] - chunk[-1]['timestamp_us']) / 1e6 > max_gap_s):
                chunks.append(chunk)
                chunk = []
            chunk.append(r)
        if chunk:
            chunks.append(chunk)
        for chunk in chunks:
            episodes.append({'scene': scene, 'instance_token': instance, 'sensors': '|'.join(combo),
                             'sensor_count': threshold, 'keyframes': len(chunk),
                             'duration_s': (chunk[-1]['timestamp_us'] - chunk[0]['timestamp_us']) / 1e6,
                             'start_sample': chunk[0]['sample_token'], 'end_sample': chunk[-1]['sample_token'],
                             'category': chunk[0]['category']})
    return sorted(episodes, key=lambda e: (e['keyframes'], e['duration_s']), reverse=True)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(dataroot, output, max_range=60.0):
    started = time.monotonic()
    root, out = Path(dataroot), Path(output)
    out.mkdir(parents=True, exist_ok=True)
    nusc = NuScenes(version='v1.0-mini', dataroot=str(root), verbose=True)
    # Build chronological streams from ALL sample_data, including non-keyframes.
    streams = defaultdict(list)
    for sd in nusc.sample_data:
        if sd['channel'] in CHANNELS:
            scene_token = nusc.get('sample', sd['sample_token'])['scene_token']
            streams[(scene_token, sd['channel'])].append(sd)
    for records in streams.values():
        records.sort(key=lambda r: r['timestamp'])
    times = {key: [r['timestamp'] for r in records] for key, records in streams.items()}
    missing = [sd['filename'] for records in streams.values() for sd in records if not (root / sd['filename']).is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} radar files missing, e.g. {missing[:3]}')

    @lru_cache(maxsize=128)
    def load(sd_token):
        sd = nusc.get('sample_data', sd_token)
        # Explicit devkit defaults; do not silently enable all invalid returns.
        pc = RadarPointCloud.from_file(str(root / sd['filename']), invalid_states=[0], dynprop_states=list(range(7)), ambig_states=[3])
        points = pc.points[:, np.isfinite(pc.points[:3]).all(axis=0)]
        cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        pose = nusc.get('ego_pose', sd['ego_pose_token'])
        return transform_points(points, cs, pose)

    configs = [{'id': f'age{age}_margin{margin:g}_points{minimum}', 'max_age_ms': age,
                'margin_m': margin, 'min_points': minimum}
               for age in (50, 100, 200) for margin in (0.0, 0.5) for minimum in (1, 2)]
    # Sensitivity to using offline target velocity to align box centers.
    configs.append({'id': 'age100_margin0.5_points1_static_boxes', 'max_age_ms': 100,
                    'margin_m': 0.5, 'min_points': 1, 'static_boxes': True})
    primary_id = 'age100_margin0.5_points1'
    rows, sweep_rows = [], []
    best = None
    for scene in nusc.scene:
        token, frame = scene['first_sample_token'], 0
        while token:
            sample = nusc.get('sample', token)
            t = sample['timestamp']
            lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            ego = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
            # All boxes participate in ambiguity exclusion, including out-of-scope categories.
            anns = [nusc.get('sample_annotation', a) for a in sample['anns']]
            velocities = [nusc.box_velocity(a['token']) for a in anns]
            selected = {}
            for channel in CHANNELS:
                key = (scene['token'], channel)
                idx = latest_causal(times.get(key, []), t, 200_000)
                if idx is None:
                    sweep_rows.append({'scene': scene['name'], 'sample_token': token, 'sensor': channel,
                                       'timestamp_us': t, 'age_ms': None, 'point_count': 0, 'sample_data_token': ''})
                    continue
                sd = streams[key][idx]
                assert sd['timestamp'] <= t
                age = (t - sd['timestamp']) / 1000
                points = load(sd['token'])
                counts = {}
                for static in (False, True):
                    shifted = []
                    for a, v in zip(anns, velocities):
                        center = np.asarray(a['translation'], dtype=float).copy()
                        if not static and np.isfinite(v[:2]).all():
                            center[:2] -= v[:2] * age / 1000
                        shifted.append({**a, 'translation': center.tolist()})
                    for margin in (0.0, 0.5):
                        counts[(margin, static)] = exclusive_counts(box_membership(points, shifted, margin))
                selected[channel] = (sd, age, points, counts)
                sweep_rows.append({'scene': scene['name'], 'sample_token': token, 'sensor': channel,
                                   'timestamp_us': t, 'age_ms': age, 'point_count': points.shape[1],
                                   'sample_data_token': sd['token'],
                                   'ambiguous_points_margin0.5': counts[(0.5, False)][1]})
            for j, (ann, velocity) in enumerate(zip(anns, velocities)):
                category = ann['category_name']
                eligible_class = (category.startswith('vehicle.') and category != 'vehicle.bicycle_rack') or category.startswith('human.pedestrian.')
                distance = float(np.linalg.norm(np.asarray(ann['translation'])[:2] - np.asarray(ego['translation'])[:2]))
                if not eligible_class or distance > max_range:
                    continue
                speed = float(np.linalg.norm(velocity[:2])) if np.isfinite(velocity[:2]).all() else None
                for config in configs:
                    support = {}
                    for channel, (_, age, _, counts) in selected.items():
                        support[channel] = int(counts[(config['margin_m'], config.get('static_boxes', False))][0][j]) if age <= config['max_age_ms'] else 0
                    sensors = [channel for channel in CHANNELS if support.get(channel, 0) >= config['min_points']]
                    row = {'config': config['id'], 'scene': scene['name'], 'scene_token': scene['token'],
                           'sample_token': token, 'timestamp_us': t, 'frame_index': frame,
                           'annotation_token': ann['token'], 'instance_token': ann['instance_token'],
                           'category': category, 'range_m': distance, 'speed_mps': speed,
                           'velocity_available': speed is not None, 'sensor_count': len(sensors), 'sensors': sensors,
                           **{f'points_{c}': support.get(c, 0) for c in CHANNELS}}
                    rows.append(row)
                    if config['id'] == primary_id and (best is None or len(sensors) > best['count']):
                        best = {'count': len(sensors), 'sample': sample, 'ann': ann, 'selected': selected,
                                'ego': ego, 'scene': scene['name'], 'row': row}
            token, frame = sample['next'], frame + 1
        print(f'Audited {scene["name"]}: {frame} keyframes', flush=True)

    df = pd.DataFrame(rows)
    serial = df.copy()
    serial['sensors'] = serial['sensors'].apply(lambda x: '|'.join(x))
    serial.to_csv(out / 'object_coverage.csv', index=False)
    pd.DataFrame(sweep_rows).to_csv(out / 'sweep_inventory.csv', index=False)
    summaries = []
    for config in configs:
        subset = df[df.config == config['id']]
        hist = subset.sensor_count.value_counts().to_dict()
        summaries.append({**config, 'object_keyframes': len(subset),
                          **{f'sensors_{i}': int(hist.get(i, 0)) for i in range(6)},
                          'at_least_2': int((subset.sensor_count >= 2).sum()),
                          'at_least_3': int((subset.sensor_count >= 3).sum())})
    pd.DataFrame(summaries).to_csv(out / 'sensitivity.csv', index=False)
    primary = df[df.config == primary_id]
    episodes = sustained_episodes(primary.to_dict('records'), 2) + sustained_episodes(primary.to_dict('records'), 3)
    episode_cols = ['scene','instance_token','sensors','sensor_count','keyframes','duration_s','start_sample','end_sample','category']
    ep = pd.DataFrame(episodes, columns=episode_cols)
    ep.to_csv(out / 'overlap_episodes.csv', index=False)
    candidates = ep[(ep.keyframes >= 3) & (ep.duration_s >= 0.9)]
    candidates.to_csv(out / 'candidate_episodes.csv', index=False)
    for grouping, filename in [('scene', 'per_scene.csv'), ('category', 'per_category.csv')]:
        group_rows = []
        for label, g in primary.groupby(grouping):
            group_rows.append({grouping: label, 'object_keyframes': len(g), 'observed': int((g.sensor_count >= 1).sum()),
                               'at_least_2': int((g.sensor_count >= 2).sum()), 'at_least_3': int((g.sensor_count >= 3).sum()),
                               'unique_instances_2plus': g[g.sensor_count >= 2].instance_token.nunique()})
        pd.DataFrame(group_rows).to_csv(out / filename, index=False)
    pair_counts = Counter()
    for sensors in primary.sensors:
        pair_counts.update(itertools.combinations(sensors, 2))
    pd.DataFrame([{'sensor_a': a, 'sensor_b': b, 'object_keyframes': n} for (a,b),n in pair_counts.items()]).to_csv(out / 'sensor_pairs.csv', index=False)
    moving = primary[primary.speed_mps >= 0.5]
    manifest = {'dataset': 'nuScenes v1.0-mini', 'scenes': len(nusc.scene), 'keyframes': len(nusc.sample),
                'radar_files': sum(map(len, streams.values())), 'primary': next(s for s in summaries if s['id'] == primary_id),
                'unique_objects_with_2plus': int(primary[primary.sensor_count >= 2].instance_token.nunique()),
                'unique_objects_with_3plus': int(primary[primary.sensor_count >= 3].instance_token.nunique()),
                'candidate_pair_episodes': int((candidates.sensor_count == 2).sum()),
                'candidate_triple_episodes': int((candidates.sensor_count == 3).sum()),
                'candidate_pair_scenes': int(candidates[candidates.sensor_count == 2].scene.nunique()),
                'candidate_triple_scenes': int(candidates[candidates.sensor_count == 3].scene.nunique()),
                'moving_object_keyframes': len(moving), 'moving_2plus': int((moving.sensor_count >= 2).sum()),
                'moving_3plus': int((moving.sensor_count >= 3).sum()),
                'unknown_velocity_object_keyframes': int(primary.speed_mps.isna().sum()),
                'parameters': {'max_range_m': max_range, 'minimum_episode_keyframes': 3, 'minimum_episode_duration_s': 0.9,
                               'radar_filters': {'invalid_states': [0], 'dynprop_states': list(range(7)), 'ambig_states': [3]},
                               'matching': 'BEV oriented box, exclusive points, latest causal sweep per sensor',
                               'target_alignment': 'offline annotation velocity backward translation; static fallback if unavailable'},
                'elapsed_s': time.monotonic() - started, 'python': sys.version, 'platform': platform.platform(),
                'packages': {p: importlib.metadata.version(p) for p in ['nuscenes-devkit','numpy','pandas','matplotlib','pyquaternion']},
                'metadata_sha256': {p.name: file_hash(p) for p in sorted((root/'v1.0-mini').glob('*.json'))},
                'code_sha256': file_hash(__file__), 'scene_tokens': [s['token'] for s in nusc.scene],
                'limitations': ['Offline annotation-assisted coverage proxy, not a detector or exact point identity.',
                                'Mini is a feasibility pilot, not training or security validation.',
                                'Velocity can use future annotations for offline matching only.',
                                'BEV padding can overcount clutter; strict/two-point sensitivities are provided.',
                                'Observed overlap is not a guarantee of independence, visibility, or recoverability.',
                                'Episodes are keyframe observations; continuity between keyframes is not established.']}
    prep = root / 'preparation_manifest.json'
    if prep.exists():
        manifest['archive'] = json.loads(prep.read_text(encoding='utf-8'))['archive']
    (out / 'summary.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    with (out / 'environment_freeze.txt').open('w', encoding='utf-8') as f:
        # importlib avoids reliance on pip being installed in a uv-created venv.
        f.write('\n'.join(sorted(f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions())))
    plots(primary, pd.DataFrame(summaries), best, out)
    write_report(manifest, summaries, candidates, out)
    print(json.dumps({k:v for k,v in manifest.items() if k in ('primary','candidate_pair_episodes','candidate_triple_episodes','moving_2plus','moving_3plus')}, indent=2))
    return manifest


def plots(primary, sensitivity, best, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    counts = primary.sensor_count.value_counts().reindex(range(6), fill_value=0)
    axes[0].bar(range(6), counts.values, color='#277da1')
    axes[0].set(xlabel='Distinct supporting radars', ylabel='Annotated object-keyframes', title='Observed radar support (primary settings)', xticks=range(6))
    for x,y in enumerate(counts.values):
        axes[0].text(x,y,str(y),ha='center',va='bottom',fontsize=9)
    subset = sensitivity[(sensitivity.max_age_ms == 100) & ~sensitivity.id.str.contains('static')]
    x = np.arange(len(subset))
    axes[1].bar(x-.18,subset.at_least_2,width=.36,label='2+ radars',color='#277da1')
    axes[1].bar(x+.18,subset.at_least_3,width=.36,label='3+ radars',color='#f9844a')
    axes[1].set(xticks=x, xticklabels=[f'{r.margin_m:g} m / {r.min_points} pt' for r in subset.itertuples()], ylabel='Object-keyframes',title='Matching sensitivity at 100 ms')
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out / 'coverage.png', dpi=160)
    plt.close(fig)
    if best:
        fig, ax = plt.subplots(figsize=(8,7))
        ego = np.asarray(best['ego']['translation'])
        for channel, (_, age, points, _) in best['selected'].items():
            if age <= 100:
                ax.scatter(points[0]-ego[0], points[1]-ego[1],s=9,label=f'{channel} ({age:.0f} ms)')
        ann = best['ann']
        yaw = Quaternion(ann['rotation']).yaw_pitch_roll[0]
        width,length,_ = ann['size']
        corners = np.array([[length/2,width/2],[length/2,-width/2],[-length/2,-width/2],[-length/2,width/2],[length/2,width/2]])
        rot = np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
        corners = corners @ rot.T + np.asarray(ann['translation'])[:2] - ego[:2]
        ax.plot(corners[:,0],corners[:,1],'k-',linewidth=2,label='Selected annotation (unpadded)')
        ax.scatter([0],[0],marker='*',s=120,c='black',label='Ego position')
        ax.set(xlim=(-65,65),ylim=(-65,65),xlabel='Global x relative to ego (m)',ylabel='Global y relative to ego (m)',title=f'{best["scene"]}: candidate with {best["count"]} supporting radars\nOffline geometric match; latest causal sweeps')
        ax.set_aspect('equal')
        ax.legend(fontsize=7,loc='upper left')
        fig.tight_layout()
        fig.savefig(out / 'example_overlap.png',dpi=160)
        plt.close(fig)
        (out / 'example_manifest.json').write_text(json.dumps({**best['row'], 'sweep_tokens': {c:v[0]['token'] for c,v in best['selected'].items()}},indent=2),encoding='utf-8')


def write_report(m, summaries, candidates, out):
    p=m['primary']; n=p['object_keyframes']; observed=n-p['sensors_0']
    lines=['# nuScenes mini radar coverage audit','',
           'This is a measured offline feasibility audit, not attack detection or mitigation validation.','',
           f"Loaded {m['scenes']} scenes, {m['keyframes']} keyframes and {m['radar_files']} radar files.",
           f"Primary denominator: {n} vehicle/pedestrian object-keyframes within 60 m of ego; {observed} have radar support.",
           '', '| Supporting sensors | Object-keyframes |','|---|---:|']
    lines += [f'| {i} | {p[f"sensors_{i}"]} |' for i in range(6)]
    lines += ['',f"Two-plus support: {p['at_least_2']}/{n} ({p['at_least_2']/max(n,1):.2%} of eligible; {p['at_least_2']/max(observed,1):.2%} of radar-observed).",
              f"Three-plus support: {p['at_least_3']}/{n} ({p['at_least_3']/max(n,1):.2%} of eligible).",'',
              f"Fixed-pair candidate episodes (>=3 consecutive keyframes, >=0.9 s endpoint span): {m['candidate_pair_episodes']} in {m['candidate_pair_scenes']} scenes.",
              f"Fixed-triple candidate episodes under the same rule: {m['candidate_triple_episodes']} in {m['candidate_triple_scenes']} scenes.",
              f"Moving objects (offline speed >=0.5 m/s): {m['moving_object_keyframes']} object-keyframes; {m['moving_2plus']} with 2+ support; {m['moving_3plus']} with 3+ support.",'',
              '## Interpretation','',
              'Pair overlap supplies candidate sequences for two-stream corruption. A fixed triple may supply two corrupted streams plus a third observing stream; it does not guarantee the third is independent or sufficient to identify truth.',
              'A few candidate episodes justify a pilot only. Sustained support across independent scenes and held-out data is needed before a publication-scale claim. Inspect candidates visually and rerun stricter matching before selecting attacks.','',
              '## Sensitivity','', '| Configuration | Eligible | 2+ | 3+ |','|---|---:|---:|---:|']
    lines += [f"| {s['id']} | {s['object_keyframes']} | {s['at_least_2']} | {s['at_least_3']} |" for s in summaries]
    lines += ['', '## Method and limitations',''] + ['- '+s for s in m['limitations']]
    lines += ['- Primary: latest sweep at or before annotation timestamp, age <=100 ms; >=1 valid return in an oriented BEV box padded by 0.5 m per side.',
              '- A point in multiple annotated boxes is excluded from every box. No stacked multisweep cloud is treated as simultaneous evidence.',
              '- Annotations and annotation-derived velocity are confined to this offline audit. They must never become online detector inputs.',
              '- Official default radar validity filters are explicit in summary.json; unfiltered radar results are not claimed.',
              '- Ground-truth category/range restriction determines the audit denominator; lack of support is not proof of lack of field of view.',
              '', 'Source: https://github.com/nutonomy/nuscenes-devkit and https://www.nuscenes.org/nuscenes',
              'Reproduce with scripts/audit_radar_overlap.py. CSVs, hashes, package versions and parameters accompany this report.']
    (out / 'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--dataroot',default='data/nuscenes')
    p.add_argument('--output',default='outputs/mini_audit')
    args=p.parse_args()
    run(args.dataroot,args.output)
