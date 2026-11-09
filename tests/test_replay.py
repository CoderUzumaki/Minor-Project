"""Known-answer checks for causality, uncertainty, labels and metric accounting."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from replay_core import Config, LocalTracker, fuse_tracks, predict, replay
from run_replay import frame_metrics, freeze_splits, inject


def event(t, x=0., sensor='A'):
    return dict(time_s=t, sensor=sensor, reports=[dict(id=f'{sensor}:{t}', position=[x, 0.], covariance=[[1., 0.], [0., 1.]])])


def test_predict_uncertainty_and_backwards_time():
    x, p = predict(np.array([0., 0., 2., 0.]), np.eye(4), .5, 3.)
    assert x[0] == 1.
    assert p[0, 0] > 1.
    assert np.linalg.eigvalsh(p).min() > 0
    with pytest.raises(ValueError):
        predict(x, p, -.1, 3.)


def test_prefix_invariance_and_no_future_reports():
    events = [event(i / 10, i / 10) for i in range(20)]
    prefix = replay(events[:10], [.3, .9])
    full = replay(events, [.3, .9, 1.9])
    assert prefix['frames'] == full['frames'][:2]
    assert all(r['time_s'] <= .9 for r in prefix['decisions'])


def test_duplicate_and_unordered_sweeps_rejected():
    with pytest.raises(ValueError):
        replay([event(0.), event(0.)], [0.])
    with pytest.raises(ValueError):
        replay([event(1.), event(0.)], [1.])


def test_protection_rejects_jump_and_stale_track_expires():
    events = [event(i / 10) for i in range(20)] + [event(2., 5.)]
    raw = replay(events, [1.9, 2.])
    protected = replay(events, [1.9, 2., 3.], True)
    assert protected['decisions'][-1]['alert']
    assert protected['frames'][1]['tracks'][0]['state'][0] == 0.
    assert raw['frames'][1]['tracks'][0]['state'][0] > 0.
    assert not protected['frames'][-1]['tracks']


def test_unmatched_report_remains_unknown_and_visible():
    r = replay([event(0), event(.1, 100)], [.1])
    assert r['decisions'][-1]['status'] == 'unknown_new'
    assert not r['decisions'][-1]['alert']


def test_invalid_covariance_rejected():
    e = event(0)
    e['reports'][0]['covariance'] = [[-1, 0], [0, 1]]
    with pytest.raises(ValueError):
        replay([e], [0])


def test_no_observation_does_not_imply_attack():
    tracker = LocalTracker('A')
    tracker.update(event(0))
    trust = tracker.tracks[0].trust
    assert tracker.update(dict(time_s=.1, sensor='A', reports=[])) == []
    assert tracker.tracks[0].trust == trust


def test_ci_does_not_double_identical_evidence_precision():
    def track(sensor):
        return dict(id=sensor, sensor=sensor, state=[0, 0, 1, 0], covariance=np.eye(4).tolist(), age_s=0, suspect=False, trust=1)
    fused = fuse_tracks([track('A'), track('B')])
    assert len(fused) == 1
    np.testing.assert_allclose(fused[0]['covariance'], np.eye(4))
    assert len(fuse_tracks([track('A'), track('A')])) == 2


def test_metrics_count_unmatched_and_empty_cases():
    assert frame_metrics([], [])['gospa'] == 0
    assert frame_metrics([], [[0, 0]])['missed_targets'] == 1
    assert frame_metrics([[100, 0]], [[0, 0]])['gospa'] == 10
    r = frame_metrics([[3, 0], [100, 0]], [[0, 0]])
    assert r['squared_error_sum'] == 9
    assert r['false_tracks'] == 1 and r['missed_targets'] == 0
    assert r['gospa'] == pytest.approx(np.sqrt(59))


def test_injector_is_repeatable_does_not_mutate_or_leak_labels():
    events = [event(0), event(.1), event(.1, sensor='B'), event(.1, sensor='C')]
    ref = [dict(time_s=0, objects=[dict(instance='target', position=[0, 0], yaw=0, size=[2, 4, 2])])]
    attack = dict(start_s=.05, end_s=1, sensors=['A', 'B'], target_instance='target', displacement_m=[4, 0], ramp_s=0)
    altered, labels = inject(events, ref, attack)
    assert altered == inject(events, ref, attack)[0]
    assert events[1]['reports'][0]['position'] == [0, 0]
    assert altered[1]['reports'][0]['position'] == [4, 0]
    assert [r['attacked'] for r in labels] == [False, True, True, False]
    assert 'attacked' not in json.dumps(altered) and 'target' not in json.dumps(altered)
    assert replay(altered, [0, .1])['frames'] == replay(json.loads(json.dumps(altered)), [0, .1])['frames']


def test_split_manifest_groups_recordings_and_refuses_changes(tmp_path):
    class Dataset:
        scene = [dict(name=f's{i}', token=f't{i}', log_token=f'l{i // 2}') for i in range(10)]
    path = tmp_path / 'splits.json'
    manifest = freeze_splits(Dataset(), path)
    for log in {r['log_token'] for r in manifest['scenes']}:
        assert len({r['split'] for r in manifest['scenes'] if r['log_token'] == log}) == 1
    assert freeze_splits(Dataset(), path) == manifest
    path.write_text('{}')
    with pytest.raises(ValueError):
        freeze_splits(Dataset(), path)
