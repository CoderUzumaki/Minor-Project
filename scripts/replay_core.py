"""Annotation-free causal tracking and conservative fusion for report replay.

This module intentionally has no nuScenes, ground-truth, or attack-label API.
Covariances and thresholds are declared pilot heuristics, not calibrated values.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Config:
    acceleration_std: float = 3.0
    association_gate_m: float = 12.0
    association_nis: float = 36.0
    alert_nis: float = 9.21
    cusum_drift: float = 2.5
    cusum_threshold: float = 10.0
    max_coast_s: float = 0.8
    confirmation_hits: int = 3
    fusion_gate_m: float = 3.0
    max_sensor_age_s: float = 0.2


@dataclass
class Track:
    id: str
    x: np.ndarray
    p: np.ndarray
    timestamp: float
    last_update: float
    hits: int = 1
    cusum: float = 0.0
    trust: float = 0.5
    suspect: bool = False


def predict(x, p, dt, acceleration_std):
    if dt < -1e-8:
        raise ValueError('Cannot predict backwards in time')
    dt = max(0.0, dt)
    f = np.eye(4)
    f[0, 2] = f[1, 3] = dt
    g = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
    return f @ x, f @ p @ f.T + acceleration_std ** 2 * g @ g.T


def innovation(track, report):
    z, r = np.asarray(report['position'], float), np.asarray(report['covariance'], float)
    residual = z - track.x[:2]
    s = track.p[:2, :2] + r
    return residual, s, float(residual @ np.linalg.solve(s, residual))


class LocalTracker:
    def __init__(self, sensor, protected=False, config=None):
        self.sensor = sensor
        self.protected = protected
        self.config = config or Config()
        self.tracks = []
        self.counter = 0
        self.last_timestamp = -np.inf

    def update(self, event):
        t = event['time_s']
        if t <= self.last_timestamp:
            raise ValueError('Each sensor sweep must be consumed once in increasing time order')
        self.last_timestamp = t
        cfg = self.config
        reports = event['reports']
        for report in reports:
            z, r = np.asarray(report['position']), np.asarray(report['covariance'])
            if z.shape != (2,) or r.shape != (2, 2) or not np.isfinite(z).all() or not np.isfinite(r).all():
                raise ValueError('Malformed report')
            if not np.allclose(r, r.T) or np.linalg.eigvalsh(r).min() <= 0:
                raise ValueError('Report covariance must be positive definite')
        self.tracks = [a for a in self.tracks if t - a.last_update <= cfg.max_coast_s]
        for a in self.tracks:
            a.x, a.p = predict(a.x, a.p, t - a.timestamp, cfg.acceleration_std)
            a.timestamp = t
        costs = np.full((len(self.tracks), len(reports)), 1e9)
        for i, a in enumerate(self.tracks):
            for j, r in enumerate(reports):
                residual, _, nis = innovation(a, r)
                if np.linalg.norm(residual) <= cfg.association_gate_m and nis <= cfg.association_nis:
                    costs[i, j] = nis
        matches = {}
        if costs.size:
            ii, jj = linear_sum_assignment(costs)
            matches = {int(j): int(i) for i, j in zip(ii, jj) if costs[i, j] < 1e9}
        decisions = []
        for j, r in enumerate(reports):
            if j not in matches:
                # Unmatched reports remain visible and monitored; no unsupported
                # accusation or trust reduction merely because a report is new.
                self.counter += 1
                a = Track(f'{self.sensor}:{self.counter}', np.r_[r['position'], 0., 0.],
                          np.diag([4., 4., 100., 100.]), t, t)
                a.p[:2, :2] = np.asarray(r['covariance'])
                self.tracks.append(a)
                decisions.append(dict(report_id=r['id'], track_id=a.id, nis=None,
                                      score=0., alert=False, status='unknown_new', trust=a.trust))
                continue
            a = self.tracks[matches[j]]
            residual, s, nis = innovation(a, r)
            # Two-dimensional innovations have expectation 2 only under a
            # correctly specified model. These pilot settings need calibration.
            a.cusum = max(0., a.cusum + nis - cfg.cusum_drift)
            established = a.hits >= cfg.confirmation_hits
            alert = established and (nis > cfg.alert_nis or a.cusum > cfg.cusum_threshold)
            score = max(nis / cfg.alert_nis, a.cusum / cfg.cusum_threshold) if established else 0.
            quality = 0. if alert else 1.
            alpha = 0.5 if alert else 0.08
            a.trust = (1 - alpha) * a.trust + alpha * quality
            a.suspect = alert
            if not (self.protected and alert):
                k = np.linalg.solve(s, a.p[:, :2].T).T
                a.x = a.x + k @ residual
                kh = np.zeros((4, 4))
                kh[:, :2] = k
                identity = np.eye(4) - kh
                a.p = identity @ a.p @ identity.T + k @ np.asarray(r['covariance']) @ k.T
                a.last_update = t
                a.hits += 1
            decisions.append(dict(report_id=r['id'], track_id=a.id, nis=nis, score=score,
                                  alert=bool(alert), status='suspected_corruption' if alert else
                                  ('accepted' if established else 'tentative'), trust=a.trust))
        return decisions

    def snapshot(self, t):
        result = []
        for a in self.tracks:
            age = t - a.last_update
            if age > self.config.max_coast_s or a.hits < self.config.confirmation_hits:
                continue
            x, p = predict(a.x, a.p, t - a.timestamp, self.config.acceleration_std)
            result.append(dict(id=a.id, sensor=self.sensor, state=x.tolist(), covariance=p.tolist(),
                               age_s=age, trust=a.trust, suspect=a.suspect,
                               status='coasting' if age > self.config.max_sensor_age_s else 'observed'))
        return result


def fuse_tracks(tracks, config=None, protected=False):
    """Complete-link spatial grouping, one report per sensor, equal-weight CI.

    Equal weights are fixed to avoid tuning on evaluation labels. No majority or
    independence guarantee is made. Unassociated hypotheses remain in the output.
    """
    cfg = config or Config()
    groups = []
    for track in sorted(tracks, key=lambda r: r['id']):
        target = None
        for group in groups:
            if track['sensor'] in {r['sensor'] for r in group}:
                continue
            if all(np.linalg.norm(np.array(track['state'][:2]) - r['state'][:2]) <= cfg.fusion_gate_m for r in group):
                target = group
                break
        if target is None:
            groups.append([track])
        else:
            target.append(track)
    fused = []
    for group in groups:
        infos = [np.linalg.inv(np.asarray(r['covariance']) / max(r['trust'], .1)
                               if protected else np.asarray(r['covariance'])) for r in group]
        precision = sum(infos) / len(infos)
        p = np.linalg.inv(precision)
        x = p @ (sum(info @ r['state'] for info, r in zip(infos, group)) / len(group))
        observed = sum(r['age_s'] <= cfg.max_sensor_age_s and not r['suspect'] for r in group)
        fused.append(dict(state=x.tolist(), covariance=p.tolist(), members=[r['id'] for r in group],
                          support=len(group), observed_support=observed,
                          status='degraded' if observed == 0 else 'single_source' if observed == 1
                          else 'multi_source_unverified'))
    return fused


def replay(events, frame_times, protected=False, config=None):
    """Only events and timestamps are accepted, never annotations or labels."""
    cfg = config or Config()
    if any(events[i]['time_s'] > events[i + 1]['time_s'] for i in range(len(events) - 1)):
        raise ValueError('Replay events must be globally time ordered')
    if any(frame_times[i] >= frame_times[i + 1] for i in range(len(frame_times) - 1)):
        raise ValueError('Frame times must increase')
    trackers, frames, decisions, latency = {}, [], [], []
    cursor = 0
    for t in frame_times:
        while cursor < len(events) and events[cursor]['time_s'] <= t:
            e = events[cursor]
            start = time.perf_counter()
            tracker = trackers.setdefault(e['sensor'], LocalTracker(e['sensor'], protected, cfg))
            for row in tracker.update(e):
                decisions.append(dict(time_s=e['time_s'], sensor=e['sensor'], **row))
            latency.append((time.perf_counter() - start) * 1000)
            cursor += 1
        tracks = [r for tracker in trackers.values() for r in tracker.snapshot(t)]
        frames.append(dict(time_s=t, tracks=tracks, fused=fuse_tracks(tracks, cfg, protected)))
    return dict(frames=frames, decisions=decisions,
                timing=dict(event_count=cursor, local_update_ms_p50=float(np.percentile(latency, 50)) if latency else 0.,
                            local_update_ms_p95=float(np.percentile(latency, 95)) if latency else 0.))
