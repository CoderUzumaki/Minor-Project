"""Render saved experimental output as an offline, self-contained replay display."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def render(out, results, summaries, metrics, references, attack, provenance):
    out = Path(out)
    names = ['clean_baseline', 'attacked_baseline', 'attacked_protected']
    colors = ['#38bdf8', '#fb7185', '#34d399']
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for name, color in zip([*names, 'clean_protected'], [*colors, '#a78bfa']):
        rows = metrics[name]
        axes[0].plot([r['time_s'] for r in rows], [r['gospa'] for r in rows], label=name.replace('_', ' '), color=color)
        axes[1].plot([r['time_s'] for r in rows], [r['false_tracks'] + r['missed_targets'] for r in rows], color=color)
    for ax in axes:
        ax.axvspan(attack['start_s'], attack['end_s'], color='orange', alpha=.15)
        ax.grid(alpha=.2)
    axes[0].set_ylabel('GOSPA (p=2, c=10m, alpha=2)')
    axes[1].set_ylabel('False + missed hypotheses')
    axes[1].set_xlabel('Scene time (s); shaded interval = injected attack')
    axes[0].legend()
    fig.suptitle(f'{provenance["scene"]} — measured training-scene pilot, heuristic baseline')
    fig.tight_layout()
    fig.savefig(out / 'tracking_comparison.png', dpi=150)
    plt.close(fig)
    report = ['# Causal radar replay pilot', '', f'Scene: `{provenance["scene"]}`. Split: **train / development only**. Run: `{provenance["run_id"]}`.', '',
              'Measured report-level simulation on recorded radar data. No learned model or RF validation.', '',
              '| Branch | Mean GOSPA ↓ | Matched RMSE (m) | False track-frames | Missed target-frames |',
              '|---|---:|---:|---:|---:|']
    for name, s in summaries.items():
        rmse = s['matched_position_rmse_m']
        report.append(f'| {name} | {s["mean_gospa"]:.3f} | {rmse:.3f} | {s["false_track_frames"]} | {s["missed_target_frames"]} |' if rmse is not None else f'| {name} | {s["mean_gospa"]:.3f} | n/a | {s["false_track_frames"]} | {s["missed_target_frames"]} |')
    report += ['', 'RMSE includes only matches within 4m and must be read with false/missed counts. GOSPA includes all emitted hypotheses and eligible annotations; reflection centroids are not object centers.', '',
               '## Detector measurements', '', '| Branch | TP | FP | FN | Precision | Recall | First event delay (s) |', '|---|---:|---:|---:|---:|---:|---:|']
    fmt = lambda x: 'n/a' if x is None else f'{x:.3f}'
    for name, s in summaries.items():
        d = s['detection']
        report.append(f'| {name} | {d["tp"]} | {d["fp"]} | {d["fn"]} | {fmt(d["precision"])} | {fmt(d["recall"])} | {fmt(d["event_delay_s"])} |')
    report += ['', 'The baseline branches run the detector in monitor-only mode. Protected branches reject flagged updates and inflate covariance using heuristic temporal trust. No detection means a null delay, never zero.', '',
               'False-alert counts are report decisions, not independent attack events. All rows come from one development scene and one injection; there is no held-out performance claim.', '',
               'Compare each attacked branch with its own clean branch. Protection also changes clean tracking, so a lower protected GOSPA alone is not evidence of successful attack mitigation.', '',
               '## Limits and next work', ''] + ['- ' + text for text in provenance['limitations']]
    report += ['', 'Next: inspect episodes and clutter, calibrate frontend/residual thresholds on training/validation recordings, add benign stresses and additional attack families, then compare GRU and temporal graph detection. See HANDOFF.md.', '',
               'Open `demo.html` for playback, scrub controls, reference overlays, and clean/attacked/protected comparisons. All display annotations are offline overlays; inference cannot access them.']
    (out / 'REPORT.md').write_text('\n'.join(report), encoding='utf-8')
    raw = {}
    for name in ('clean', 'attacked'):
        events = json.loads((out / f'inputs/{name}.json').read_text())['events']
        latest, cursor, snapshots = {}, 0, []
        for ref in references:
            while cursor < len(events) and events[cursor]['time_s'] <= ref['time_s']:
                e = events[cursor]
                latest[e['sensor']] = e
                cursor += 1
            snapshots.append([r['position'] for e in latest.values() if ref['time_s'] - e['time_s'] <= .2 for r in e['reports']])
        raw[name] = snapshots
    compact = {name: [dict(fused=f['fused'], tracks=[dict(state=t['state'], trust=t['trust'], sensor=t['sensor']) for t in f['tracks']])
                      for f in results[name]['frames']] for name in names}
    payload = dict(scene=provenance['scene'], run_id=provenance['run_id'], attack=attack, summaries=summaries, references=references,
                   results=compact, raw=raw, metrics=metrics, sources=provenance['sources'])
    html = TEMPLATE.replace('__DATA__', json.dumps(payload, allow_nan=False).replace('</', '<\\/'))
    (out / 'demo.html').write_text(html, encoding='utf-8')


TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Radar Trust Lab — Recorded Replay</title>
<style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#09111e;color:#e2e8f0}body{margin:0;padding:28px;max-width:1600px;margin:auto}h1{font-size:30px;margin:8px 0}h2{font-size:17px;margin:0 0 8px}p{color:#aebbd0;line-height:1.6}.eyebrow{color:#38bdf8;font-size:12px;letter-spacing:.15em;text-transform:uppercase}.badge{display:inline-block;background:#20314a;padding:5px 10px;border-radius:20px;font-size:12px;margin-right:6px}.controls{display:flex;gap:18px;align-items:center;flex-wrap:wrap;padding:18px;background:#121f32;border-radius:12px;margin:20px 0}button,select{background:#1d3652;color:white;border:1px solid #466382;border-radius:6px;padding:9px;cursor:pointer}input[type=range]{flex:1;min-width:160px;accent-color:#38bdf8}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}.card{background:#111e30;border:1px solid #24354d;border-radius:12px;padding:16px;min-width:0}canvas{width:100%;height:auto;background:#0a1525;border-radius:8px}.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:13px;margin:12px 0}.note{border-left:3px solid #fbbf24;padding:10px 16px;background:#29251a;color:#f3ddb0}table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:10px;border-bottom:1px solid #24354d}.scroll{overflow-x:auto}.small{font-size:12px;color:#9fb0c7}.metric{font-size:13px;margin-top:12px;line-height:1.7}.bottom{display:grid;grid-template-columns:1.3fr 1fr;gap:16px;margin-top:18px}a{color:#7dd3fc}summary{cursor:pointer}#state{font-weight:600;color:#fbbf24}@media(max-width:1000px){.grid,.bottom{grid-template-columns:1fr}body{padding:16px}}
</style></head><body>
<div class="eyebrow">Radar Trust Lab / Research prototype</div><h1>When radar reports disagree</h1>
<p>Replay recorded driving data, introduce a shared displacement into two radar streams, and inspect a causal detector’s response.</p>
<span class="badge" id="scene"></span><span class="badge">Training-scene pilot</span><span class="badge">Heuristic residual detector · no trained GNN</span>
<div class="controls"><button id="play">▶ Play</button><button id="reset">Restart</button><input id="scrub" type="range" min="0" value="0" aria-label="Replay frame"><span id="time"></span><select id="speed" aria-label="Playback speed"><option value="1">1× speed</option><option value="2">2× speed</option><option value="0.5">0.5× speed</option></select><label><input type="checkbox" id="focus" checked> Focus target</label><label><input type="checkbox" id="truth" checked> Offline reference</label></div>
<div id="state" aria-live="polite"></div><div class="legend"><span style="color:#94a3b8">· Radar report centroids</span><span style="color:#e2e8f0">□ Offline annotated boxes</span><span style="color:#fbbf24">◇ Selected target</span><span>○ Fused estimates + 1σ position ellipses</span></div>
<div class="grid"><section class="card"><h2 style="color:#38bdf8">1 / Clean baseline</h2><div class="small">Recorded input · monitoring only</div><canvas id="clean" width="520" height="420"></canvas><div class="metric" id="m0"></div></section><section class="card"><h2 style="color:#fb7185">2 / Attacked baseline</h2><div class="small">Modified input · monitoring only</div><canvas id="attacked" width="520" height="420"></canvas><div class="metric" id="m1"></div></section><section class="card"><h2 style="color:#34d399">3 / Protected replay</h2><div class="small">Same modified input · reject + coast + trust inflation</div><canvas id="protected" width="520" height="420"></canvas><div class="metric" id="m2"></div></section></div>
<div class="bottom"><section class="card"><h2>Tracking error through the replay</h2><canvas id="chart" width="850" height="240"></canvas><p class="small">GOSPA includes position error, false tracks and missed targets. Lower is better. Shading marks the injection interval. The white cursor follows playback.</p></section><section class="card"><h2>What the system knows</h2><p id="attack"></p><p class="note">Agreement does not prove truth. Two coherently altered streams can evade this detector. New or unmatched reports remain unknown; missing evidence is not proof of an attack.</p><p class="small">Trust is a heuristic score, not a calibrated probability. Reference boxes, target selection and attack status are added only by this offline display. Focus mode also uses the offline target position.</p></section></div>
<section class="card" style="margin-top:18px"><h2>Measured results — including the clean-data cost</h2><div class="scroll"><table><thead><tr><th>Branch</th><th>Mean GOSPA ↓</th><th>Matched RMSE (m)</th><th>False / missed track-frames</th><th>Detection precision / recall</th><th>False alert reports/min</th></tr></thead><tbody id="metrics"></tbody></table></div><p class="small">One development scene and one injected event. These are not held-out accuracy results. RMSE uses 4m gated matches; read it alongside false/missed counts. All branches use the same frontend and fusion rule.</p></section>
<details style="margin-top:18px"><summary>Method, reproducibility and presentation notes</summary><p>Five causal radar streams → position clustering → local Kalman trackers → innovation/CUSUM monitoring → optional rejection and trust inflation → equal-weight covariance intersection. A stale track coasts for at most 0.8s; no hidden reconstruction of deleted information.</p><p>Press Play, pause inside the highlighted attack interval, compare the same target across the three panels, then read the measured results. An unchanged or worse protected result is a valid observed outcome, not a display error.</p><p>Coordinate axes are east/north offsets in a translated global frame. Ellipses depict modeled 1σ position uncertainty, not a certified coverage guarantee. Background clutter and sparse returns affect the metrics.</p><p><a href="REPORT.md">Measured report</a> · <a href="summary.json">Metrics JSON</a> · <a href="provenance.json">Source hashes and parameters</a> · <a href="tracking_comparison.png">Exportable comparison figure</a></p><pre id="hashes" style="white-space:pre-wrap;font-size:11px"></pre></details>
<script>
const D=__DATA__;
const names=['clean_baseline','attacked_baseline','attacked_protected'], colors=['#38bdf8','#fb7185','#34d399'];
const canvases=['clean','attacked','protected'].map(id=>document.getElementById(id));
const $=id=>document.getElementById(id);let index=0,playing=false,elapsed=0,last=0;
const fmt=x=>x===null||x===undefined?'n/a':Number(x).toFixed(2);
$('scene').textContent=D.scene+' · run '+D.run_id;$('scrub').max=D.references.length-1;
$('attack').textContent=`Injection: ${D.attack.sensors.join(' + ')}. Shared offset [${D.attack.displacement_m.join(', ')}] m, ${fmt(D.attack.start_s)}–${fmt(D.attack.end_s)}s. Actual altered reports: ${Object.values(D.attack.actual_corrupted_report_counts).join(' + ')}.`;
$('hashes').textContent=JSON.stringify(D.sources,null,2);
for(const [name,s] of Object.entries(D.summaries)){const tr=document.createElement('tr');for(const value of [name.replaceAll('_',' '),fmt(s.mean_gospa),fmt(s.matched_position_rmse_m),`${s.false_track_frames} / ${s.missed_target_frames}`,`${fmt(s.detection.precision)} / ${fmt(s.detection.recall)}`,fmt(s.detection.false_alert_reports_per_minute)]){const td=document.createElement('td');td.textContent=value;tr.appendChild(td)}$('metrics').appendChild(tr)}
function bounds(ref){const target=ref.objects.find(o=>o.instance===D.attack.target_instance);const center=$('focus').checked&&target?target.position:ref.ego;const span=$('focus').checked?35:130;return {center,span}}
function drawPanel(canvas,k,ref){const ctx=canvas.getContext('2d'),w=canvas.width,h=canvas.height,b=bounds(ref),scale=Math.min(w,h)/b.span;
const project=p=>[w/2+(p[0]-b.center[0])*scale,h/2-(p[1]-b.center[1])*scale];ctx.clearRect(0,0,w,h);ctx.strokeStyle='#1d3048';ctx.lineWidth=1;
for(let a=-1000;a<=1000;a+=5){const x=project([a,0])[0],y=project([0,a])[1];if(x>=0&&x<=w){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke()}if(y>=0&&y<=h){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke()}}
ctx.fillStyle='#73849b';for(const p of D.raw[k===0?'clean':'attacked'][index]){const [x,y]=project(p);ctx.beginPath();ctx.arc(x,y,2,0,Math.PI*2);ctx.fill()}
if($('truth').checked)for(const o of ref.objects){if(!o.eligible)continue;const [x,y]=project(o.position);ctx.save();ctx.translate(x,y);ctx.rotate(-o.yaw);ctx.strokeStyle=o.instance===D.attack.target_instance?'#fbbf24':'#cbd5e180';ctx.lineWidth=o.instance===D.attack.target_instance?2:1;ctx.strokeRect(-o.size[1]*scale/2,-o.size[0]*scale/2,o.size[1]*scale,o.size[0]*scale);ctx.restore()}
for(const r of D.results[names[k]][index].fused){const [x,y]=project(r.state),p=r.covariance,aa=p[0][0],bb=p[0][1],cc=p[1][1],disc=Math.sqrt((aa-cc)**2+4*bb*bb),l1=Math.max(.01,(aa+cc+disc)/2),l2=Math.max(.01,(aa+cc-disc)/2);ctx.strokeStyle=colors[k]+'70';ctx.beginPath();ctx.ellipse(x,y,Math.sqrt(l1)*scale,Math.sqrt(l2)*scale,-.5*Math.atan2(2*bb,aa-cc),0,2*Math.PI);ctx.stroke();ctx.strokeStyle=colors[k];ctx.beginPath();ctx.arc(x,y,4,0,2*Math.PI);ctx.stroke();ctx.fillStyle=colors[k];ctx.font='10px system-ui';ctx.fillText(`${r.support}R`,x+6,y-5)}
ctx.fillStyle='#aebbd0';ctx.font='11px system-ui';ctx.fillText('North ↑   East →   grid: 5 m',12,h-12);
const m=D.metrics[names[k]][index],tracks=D.results[names[k]][index].tracks,trust=tracks.length?tracks.reduce((s,t)=>s+t.trust,0)/tracks.length:null;
$('m'+k).textContent=`GOSPA ${fmt(m.gospa)} · false ${m.false_tracks} · missed ${m.missed_targets} | mean local-track trust ${fmt(trust)}`;
}
function chart(){const c=$('chart'),ctx=c.getContext('2d'),w=c.width,h=c.height,left=45,top=15,bottom=h-28,end=D.references.at(-1).time_s,max=Math.max(1,...names.flatMap(n=>D.metrics[n].map(r=>r.gospa)))*1.1;const x=t=>left+t/end*(w-left-15),y=v=>bottom-v/max*(bottom-top);ctx.clearRect(0,0,w,h);ctx.fillStyle='#fbbf2420';ctx.fillRect(x(D.attack.start_s),top,x(D.attack.end_s)-x(D.attack.start_s),bottom-top);ctx.font='11px system-ui';ctx.fillStyle='#9fb0c7';for(let j=0;j<=4;j++){const v=j*max/4;ctx.fillText(v.toFixed(0),5,y(v)+4);ctx.strokeStyle='#24354d';ctx.beginPath();ctx.moveTo(left,y(v));ctx.lineTo(w,y(v));ctx.stroke()}
names.forEach((n,k)=>{ctx.strokeStyle=colors[k];ctx.lineWidth=2;ctx.beginPath();D.metrics[n].forEach((r,j)=>j?ctx.lineTo(x(r.time_s),y(r.gospa)):ctx.moveTo(x(r.time_s),y(r.gospa)));ctx.stroke()});ctx.strokeStyle='white';ctx.beginPath();ctx.moveTo(x(D.references[index].time_s),top);ctx.lineTo(x(D.references[index].time_s),bottom);ctx.stroke();ctx.fillText('0s',left,h-6);ctx.fillText(end.toFixed(1)+'s',w-40,h-6)}
function render(){const ref=D.references[index];$('scrub').value=index;$('time').textContent=fmt(ref.time_s)+' s';$('state').textContent=ref.time_s>=D.attack.start_s&&ref.time_s<=D.attack.end_s?'INJECTION WINDOW · offline experiment label':ref.time_s<D.attack.start_s?'BEFORE INJECTION · establish causal history':'AFTER INJECTION · inspect recovery and stale tracks';canvases.forEach((c,k)=>drawPanel(c,k,ref));chart()}
$('play').onclick=()=>{if(index===D.references.length-1){index=0;elapsed=0}playing=!playing;$('play').textContent=playing?'❚❚ Pause':'▶ Play';render()};$('reset').onclick=()=>{index=0;elapsed=0;playing=false;$('play').textContent='▶ Play';render()};$('scrub').oninput=e=>{index=Number(e.target.value);elapsed=0;render()};$('focus').onchange=render;$('truth').onchange=render;
function tick(now){if(playing&&last){elapsed+=(now-last)/1000*Number($('speed').value);while(index<D.references.length-1&&elapsed>=D.references[index+1].time_s-D.references[index].time_s){elapsed-=D.references[index+1].time_s-D.references[index].time_s;index++;render()}if(index===D.references.length-1){playing=false;$('play').textContent='▶ Play'}}last=now;requestAnimationFrame(tick)}render();requestAnimationFrame(tick);
</script></body></html>'''
