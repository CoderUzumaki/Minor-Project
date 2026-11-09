"""Build a self-contained Colab notebook from the reviewed project sources."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def cell(kind,text):
    base={'cell_type':kind,'metadata':{},'source':text.splitlines(keepends=True)}
    if kind=='code':
        base.update(execution_count=None,outputs=[])
    return base


payload={str(p.relative_to(ROOT)).replace('\\','/'):p.read_text(encoding='utf-8') for p in [ROOT/'requirements.txt', ROOT/'HANDOFF.md', ROOT/'DEMO_GUIDE.md', ROOT/'README.md', ROOT/'Research_Plan.md', *sorted((ROOT/'scripts').glob('*.py')),*sorted((ROOT/'tests').glob('*.py'))] if p.name!='build_notebook.py'}
cells=[cell('markdown','''# Coordinated Radar Research — 01: nuScenes mini coverage audit

CPU runtime is sufficient. No model is trained here. This notebook downloads the official
4.17 GB mini archive and extracts radar streams, metadata and base maps only.
Keep dataset use within the [nuScenes terms](https://www.nuscenes.org/terms-of-use).

Run all cells in order. Installation uses an isolated environment, leaving Colab's
preinstalled NumPy/PyTorch untouched. Source scripts are embedded so no GitHub repository
or additional upload is required. Outputs are measured, never prefilled.

**Question:** how many annotated objects have observations from 2 or 3 distinct radars
in a causal window, and for how many consecutive labeled keyframes?
'''),cell('code','''from pathlib import Path
import sys, subprocess, json, shutil, platform
PROJECT = Path('/content/radar-project') if Path('/content').exists() else Path.cwd() / 'colab_run'
PROJECT.mkdir(parents=True, exist_ok=True)
print('Host:', platform.platform(), sys.version)
print('Free space (GB):', round(shutil.disk_usage(PROJECT).free / 1e9, 1))
if shutil.disk_usage(PROJECT).free < 8e9:
    raise RuntimeError('At least 8 GB free space is required for this audit.')
'''),cell('code','FILES = '+repr(payload)+'''
for relative, source in FILES.items():
    destination = PROJECT / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source, encoding='utf-8')
print('Project sources installed:', sorted(FILES))
'''),cell('code','''# Isolate Python 3.12 from Colab's kernel; uv does not require ensurepip.
def run_logged(command, cwd=PROJECT):
    process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in process.stdout:
        print(line, end='', flush=True)
    if process.wait() != 0:
        raise RuntimeError(f'Command failed: {command}')
run_logged([sys.executable, '-m', 'pip', 'install', 'uv'])
PYTHON = PROJECT / '.venv' / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
if not PYTHON.exists():
    run_logged([sys.executable, '-m', 'uv', 'venv', '--python', '3.12', str(PROJECT / '.venv')])
run_logged([sys.executable, '-m', 'uv', 'pip', 'install', '--python', str(PYTHON), '-r', str(PROJECT / 'requirements.txt')])
run_logged([sys.executable, '-m', 'uv', 'pip', 'check', '--python', str(PYTHON)])
run_logged([str(PYTHON), '-m', 'pytest', '-q', '--basetemp=outputs/test_tmp', 'tests'])
'''),cell('markdown','''## Download and prepare
The first run downloads ~4.17 GB. The script resumes interrupted transfers and records
the archive SHA-256. Extraction retains no camera or LiDAR payload. Repeat runs reuse
the prepared subset after a size check. Files live in the temporary runtime; download
the results before disconnecting. Google Drive is not mounted or granted broad access.
'''),cell('code','''run_logged([str(PYTHON), '-u', 'scripts/prepare_nuscenes.py'])
'''),cell('markdown','''## Run the offline coverage audit
Uses the latest sweep **at or before** each annotation timestamp. Tests 50/100/200 ms,
0/0.5 m BEV padding, and 1/2 valid points per sensor. No multisweep accumulation.
Overlapping-box points are excluded. The fixed-pair/triple episode metric requires
at least 3 consecutive keyframes and a 0.9 s endpoint span.

**Important:** annotation-derived velocity can use future annotations, solely to align
boxes for this offline audit. It must never be exposed to the later detector. Radar
points in a box are a geometric coverage proxy, not proven point-to-object identities.
'''),cell('code','''run_logged([str(PYTHON), '-u', 'scripts/audit_radar_overlap.py'])
'''),cell('code','''from IPython.display import display, Markdown, Image
RESULTS = PROJECT / 'outputs/mini_audit'
display(Markdown((RESULTS / 'REPORT.md').read_text()))
display(Image(filename=str(RESULTS / 'coverage.png')))
display(Image(filename=str(RESULTS / 'example_overlap.png')))
'''),cell('markdown','''## Save the evidence
The results include object-level counts, per-scene/category summaries, exact candidate
episode IDs, sensitivity tables, code/data hashes and package versions. Mini establishes
feasibility only: do not train and claim generalization on these same 10 scenes.
'''),cell('code','''archive = shutil.make_archive(str(PROJECT / 'mini_audit_results'), 'zip', RESULTS)
print('Results archive:', archive)
# Download explicitly when ready; this does not mount Drive or share your data.
# from google.colab import files
# files.download(archive)
''')]
notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'colab':{'name':'01_nuscenes_mini_radar_audit.ipynb','provenance':[]},'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'cells':cells}
out=ROOT/'notebooks/01_nuscenes_mini_radar_audit.ipynb'
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(notebook,indent=2),encoding='utf-8')
print(out)

# Reuse the same installation, embedded sources, preparation and audit steps.
# Both notebooks remain self-contained and reproducible from a fresh CPU runtime.
replay_cells = [cell('markdown', '''# Coordinated Radar Research — 02: Causal replay and display

Run all cells using a CPU runtime. The first run downloads the official ~4.17 GB
mini archive. Existing data under /content/radar-project are reused.

This is a **development baseline**, not a trained GNN or validated attack defense.
The generated offline HTML compares clean, attacked and protected runs, including
false alerts and missed detections. No fabricated results or target percentages.
'''), *cells[1:8], cell('markdown', '''## Build the replay experiment

Freeze recording-grouped splits, select an audited **training** episode, export
annotation-free reports, and run four paired branches. Injection uses offline
annotations; inference has no access to them. Heuristic parameters are fixed in source.
'''), cell('code', '''run_logged([str(PYTHON), '-u', 'scripts/run_replay.py'])
run_logged([str(PYTHON), '-u', 'scripts/verify_replay.py'])
'''), cell('code', '''from IPython.display import display, Markdown, Image
RESULTS = PROJECT / 'outputs/replay_pilot'
display(Markdown((RESULTS / 'REPORT.md').read_text()))
display(Image(filename=str(RESULTS / 'tracking_comparison.png')))
'''), cell('markdown', '''## Present and preserve the experiment

Download the ZIP, extract it, and open **demo.html** in a desktop browser. It has
embedded replay data, playback, scrubbing, reference overlays, and measured results.
No server or internet connection is required for the HTML itself.

For your presentation: explain the clean replay, pause in the attack window,
compare the protected branch, and discuss false alerts and observability limits.
The current baseline can fail; this is explicitly shown, not hidden.
'''), cell('code', '''archive = shutil.make_archive(str(PROJECT / 'replay_pilot_results'), 'zip', RESULTS)
print('Download this archive before the runtime ends:', archive)
print('Desktop demo:', RESULTS / 'demo.html')
# from google.colab import files
# files.download(archive)
''')]
replay_notebook = dict(notebook, cells=replay_cells,
    metadata={**notebook['metadata'], 'colab':{'name':'02_causal_radar_replay.ipynb','provenance':[]}})
replay_out = ROOT / 'notebooks/02_causal_radar_replay.ipynb'
replay_out.write_text(json.dumps(replay_notebook, indent=2), encoding='utf-8')
print(replay_out)
