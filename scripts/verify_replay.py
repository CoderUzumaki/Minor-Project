"""Re-run protected inference in a directory containing no labels or annotations."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='outputs/replay_pilot')
    args = parser.parse_args()
    out = Path(args.output).resolve()
    root = Path(__file__).resolve().parents[1]
    checks = out / 'checks'
    checks.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='annotation_free_', dir=checks) as temporary:
        work = Path(temporary)
        for name in ('replay_core.py', 'infer_replay.py'):
            shutil.copy2(root / 'scripts' / name, work / name)
        shutil.copy2(out / 'inputs/attacked.json', work / 'input.json')
        subprocess.run([sys.executable, str(work / 'infer_replay.py'), 'input.json', 'result.json', '--protected'],
                       cwd=work, check=True)
        actual = json.loads((work / 'result.json').read_text())
        expected = json.loads((out / 'runs/attacked_protected.json').read_text())
        # Wall-clock profiling is explicitly nondeterministic; scientific outputs are not.
        actual.pop('timing')
        expected.pop('timing')
        if actual != expected:
            raise AssertionError('Isolated inference differs from saved experiment')
        result = dict(passed=True, check='isolated annotation-free inference reproduces frames and decisions',
                      compared_fields=['frames', 'decisions'], excluded_fields=['timing'],
                      input_sha256=hashlib.sha256((work / 'input.json').read_bytes()).hexdigest(),
                      frame_count=len(actual['frames']), decision_count=len(actual['decisions']))
    (checks / 'inference_isolation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
