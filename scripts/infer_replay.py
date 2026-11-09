"""Run inference using a canonical input file, without annotations or labels."""
import argparse
import json
from pathlib import Path
from replay_core import replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('--protected', action='store_true')
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text())
    result = replay(data['events'], data['frame_times'], args.protected)
    Path(args.output).write_text(json.dumps(result, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
