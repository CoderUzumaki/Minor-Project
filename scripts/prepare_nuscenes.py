"""Download official mini archive; safely extract radar, metadata and base maps.

The compressed archive contains all modalities (~4.17 GB). Selective extraction
reduces disk use, not download size. No credentials or unofficial mirrors.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import tarfile
import time
import urllib.request
from pathlib import Path, PurePosixPath

URL = 'https://www.nuscenes.org/data/v1.0-mini.tgz'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def download(url, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=60) as r:
        size = int(r.headers['Content-Length'])
        etag = r.headers.get('ETag')
    if target.exists() and target.stat().st_size == size:
        return {'url': url, 'bytes': size, 'etag': etag, 'sha256': sha256(target)}
    part = target.with_suffix(target.suffix + '.part')
    meta = part.with_suffix(part.suffix + '.json')
    identity = {'url': url, 'bytes': size, 'etag': etag}
    if part.exists() and (not meta.exists() or json.loads(meta.read_text()) != identity):
        raise RuntimeError('Partial download identity mismatch; move it aside before retrying.')
    meta.write_text(json.dumps(identity), encoding='utf-8')
    for attempt in range(5):
        start = part.stat().st_size if part.exists() else 0
        if start > size:
            raise RuntimeError('Partial file larger than source.')
        if start == size:
            break
        req = urllib.request.Request(url, headers={'Range': f'bytes={start}-'} if start else {})
        try:
            with urllib.request.urlopen(req, timeout=90) as response:
                if start and (response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {start}-')):
                    raise RuntimeError('Server did not honor resume range; refusing to append.')
                with part.open('ab' if start else 'wb') as out:
                    last = time.monotonic()
                    while chunk := response.read(4 * 1024 * 1024):
                        out.write(chunk)
                        start += len(chunk)
                        if time.monotonic() - last > 15:
                            print(f'Download {start / size:.1%} ({start / 1e9:.2f}/{size / 1e9:.2f} GB)', flush=True)
                            last = time.monotonic()
            if part.stat().st_size == size:
                break
        except (OSError, TimeoutError) as exc:
            print(f'Download attempt {attempt+1}: {exc}; resumable partial retained.', flush=True)
            if attempt == 4:
                raise
    if part.stat().st_size != size:
        raise RuntimeError('Incomplete archive.')
    part.replace(target)
    return {**identity, 'sha256': sha256(target)}


def selected_member(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
        raise ValueError(f'Unsafe archive path: {name}')
    return (p.parts and p.parts[0] in ('v1.0-mini', 'maps')) or (
        len(p.parts) >= 3 and p.parts[0] in ('samples', 'sweeps') and p.parts[1].startswith('RADAR_'))


def extract(archive, root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    files = []
    # Streaming mode avoids keeping a multi-GB archive index in memory.
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            if not selected_member(member.name) or not member.isfile():
                continue
            dest = root.joinpath(*PurePosixPath(member.name).parts)
            if not dest.resolve().is_relative_to(root):
                raise ValueError('Extraction escapes destination.')
            dest.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            with source, dest.open('wb') as out:
                shutil.copyfileobj(source, out)
            files.append({'path': dest.relative_to(root).as_posix(), 'bytes': member.size, 'sha256': sha256(dest)})
            if len(files) % 5000 == 0:
                print(f'Extracted {len(files)} selected files', flush=True)
    if not (root / 'v1.0-mini/scene.json').exists():
        raise RuntimeError('Missing mini metadata after extraction.')
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', default='downloads/v1.0-mini.tgz')
    parser.add_argument('--dataroot', default='data/nuscenes')
    parser.add_argument('--url', default=URL)
    args = parser.parse_args()
    root = Path(args.dataroot)
    marker = root / 'preparation_manifest.json'
    if marker.exists():
        manifest = json.loads(marker.read_text(encoding='utf-8'))
        if all((root / f['path']).exists() and (root / f['path']).stat().st_size == f['bytes'] for f in manifest['files']):
            print('Prepared dataset present (size check); use audit hashes for provenance.')
            return
        raise RuntimeError('Prepared dataset incomplete; move preparation_manifest.json aside to re-extract.')
    print('Downloading official nuScenes mini; dataset use remains subject to nuScenes terms.', flush=True)
    provenance = download(args.url, args.archive)
    files = extract(args.archive, root)
    marker.write_text(json.dumps({'archive': provenance, 'extraction': 'radar + metadata + base maps', 'files': files}, indent=2), encoding='utf-8')
    print(f'Prepared {len(files)} files at {root.resolve()}', flush=True)


if __name__ == '__main__':
    main()
