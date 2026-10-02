#!/usr/bin/env python3
"""Download, verify and restore the complete public GitHub Release (stdlib only)."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import tarfile
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def request(url):
    return urlopen(Request(url, headers={'User-Agent': 'qwen-lora-enhance-release-restore'}), timeout=60)


class JoinedParts(io.RawIOBase):
    def __init__(self, paths):
        self.paths = iter(paths)
        self.handle = None

    def readable(self):
        return True

    def read(self, amount=-1):
        if amount < 0:
            raise ValueError('Only bounded streaming reads are supported')
        blocks = []
        remaining = amount
        while remaining:
            if self.handle is None:
                path = next(self.paths, None)
                if path is None:
                    break
                self.handle = path.open('rb')
            block = self.handle.read(remaining)
            if not block:
                self.handle.close()
                self.handle = None
                continue
            blocks.append(block)
            remaining -= len(block)
        return b''.join(blocks)

    def close(self):
        if self.handle:
            self.handle.close()
        super().close()


def download(url, path, size, expected):
    if path.is_file() and path.stat().st_size == size and sha256(path) == expected:
        print(f'Already verified: {path.name}', flush=True)
        return
    temporary = path.with_name(path.name + '.download')
    with request(url) as response, temporary.open('wb') as target:
        shutil.copyfileobj(response, target, length=8*1024*1024)
    if temporary.stat().st_size != size or sha256(temporary) != expected:
        raise RuntimeError(f'Download checksum or size mismatch: {path.name}')
    temporary.replace(path)
    print(f'Downloaded and verified: {path.name}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True, help='Parent folder for a NEW project directory')
    parser.add_argument('--download-dir', type=Path, help='Folder for downloaded parts; keep it to resume completed parts')
    parser.add_argument('--repository', default='SHIBEI1/qwen_lora_enhance_final')
    parser.add_argument('--tag', default='v1.0.0')
    args = parser.parse_args()
    destination = args.destination.expanduser().resolve()
    cache = (args.download_dir or destination / '_github_release_parts').expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    with request(f'https://api.github.com/repos/{args.repository}/releases/tags/{args.tag}') as response:
        release = json.load(response)
    assets = {entry['name']: entry for entry in release['assets']}
    manifest_asset = assets['FULL_PROJECT_MANIFEST.json']
    with request(manifest_asset['browser_download_url']) as response:
        raw = response.read()
    actual = hashlib.sha256(raw).hexdigest()
    if manifest_asset.get('digest') != f'sha256:{actual}':
        raise RuntimeError('Manifest does not match the GitHub server-side SHA-256 digest')
    manifest = json.loads(raw)
    if manifest['repository'] != args.repository or manifest['tag'] != args.tag:
        raise RuntimeError('Unexpected release identity in the manifest')
    project_name = manifest['project_directory']
    if project_name != 'qwen_lora_enhance_final':
        raise RuntimeError('Unexpected top-level project directory')
    project = destination / project_name
    if project.exists():
        raise RuntimeError(f'Refusing to overwrite an existing project: {project}')
    if not manifest['parts'] or len(manifest['parts']) > 1000:
        raise RuntimeError('Invalid archive part count')
    remaining_download = sum(item['size'] for item in manifest['parts']
                             if not (cache / item['name']).is_file()
                             or (cache / item['name']).stat().st_size != item['size'])
    if shutil.disk_usage(destination).free < manifest['source_bytes'] + remaining_download + 1024**3:
        raise RuntimeError('Allow approximately 112 GiB free space for downloaded parts plus restored files')
    manifest_path = cache / 'FULL_PROJECT_MANIFEST.json'
    manifest_path.write_bytes(raw)
    paths = []
    archive_hash = hashlib.sha256()
    for entry in manifest['parts']:
        name = entry['name']
        if Path(name).name != name or not name.startswith('qwen_lora_enhance_final.tar.part'):
            raise RuntimeError('Unsafe archive part name')
        asset = assets[name]
        if asset['size'] != entry['size'] or asset.get('digest') != 'sha256:' + entry['sha256']:
            raise RuntimeError(f'GitHub asset checksum mismatch: {name}')
        path = cache / name
        download(asset['browser_download_url'], path, entry['size'], entry['sha256'])
        paths.append(path)
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(8*1024*1024), b''):
                archive_hash.update(block)
    if archive_hash.hexdigest() != manifest['archive_sha256']:
        raise RuntimeError('Combined tar checksum mismatch')
    expected = {item['path']: item for item in manifest['files']}
    seen = set()
    with JoinedParts(paths) as stream, tarfile.open(fileobj=stream, mode='r|') as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts or not name.parts or name.parts[0] != project_name:
                raise RuntimeError(f'Unsafe archive member: {member.name}')
            target = destination.joinpath(*name.parts)
            if not target.resolve().is_relative_to(project):
                raise RuntimeError('Archive path escapes the project directory')
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            relative = PurePosixPath(*name.parts[1:]).as_posix()
            if not member.isfile() or relative not in expected or relative in seen:
                raise RuntimeError(f'Unexpected, duplicate or linked archive member: {member.name}')
            target.parent.mkdir(parents=True, exist_ok=True)
            value = hashlib.sha256()
            with archive.extractfile(member) as source, target.open('xb') as output:
                for block in iter(lambda: source.read(8*1024*1024), b''):
                    output.write(block)
                    value.update(block)
            if target.stat().st_size != expected[relative]['size'] or value.hexdigest() != expected[relative]['sha256']:
                raise RuntimeError(f'Restored file checksum mismatch: {relative}')
            os.chmod(target, member.mode & 0o777)
            os.utime(target, (member.mtime, member.mtime))
            seen.add(relative)
            if len(seen) % 200 == 0:
                print(f'Restored and verified {len(seen)}/{len(expected)} files', flush=True)
    if seen != set(expected):
        raise RuntimeError('Restoration incomplete: missing release files')
    print(f'Complete: {len(seen)} verified real files restored to {project}', flush=True)
    print('Download parts were retained for recovery; remove them manually after checking the project.', flush=True)


if __name__ == '__main__':
    main()
