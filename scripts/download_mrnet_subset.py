"""Explicit, size-bounded MRNet subset download from a public Kaggle mirror.

Preserves official splits and records the mirror's unspecified license metadata.
No credentials are required for this public endpoint or stored by this script.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
from threading import Lock
import time
from urllib.parse import quote
from urllib.request import Request, urlopen
import zipfile
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.mrnet import TASKS, PLANES
from src.data.validation import validate_volume
from src.utils.logging import write_json

MIRROR = 'cjinny/mrnet-v1'
VERSION = 1
MAX_FILE_BYTES = 32 * 1024**2


def fetch(relative):
    url = (f'https://www.kaggle.com/api/v1/datasets/download/{MIRROR}/'
           f'{quote("MRNet-v1.0/" + relative, safe="")}?datasetVersionNumber={VERSION}')
    for attempt in range(5):
        try:
            with urlopen(Request(url, headers={'User-Agent': '3d-mri-research/1.0'}), timeout=90) as response:
                data = response.read(MAX_FILE_BYTES + 1)
            if len(data) > MAX_FILE_BYTES:
                raise ValueError('A source file exceeds the 32 MiB per-file download limit')
            if zipfile.is_zipfile(io.BytesIO(data)):
                with zipfile.ZipFile(io.BytesIO(data)) as archive:
                    files = [info for info in archive.infolist() if not info.is_dir()]
                    if (len(files) != 1 or files[0].file_size > MAX_FILE_BYTES
                            or Path(files[0].filename).name != Path(relative).name):
                        raise ValueError('Unexpected single-file Kaggle download archive')
                    data = archive.read(files[0])
            return data
        except (OSError, TimeoutError):
            if attempt == 4:
                raise
            time.sleep(2**attempt)


def stratified_ids(labels, count, seed):
    """Proportional joint-label strata; selection never crosses official splits."""
    if not 1 <= count <= len(labels):
        raise ValueError(f'Subset count must be between 1 and {len(labels)}')
    groups: dict[tuple[int, ...], list[str]] = {}
    for exam, values in sorted(labels.items()):
        groups.setdefault(tuple(values), []).append(exam)
    keys = sorted(groups)
    exact = np.array([len(groups[k]) for k in keys]) * count / len(labels)
    sizes = np.floor(exact).astype(int)
    for i in np.argsort(-(exact - sizes), kind='stable')[:count - sizes.sum()]:
        sizes[i] += 1
    rng = np.random.default_rng(seed)
    chosen = []
    for key, size in zip(keys, sizes):
        chosen.extend(rng.permutation(groups[key])[:size].tolist())
    return sorted(chosen)


def download_subset(output, train_count=96, validation_count=24, seed=42,
                    planes=PLANES, max_mib=1024, workers=6):
    output = Path(output)
    spec = dict(mirror=MIRROR, version=VERSION, train_count=train_count,
                validation_count=validation_count, seed=seed, planes=list(planes))
    manifest_path = output / 'source_manifest.json'
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if previous and previous['selection'] != spec:
        raise ValueError('Existing subset has different settings; choose a new output folder')
    previous_files = {row['file']: row for row in previous.get('files', [])}
    output.mkdir(parents=True, exist_ok=True)
    labels, label_sources = {}, []
    for split in ('train', 'valid'):
        by_task = {}
        for task in TASKS:
            name = f'{split}-{task}.csv'
            data = fetch(name)
            values = {}
            for row in csv.reader(io.StringIO(data.decode('utf-8-sig'))):
                if len(row) != 2 or not row[0].isdigit() or row[1] not in ('0', '1'):
                    raise ValueError('Unexpected upstream MRNet label format')
                exam = row[0].zfill(4)
                if exam in values:
                    raise ValueError('Duplicate upstream exam label')
                values[exam] = int(row[1])
            by_task[task] = values
            label_sources.append(dict(file=name, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)))
        if any(set(by_task[t]) != set(by_task[TASKS[0]]) for t in TASKS):
            raise ValueError('Upstream label files disagree on exam IDs')
        labels[split] = {exam: [by_task[t][exam] for t in TASKS] for exam in by_task[TASKS[0]]}
    if set(labels['train']) & set(labels['valid']):
        raise ValueError('Official split exam IDs overlap')
    selected = dict(train=stratified_ids(labels['train'], train_count, seed),
                    valid=stratified_ids(labels['valid'], validation_count, seed + 1))
    for split in selected:
        for i, task in enumerate(TASKS):
            (output / f'{split}-{task}.csv').write_text(
                ''.join(f'{exam},{labels[split][exam][i]}\n' for exam in selected[split]), encoding='utf-8')
    manifest: dict[str, Any] = dict(selection=spec, complete=False,
        source_url=f'https://www.kaggle.com/datasets/{MIRROR}',
        upstream_url='https://stanfordmlgroup.github.io/competitions/mrnet/',
        mirror_license='Unknown (as reported by Kaggle); no new license is asserted',
        purpose='Non-commercial academic M3/M4 training subset; not the full MRNet benchmark',
        split_policy='Official train/valid assignment preserved; trust Stanford patient-disjoint split',
        patient_identifiers_available=False, selection_method='Seeded proportional joint-label strata',
        selected_exams=selected, original_label_files=label_sources,
        class_counts={s: {t: sum(labels[s][e][i] for e in selected[s]) for i, t in enumerate(TASKS)}
                      for s in selected}, files=[])
    jobs = [(s, e, p) for s in selected for e in selected[s] for p in planes]
    total = 0
    reserved = 0
    budget_lock = Lock()
    def reserve(size):
        nonlocal reserved
        with budget_lock:
            if reserved + size > max_mib * 1024**2:
                raise ValueError('Subset exceeds the configured total source-array byte budget')
            reserved += size
    def retrieve(job):
        split, exam, plane = job
        relative = f'{split}/{plane}/{exam}.npy'
        path = output / relative
        cached = previous_files.get(relative)
        if cached and path.is_file():
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() == cached['sha256']:
                reserve(len(data))
                return cached
        data = fetch(relative)
        volume = np.load(io.BytesIO(data), allow_pickle=False)
        validate_volume(volume)
        if volume.dtype != np.uint8 or volume.shape[1:] != (256, 256):
            raise ValueError('Unexpected MRNet source dtype or in-plane dimensions')
        reserve(len(data))
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix('.partial')
        partial.write_bytes(data)
        partial.replace(path)
        return dict(file=relative, sha256=hashlib.sha256(data).hexdigest(),
                    bytes=len(data), shape=list(volume.shape), dtype=str(volume.dtype))
    write_json(manifest_path, manifest)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(retrieve, job) for job in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            row = future.result()
            total += row['bytes']
            manifest['files'].append(row)
            if i % 12 == 0 or i == len(jobs):
                write_json(manifest_path, manifest)
                print(f'Downloaded/verified {i}/{len(jobs)} series; {total/1024**2:.1f} MiB', flush=True)
    manifest['files'].sort(key=lambda row: row['file'])
    manifest.update(complete=True, source_array_bytes=total,
                    completed_at=datetime.now(timezone.utc).isoformat())
    write_json(manifest_path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='data/mrnet/MRNet-v1.0')
    parser.add_argument('--train', type=int, default=96)
    parser.add_argument('--validation', type=int, default=24)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--planes', nargs='+', choices=PLANES, default=list(PLANES))
    parser.add_argument('--max-mib', type=int, default=1024)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if args.max_mib < 1 or not 1 <= args.workers <= 12:
        parser.error('Use a positive byte budget and 1–12 workers')
    result = download_subset(args.output, args.train, args.validation, args.seed,
                             args.planes, args.max_mib, args.workers)
    print(json.dumps({k: result[k] for k in ('selection', 'class_counts', 'source_array_bytes', 'complete')}, indent=2))


if __name__ == '__main__':
    main()
