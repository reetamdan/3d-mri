"""Streaming processing with isolated run directories and per-scan provenance."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.metadata
import uuid
import numpy as np
from .indexing import index_dataset
from .loader import load_mri
from .preprocessing import preprocess_volume
from .splitting import split_patients
from .reporting import summarize
from .validation import VolumeValidationError
from src.utils.config import validate_config
from src.utils.logging import write_json, write_csv
from src.utils.seed import set_seed
from src.utils.integrity import sha256_file


def safe_error(record: dict, exc: Exception) -> dict:
    # Unexpected exception text may contain local paths or identifying filenames.
    return {'scan_id': record['scan_id'], 'code': getattr(exc, 'code', 'processing'),
            'reason': str(exc) if isinstance(exc, (VolumeValidationError, ValueError)) else type(exc).__name__}


def inspect_dataset(config: dict) -> tuple[list[dict], list[dict], dict]:
    indexed = index_dataset(config['dataset'])
    valid, errors = [], []
    for record in indexed:
        try:
            _, meta = load_mri(record['path'], config['preprocessing'].get('max_voxels', 67108864))
            meta.pop('file_path')
            valid.append({**record, **meta})
        except Exception as exc:
            errors.append(safe_error(record, exc))
    return valid, errors, summarize(valid, errors, len(indexed))


def source_hash(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def run_pipeline(config: dict, progress=None) -> dict:
    config = validate_config(config)
    set_seed(config['seed'])
    indexed = index_dataset(config['dataset'])
    if not indexed:
        raise ValueError('No NIfTI scans found; add a dataset before processing')
    assignment = split_patients([r['patient_id'] for r in indexed], config['split'], config['seed'])
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8]
    output = Path(config['dataset']['processed_dir']) / run_id
    for split in ('train', 'validation', 'test'):
        (output / split).mkdir(parents=True, exist_ok=False)
    rows, raw_records, errors = [], [], []
    for i, record in enumerate(indexed):
        try:
            volume, meta = load_mri(record['path'], config['preprocessing'].get('max_voxels', 67108864))
            result, processed = preprocess_volume(volume, meta, config['preprocessing'])
            split = assignment[record['patient_id']]
            ext = config.get('export_format', 'npy')
            relative = f'{split}/{record["scan_id"]}.{ext}'
            tensor = np.ascontiguousarray(result[None], dtype=np.float32)
            checksum = source_hash(record['path'])
            if ext == 'pt':
                import torch
                torch.save(torch.from_numpy(tensor), output / relative)
            else:
                np.save(output / relative, tensor, allow_pickle=False)
            meta.pop('file_path')
            raw_records.append({**meta, 'patient_id': record['patient_id']})
            row = {'scan_id': record['scan_id'], 'patient_id': record['patient_id'],
                   'output_sha256': sha256_file(output / relative),
                   'split': split, 'output_file': relative, 'source_sha256': checksum,
                   'original': meta, 'processed': processed}
            rows.append(row)
        except Exception as exc:
            errors.append(safe_error(record, exc))
        if progress:
            progress((i + 1) / len(indexed))
    report = summarize(raw_records, errors, len(indexed))
    report['split_scans'] = {s: sum(r['split'] == s for r in rows) for s in ('train', 'validation', 'test')}
    report['split_subjects'] = {s: len({r['patient_id'] for r in rows if r['split'] == s}) for s in ('train', 'validation', 'test')}
    report['run_id'] = run_id
    versions = {p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'nibabel', 'torch', 'streamlit')}
    write_json(output / 'experiment.json', {'config': config, 'versions': versions, 'run_id': run_id,
                'dataset_fingerprint': hashlib.sha256(''.join(sorted(r['source_sha256'] for r in rows)).encode()).hexdigest(),
                'tensor_layout': '[C, R, A, S]; batch as [B, C, R, A, S]', 'subject_assignments': assignment})
    write_csv(output / 'metadata.csv', rows, ['scan_id', 'patient_id', 'split', 'output_file', 'output_sha256', 'source_sha256', 'original', 'processed'])
    write_json(output / 'report.json', report)
    write_csv(output / 'validation_errors.csv', errors, ['scan_id', 'code', 'reason'])
    write_csv(output / 'processing_log.csv',
              [{'scan_id': r['scan_id'], 'status': 'exported', 'detail': r['output_file']} for r in rows]
              + [{'scan_id': e['scan_id'], 'status': 'rejected', 'detail': e['code']} for e in errors],
              ['scan_id', 'status', 'detail'])
    write_csv(Path(config['logs_dir']) / 'validation_errors.csv', errors, ['scan_id', 'code', 'reason'])
    write_json(Path(config['dataset']['metadata_dir']) / 'latest_run.json', {'output_dir': str(output), 'report': report})
    return {'output_dir': str(output), 'report': report, 'rows': rows, 'errors': errors}
