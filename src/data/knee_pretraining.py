"""Explicit local-image adapters; no downloads, invented pathology labels, or slice splits."""
import csv
from math import prod
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .indexing import token
from .preprocessing import normalize
from .validation import validate_volume
from src.utils.integrity import confined_path, sha256_file
from src.utils.logging import write_csv, write_json


def prepare_knee_pretraining(source: str | Path, manifest: str | Path, output: str | Path, *,
                             dataset: str, acquisition: str, image_key: str, shape=(128, 128, 32)) -> dict[str, Any]:
    """Import authorized fastMRI magnitude targets or SKM-TEA image echoes.

    Manifest: file_path, split, patient_id (blank if genuinely unavailable),
    acquisition. Acquisition is mandatory and filtered exactly. Patient grouping
    is enforced across all provided rows before volume conversion.
    """
    import h5py
    if dataset not in ('fastmri', 'skm-tea') or not acquisition:
        raise ValueError('Declare fastmri/skm-tea and one acquisition protocol')
    allowed = ('reconstruction_rss', 'reconstruction_esc') if dataset == 'fastmri' else ('echo1', 'echo2')
    if image_key not in allowed or len(shape) != 3 or any(n < 8 or n % 4 for n in shape) or prod(shape) > 67108864:
        raise ValueError('Use a documented magnitude-image key and compatible target geometry')
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Pretraining output exists; choose a fresh immutable run')
    with Path(manifest).open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not {'file_path', 'split', 'patient_id', 'acquisition'}.issubset(reader.fieldnames or []):
            raise ValueError('Manifest requires file_path, split, patient_id, acquisition')
        provided = list(reader)
    identities: dict[str, str] = {}
    paths: set[Path] = set()
    for row in provided:
        if row['split'] not in ('train', 'validation', 'test'):
            raise ValueError('Use explicit train/validation/test volume assignments')
        path = confined_path(source, row['file_path'])
        if path in paths:
            raise ValueError('Duplicate pretraining volume')
        paths.add(path)
        patient = row['patient_id'].strip()
        if patient and identities.get(patient, row['split']) != row['split']:
            raise ValueError('Pretraining patient leakage across splits')
        if patient:
            identities[patient] = row['split']
    output.mkdir(parents=True)
    rows, errors = [], []
    for record in provided:
        if record['acquisition'] != acquisition:
            continue
        path = confined_path(source, record['file_path'])
        scan = token(f'{dataset}/{record["file_path"]}/{image_key}')
        relative, mask_file = '', ''
        try:
            with h5py.File(path, 'r') as file:
                actual_acquisition = file.attrs.get('acquisition')
                if isinstance(actual_acquisition, bytes):
                    actual_acquisition = actual_acquisition.decode()
                if dataset == 'fastmri' and actual_acquisition is not None and actual_acquisition != acquisition:
                    raise ValueError('HDF5 acquisition disagrees with declared protocol')
                if image_key not in file or file[image_key].size > 67108864:
                    raise ValueError('Image target missing or exceeds volume allocation limit')
                array = np.asarray(file[image_key])
                if np.iscomplexobj(array):
                    raise ValueError('Use documented magnitude image targets, not raw complex k-space')
                validate_volume(array)
                if min(array.shape) < 3:
                    raise ValueError('A 3D representation requires a volume, not a repeated single slice')
                if dataset == 'fastmri':
                    array = array.transpose(1, 2, 0)
                original_shape = list(array.shape)
                tensor = torch.from_numpy(array.astype(np.float32).copy())[None, None]
                resized = F.interpolate(tensor, size=shape, mode='trilinear', align_corners=False)[0, 0].numpy()
                validate_volume(resized)
                normalized = normalize(resized)
                validate_volume(normalized)
                relative = f'{record["split"]}/{scan}.npy'
                (output/record['split']).mkdir(exist_ok=True)
                np.save(output/relative, normalized[None], allow_pickle=False)
                mask_file, mask_sha = '', ''
                if dataset == 'skm-tea' and 'seg' in file:
                    mask = np.asarray(file['seg'])
                    if mask.shape != (*original_shape, 6) or not np.isin(mask, [0, 1]).all():
                        raise ValueError('SKM-TEA seg must be aligned binary [X,Y,Z,6] tissue channels')
                    mask_tensor = torch.from_numpy(mask.astype(np.float32).transpose(3, 0, 1, 2).copy())[None]
                    masks = F.interpolate(mask_tensor, size=shape, mode='nearest')[0].numpy().astype(np.uint8)
                    mask_file = f'{record["split"]}/{scan}-tissues.npy'
                    np.save(output/mask_file, masks, allow_pickle=False); mask_sha = sha256_file(output/mask_file)
            rows.append(dict(scan_id=scan, exam_id=token(f'{dataset}/{record["file_path"]}'),
                patient_id=token(record['patient_id'].strip()) if record['patient_id'].strip() else '',
                split=record['split'], labels='', output_file=relative, output_sha256=sha256_file(output/relative),
                source_sha256=sha256_file(path), original_shape=original_shape, output_shape=list(shape),
                acquisition=acquisition, image_key=image_key, mask_file=mask_file, mask_sha256=mask_sha))
        except (OSError, ValueError, KeyError) as exc:
            for relative_path in (relative, mask_file):
                if relative_path:
                    (output/relative_path).unlink(missing_ok=True)
            errors.append(dict(scan_id=scan, error=type(exc).__name__, reason=str(exc)))
    fields = ['scan_id', 'exam_id', 'patient_id', 'split', 'labels', 'output_file', 'output_sha256', 'source_sha256',
              'original_shape', 'output_shape', 'acquisition', 'image_key', 'mask_file', 'mask_sha256']
    write_csv(output/'metadata.csv', rows, fields)
    write_json(output/'experiment.json', dict(dataset=dataset, acquisition=acquisition, image_key=image_key,
               shape=list(shape), source_manifest_sha256=sha256_file(manifest),
               tensor_layout='[C,row,column,slice]; input orientation/spacing require separate source metadata',
               label_policy='Representation pretraining only; no MRNet pathology labels assigned',
               tissue_channels=['patellar_cartilage', 'femoral_cartilage', 'tibial_medial', 'tibial_lateral', 'meniscus_medial', 'meniscus_lateral']))
    report = dict(valid_scans=len(rows), errors=errors, split_counts={s: sum(r['split'] == s for r in rows)
                  for s in ('train', 'validation', 'test')}, acquisition=acquisition)
    write_json(output/'report.json', report)
    return report
