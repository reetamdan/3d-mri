"""MRNet adapter preserves official splits and treats planes as independent series.

MRNet .npy arrays carry no NIfTI affine or reliable physical voxel spacing.
No physical RAS orientation or millimeter resampling is invented.
"""
import csv
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .validation import validate_volume
from .preprocessing import normalize
from .indexing import token
from src.utils.logging import write_json, write_csv
from src.utils.integrity import sha256_file

TASKS = ('abnormal', 'acl', 'meniscus')
PLANES = ('axial', 'sagittal', 'coronal')


def labels_for(root, split):
    result = {}
    for task in TASKS:
        with (Path(root)/f'{split}-{task}.csv').open(newline='', encoding='utf-8-sig') as f:
            values = {}
            for row in csv.reader(f):
                if len(row) != 2 or row[1] not in ('0', '1') or not row[0].isdigit():
                    raise ValueError(f'Invalid {split}-{task}.csv; expected headerless exam_id,0_or_1')
                exam = row[0].zfill(4)
                if exam in values:
                    raise ValueError('Duplicate exam label')
                values[exam] = int(row[1])
        result[task] = values
    if not all(set(result[t]) == set(result[TASKS[0]]) for t in TASKS):
        raise ValueError('MRNet label files do not contain identical exam IDs')
    return {exam: [result[t][exam] for t in TASKS] for exam in result[TASKS[0]]}


def load_series(path, max_voxels=67108864):
    array = np.load(path, allow_pickle=False, mmap_mode='r')
    if array.size > max_voxels:
        raise ValueError('MRNet series exceeds max_voxels')
    validate_volume(array)
    return np.asarray(array, dtype=np.float32)


def prepare_mrnet(root, output, plane='sagittal', shape=(128, 128, 32), limit=None):
    root, output = Path(root), Path(output)
    if plane not in PLANES or len(shape) != 3 or any(s < 8 or s % 4 for s in shape):
        raise ValueError('Choose an MRNet plane and dimensions >=8 divisible by 4')
    if np.prod(shape, dtype=np.int64) > 67108864:
        raise ValueError('Target shape exceeds the allocation limit')
    if limit is not None and limit < 1:
        raise ValueError('limit must be positive')
    if output.exists():
        raise ValueError('Choose a new output directory; processed runs are immutable')
    labels = {s: labels_for(root, s) for s in ('train', 'valid')}
    if set(labels['train']) & set(labels['valid']):
        raise ValueError('Exam identifiers overlap between official train and validation splits')
    rows, errors = [], []
    output.mkdir(parents=True)
    for source_split, split in [('train', 'train'), ('valid', 'validation')]:
        (output/split).mkdir()
        for exam in sorted(labels[source_split])[:limit]:
            scan_id = token(f'MRNet/{exam}/{plane}')
            path = root/source_split/plane/f'{exam}.npy'
            try:
                series = load_series(path)
                # [slices, rows, columns] -> [1, rows, columns, slices]
                array = series.transpose(1, 2, 0)
                tensor = torch.from_numpy(array.copy())[None, None]
                tensor = F.interpolate(tensor, size=shape, mode='trilinear', align_corners=False)[0, 0].numpy()
                tensor = normalize(tensor)[None]
                validate_volume(tensor[0])
                relative = f'{split}/{scan_id}.npy'
                np.save(output/relative, tensor, allow_pickle=False)
                rows.append(dict(scan_id=scan_id, exam_id=token(exam), patient_id='', split=split,
                    output_sha256=sha256_file(output/relative),
                    output_file=relative, labels=labels[source_split][exam], plane=plane,
                    source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    original_shape=list(series.shape), output_shape=list(shape)))
            except Exception as exc:
                errors.append(dict(scan_id=scan_id, code=type(exc).__name__, reason='Unable to load or preprocess MRNet series'))
    write_csv(output/'metadata.csv', rows, ['scan_id', 'exam_id', 'patient_id', 'split', 'output_file', 'output_sha256', 'labels', 'plane', 'source_sha256', 'original_shape', 'output_shape'])
    write_csv(output/'validation_errors.csv', errors, ['scan_id', 'code', 'reason'])
    write_json(output/'experiment.json', dict(dataset='MRNet', plane=plane, shape=list(shape),
        tensor_layout='[C, row, column, slice]; physical orientation/spacing unavailable',
        split_policy='Official MRNet train/valid preserved; no exam-level random resplitting',
        patient_identifiers_available=False, labels=list(TASKS), root=str(root.resolve())))
    report = dict(valid_scans=len(rows), invalid_scans=len(errors),
        split_scans={s: sum(r['split'] == s for r in rows) for s in ('train', 'validation', 'test')})
    write_json(output/'report.json', report)
    return report
