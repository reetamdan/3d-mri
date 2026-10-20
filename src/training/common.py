import csv
import hashlib
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from src.data.dataset import MRIDataset
from src.utils.logging import write_json
from src.utils.integrity import confined_path, sha256_file


def device_for(name='auto'):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable in this PyTorch installation; install a compatible CUDA build or use cpu')
    return torch.device(name)


class TrainingDataset(MRIDataset):
    def __init__(self, root, split, shape, limit=None, require_labels=False):
        super().__init__(root, split)
        if limit is not None:
            self.rows = self.rows[:limit]
        self.shape = tuple(shape)
        self.require_labels = require_labels
        if require_labels and any(not r.get('labels') for r in self.rows):
            raise ValueError('This stage requires real MRNet labels; unlabeled samples cannot be assigned invented labels')
        self.ids = [r['scan_id'] for r in self.rows]
        if len(set(self.ids)) != len(self.ids):
            raise ValueError('Duplicate scan IDs')

    def __getitem__(self, i):
        x = super().__getitem__(i)
        if x.min() < -1.0001 or x.max() > 1.0001:
            raise ValueError('Model inputs require min-max normalized [-1,1] tensors')
        if tuple(x.shape[1:]) != self.shape:
            x = F.interpolate(x[None], size=self.shape, mode='trilinear', align_corners=False)[0]
        raw = self.rows[i].get('labels')
        labels = json.loads(raw) if raw else []
        if labels and (len(labels) != 3 or any(v not in (0, 1) for v in labels)):
            raise ValueError('Expected three binary MRNet labels')
        return x, torch.tensor(labels, dtype=torch.float32)


def fingerprint(root):
    metadata = Path(root)/'metadata.csv'
    digest = hashlib.sha256(metadata.read_bytes())
    with metadata.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            path = confined_path(root, row['output_file'])
            actual = sha256_file(path)
            if row.get('output_sha256') and actual != row['output_sha256']:
                raise ValueError('Processed tensor checksum mismatch; reprocess the source dataset')
            digest.update(row['output_file'].encode())
            digest.update(actual.encode())
    return digest.hexdigest()


def save_checkpoint(path, payload):
    path = Path(path)
    partial = path.with_suffix('.partial')
    torch.save(payload, partial)
    partial.replace(path)


def read_checkpoint(path, kind=None):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if kind and checkpoint['kind'] != kind:
        raise ValueError(f'Expected {kind} checkpoint')
    return checkpoint


def fresh_output(path, resume=False):
    path = Path(path)
    if path.exists() and any(path.iterdir()) and not resume:
        raise ValueError('Output exists; choose a new run directory or resume explicitly')
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_config(path):
    import yaml
    c = yaml.safe_load(Path(path).read_text())
    if len(c['shape']) != 3 or any(s < 8 or s % 8 for s in c['shape']):
        raise ValueError('Training shape must be three multiples of 8, at least 8')
    for key in ('epochs', 'batch_size', 'steps_per_epoch', 'threads'):
        if c[key] < 1:
            raise ValueError(f'{key} must be positive')
    if c['learning_rate'] <= 0 or c.get('limit', 1) < 1:
        raise ValueError('Learning rate and optional sample limit must be positive')
    return c
