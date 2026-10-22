from pathlib import Path
import csv
import numpy as np
import torch
from torch.utils.data import Dataset
from src.utils.integrity import confined_path, sha256_file


class MRIDataset(Dataset):
    """Read a completed run as float32 [C, R, A, S] tensors for future models."""
    def __init__(self, run_dir: str | Path, split: str = 'train'):
        if split not in ('train', 'validation', 'test'):
            raise ValueError('Unknown split')
        self.root = Path(run_dir).resolve()
        with (self.root / 'metadata.csv').open(newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        seen: set[str] = set()
        subjects: dict[tuple[str, str], str] = {}
        for row in rows:
            if row['split'] not in ('train', 'validation', 'test') or row['scan_id'] in seen:
                raise ValueError('Duplicate scan ID or invalid split in metadata')
            seen.add(row['scan_id'])
            for field in ('patient_id', 'exam_id'):
                identity = row.get(field)
                if identity:
                    key = (field, identity)
                    if key in subjects and subjects[key] != row['split']:
                        raise ValueError('Patient/exam leakage across splits')
                    subjects[key] = row['split']
        self.rows = [r for r in rows if r['split'] == split]
        self._verified: dict[Path, tuple[int, int]] = {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = confined_path(self.root, row['output_file'])
        expected = row.get('output_sha256')
        stamp = (path.stat().st_mtime_ns, path.stat().st_size)
        if expected and self._verified.get(path) != stamp:
            if sha256_file(path) != expected:
                raise ValueError('Stored tensor checksum mismatch')
            self._verified[path] = stamp
        tensor = torch.load(path, map_location='cpu', weights_only=True) if path.suffix == '.pt' else torch.from_numpy(np.load(path, allow_pickle=False))
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.numel() == 0 or not torch.isfinite(tensor).all():
            raise ValueError('Invalid stored tensor')
        return tensor.float()
