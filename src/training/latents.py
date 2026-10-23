"""Integrity-checked contiguous/memory-mapped latent cache with legacy fallback."""
from pathlib import Path
import numpy as np
import torch
from src.utils.integrity import confined_path, sha256_file


def transform_latents(z: torch.Tensor, manifest: dict, inverse=False) -> torch.Tensor:
    """Normalize 5D tensors; legacy scalar-cache behavior stays algebraically identical."""
    if 'channel_mean' not in manifest:
        mean, std = manifest['mean'], manifest['std']
    else:
        if z.ndim != 5 or len(manifest['channel_mean']) != z.shape[1] or len(manifest['channel_std']) != z.shape[1]:
            raise ValueError('Latent scaling channels do not match the model')
        mean = z.new_tensor(manifest['channel_mean']).view(1, -1, 1, 1, 1)
        std = z.new_tensor(manifest['channel_std']).view(1, -1, 1, 1, 1)
    if not torch.isfinite(torch.as_tensor(mean)).all() or not torch.isfinite(torch.as_tensor(std)).all() or torch.any(torch.as_tensor(std) <= 0):
        raise ValueError('Invalid latent scaling constants')
    return z*std+mean if inverse else (z-mean)/std


def pack_latents(root, rows):
    root = Path(root)
    packed = {}
    for split in ('train', 'validation', 'test'):
        selected = [r for r in rows if r['split'] == split]
        if not selected:
            continue
        first = torch.load(root/selected[0]['file'], weights_only=True).numpy()
        path = root/f'{split}-packed.npy'
        array = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(len(selected), *first.shape))
        for i, row in enumerate(selected):
            array[i] = torch.load(root/row['file'], weights_only=True).numpy()
            row['packed_index'] = i
        array.flush(); del array
        packed[split] = dict(file=path.name, sha256=sha256_file(path), count=len(selected))
    return packed


class LatentCache:
    def __init__(self, root, manifest):
        self.root = Path(root)
        self.arrays = {}
        for split, info in manifest.get('packed', {}).items():
            path = confined_path(root, info['file'])
            if sha256_file(path) != info['sha256']:
                raise ValueError('Packed latent cache checksum mismatch')
            array = np.load(path, mmap_mode='r', allow_pickle=False)
            if array.shape != (info['count'], *manifest['latent_shape']) or array.dtype != np.float32:
                raise ValueError('Invalid packed latent cache geometry')
            self.arrays[split] = array

    def tensor(self, row):
        if row['split'] in self.arrays:
            return torch.from_numpy(self.arrays[row['split']][row['packed_index']].copy())
        return torch.load(confined_path(self.root, row['file']), weights_only=True)

    def batch(self, rows):
        return torch.stack([self.tensor(row) for row in rows])

    def close(self):
        for array in self.arrays.values():
            array._mmap.close()
        self.arrays.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
