import json
from collections.abc import Iterator
from pathlib import Path
import torch
from torch.nn import functional as F
from src.data.dataset import MRIDataset
from src.utils.cohort import verify_cohort


def device_for(name='auto'):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable in this PyTorch installation; install a compatible CUDA build or use cpu')
    return torch.device(name)


class TrainingDataset(MRIDataset):
    def __init__(self, root, split, shape, limit=None, require_labels=False, strict_shape=False):
        super().__init__(root, split)
        if limit is not None:
            self.rows = self.rows[:limit]
        self.shape = tuple(shape)
        self.require_labels = require_labels
        self.strict_shape = strict_shape
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
            if self.strict_shape:
                raise ValueError('Frozen preprocessing shape differs from model; no silent resize is allowed')
            x = F.interpolate(x[None], size=self.shape, mode='trilinear', align_corners=False)[0]
        raw = self.rows[i].get('labels')
        labels = json.loads(raw) if raw else []
        if labels and (len(labels) != 3 or any(v not in (0, 1) for v in labels)):
            raise ValueError('Expected three binary MRNet labels')
        return x, torch.tensor(labels, dtype=torch.float32)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for index in range(len(self)):
            yield self[index]


def fingerprint(root):
    return verify_cohort(root).fingerprint


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


def validate_output(path, resume=False):
    path = Path(path)
    if path.exists() and (not path.is_dir() or (any(path.iterdir()) and not resume)):
        raise ValueError('Output exists; choose a new run directory or resume explicitly')
    return path


def fresh_output(path, resume=False):
    path = validate_output(path, resume)
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_config(path=None, *, preset='models.yaml', stage=None):
    from mri3d.configuration import read_configuration
    import math
    c = read_configuration(path, preset=preset)
    shape = c.get('shape')
    if (not isinstance(shape, (list, tuple)) or len(shape) != 3
            or any(not isinstance(s, int) or isinstance(s, bool) or s < 8 or s % 8 for s in shape)):
        raise ValueError('Training shape must be three multiples of 8, at least 8')
    if math.prod(shape) > 67108864:
        raise ValueError('Training shape exceeds the per-volume allocation limit')
    for key in ('epochs', 'batch_size', 'steps_per_epoch', 'threads'):
        if not isinstance(c.get(key), int) or isinstance(c[key], bool) or c[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    rate = c.get('learning_rate')
    if not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
        raise ValueError('Learning rate must be finite and positive')
    if not isinstance(c.get('limit', 1), int) or c.get('limit', 1) < 1:
        raise ValueError('Optional sample limit must be a positive integer')
    if not isinstance(c.get('seed'), int) or isinstance(c['seed'], bool) or not 0 <= c['seed'] < 2**32:
        raise ValueError('seed must be an integer in [0,2**32)')
    if c.get('prediction_type', 'epsilon') not in ('epsilon', 'v'):
        raise ValueError('prediction_type must be epsilon or v')
    if c.get('min_snr_gamma') is not None and (not isinstance(c['min_snr_gamma'], (int, float))
            or not math.isfinite(c['min_snr_gamma']) or c['min_snr_gamma'] <= 0):
        raise ValueError('min_snr_gamma must be finite and positive')
    if c.get('conditioning_mode', 'film') not in ('film', 'cross_attention'):
        raise ValueError('conditioning_mode must be film or cross_attention')
    if not isinstance(c.get('gradient_accumulation', 1), int) or c.get('gradient_accumulation', 1) < 1:
        raise ValueError('gradient_accumulation must be a positive integer')
    if not isinstance(c.get('label_dropout', 0.), (int, float)) or not 0 <= c.get('label_dropout', 0.) < 1:
        raise ValueError('label_dropout must be in [0,1)')
    if c.get('ema_decay') is not None and (not isinstance(c['ema_decay'], (int, float)) or not 0 < c['ema_decay'] < 1):
        raise ValueError('ema_decay must be in (0,1)')
    if (c.get('lr_schedule', 'none') not in ('none', 'plateau')
            or not isinstance(c.get('early_stopping_patience', 0), int) or c.get('early_stopping_patience', 0) < 0):
        raise ValueError('Invalid scheduler or early-stopping patience')
    if (not isinstance(c.get('diffusion_steps'), int) or c['diffusion_steps'] < 2
            or not isinstance(c.get('validation_draws', 1), int) or not 1 <= c.get('validation_draws', 1) <= c['diffusion_steps']):
        raise ValueError('Use >=2 diffusion steps and 1..diffusion_steps validation draws')
    multipliers = c.get('channel_mult')
    if multipliers is not None and (not isinstance(multipliers, list) or len(multipliers) < 2
            or any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in multipliers)
            or min(c['shape'])//4 < 2**(len(multipliers)-1)):
        raise ValueError('Training geometry cannot support the configured U-Net depth')
    if not isinstance(c.get('gan_weight', 0), (int, float)) or not math.isfinite(c.get('gan_weight', 0)) or c.get('gan_weight', 0) < 0:
        raise ValueError('Learning rate must be finite and GAN weight nonnegative')
    if stage != 'diffusion' or 'vq' in c:
        if not isinstance(c.get('vq'), dict):
            raise ValueError('vq must define an encoder/codebook configuration')
        for key in ('base', 'latent_channels', 'codes'):
            if not isinstance(c['vq'].get(key), int) or isinstance(c['vq'][key], bool) or c['vq'][key] < 1:
                raise ValueError(f'vq.{key} must be a positive integer')
        if c['vq']['codes'] < 2:
            raise ValueError('The vector codebook requires at least two codes')
    base, heads = c.get('unet_base'), c.get('attention_heads', 4)
    if not isinstance(base, int) or base < 2 or not isinstance(heads, int) or heads < 1:
        raise ValueError('Invalid U-Net width or attention-head count')
    if c.get('attention') and (base*(multipliers[-1] if multipliers else 2)) % heads:
        raise ValueError('Attention heads must divide the bottleneck channel count')
    for flag in ('mixed_precision', 'select_best', 'initialize_codebook', 'conditional', 'smoke_test', 'attention', 'coordinate_channels'):
        if flag in c and not isinstance(c[flag], bool):
            raise ValueError(f'{flag} must be true or false')
    return c
