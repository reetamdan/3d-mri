"""Validated YAML configuration; relative paths resolve against the YAML file."""
from copy import deepcopy
from pathlib import Path
import math
import yaml


def validate_config(config: dict) -> dict:
    c = deepcopy(config)
    p = c['preprocessing']
    for key in ('target_shape', 'crop_size', 'target_spacing'):
        value = p.get(key)
        if value is None and key != 'target_shape':
            continue
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError(f'{key} must contain three positive numbers')
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in value):
            raise ValueError(f'Invalid {key}')
        if key != 'target_spacing' and any(int(v) != v for v in value):
            raise ValueError(f'{key} must contain integers')
    if p.get('normalization') not in ('minmax', 'zscore'):
        raise ValueError('normalization must be minmax or zscore')
    limits = p.get('normalization_range', [-1, 1])
    if len(limits) != 2 or not all(math.isfinite(x) for x in limits) or limits[0] >= limits[1]:
        raise ValueError('normalization_range must be finite and increasing')
    if p.get('padding_mode', 'constant') not in ('constant', 'edge', 'reflect'):
        raise ValueError('Unsupported padding mode')
    ratios = [c['split'][k] for k in ('train', 'validation', 'test')]
    if any(not math.isfinite(x) or x < 0 for x in ratios) or not math.isclose(sum(ratios), 1):
        raise ValueError('Split ratios must be nonnegative and sum to 1')
    if not isinstance(c['seed'], int) or c['seed'] < 0 or c['seed'] >= 2**32:
        raise ValueError('seed must be an integer in [0, 2**32)')
    if c.get('export_format', 'npy') not in ('npy', 'pt'):
        raise ValueError('export_format must be npy or pt')
    limit = p.get('max_voxels', 67108864)
    if not isinstance(limit, int) or limit <= 0:
        raise ValueError('max_voxels must be a positive integer')
    for key in ('target_shape', 'crop_size'):
        if p.get(key) and math.prod(p[key]) > limit:
            raise ValueError(f'{key} exceeds max_voxels')
    return c


def load_config(path: str | Path = 'configs/config.yaml') -> dict:
    path = Path(path).resolve()
    with path.open(encoding='utf-8') as f:
        c = validate_config(yaml.safe_load(f))
    for key in ('raw_dir', 'processed_dir', 'metadata_dir', 'manifest'):
        if c['dataset'].get(key):
            c['dataset'][key] = str((path.parent / c['dataset'][key]).resolve())
    c['logs_dir'] = str((path.parent / c.get('logs_dir', '../logs')).resolve())
    return c
