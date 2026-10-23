"""Joint-label population control for future qualified augmentation experiments."""
from collections import Counter
import json

from .common import fresh_output, read_checkpoint
from .experiments import generate
from src.utils.logging import write_json


def generate_balanced(autoencoder, diffusion, output, count=40, seed=42, device='auto',
                      sampler='ddpm', sampling_steps=None, guidance_scale=1.):
    checkpoint = read_checkpoint(diffusion, 'diffusion')
    manifest = checkpoint['latent_manifest']
    if checkpoint['model_config']['label_dim'] != 3:
        raise ValueError('Balanced pathology requests require a three-label conditional model')
    support = Counter(tuple(r['labels']) for r in manifest['rows'] if r['split'] == 'train')
    if not support or any(len(c) != 3 or any(v not in (0, 1) for v in c) for c in support) or count < len(support):
        raise ValueError('Need genuine binary training support and at least one sample per observed joint condition')
    out = fresh_output(output)
    combined, common, offset = [], None, 0
    for i, condition in enumerate(sorted(support)):
        size = count//len(support)+int(i < count % len(support))
        folder = out/f'condition-{i:02d}'
        generate(autoencoder, diffusion, folder, count=size, seed=seed+offset, labels=list(condition), device=device,
                 sampler=sampler, sampling_steps=sampling_steps, guidance_scale=guidance_scale)
        current = json.loads((folder/'manifest.json').read_text())
        if common is None:
            common = {k: v for k, v in current.items() if k not in ('rows', 'conditioning_training_support')}
        for row in current['rows']:
            for field in ('file', 'latent_file'):
                row[field] = f'{folder.name}/{row[field]}'
            combined.append(row)
        offset += size
    result = dict(common or {}, rows=combined, condition_policy='uniform observed training joint-label strata',
                  conditioning_training_support_by_vector={','.join(map(str, c)): n for c, n in support.items()},
                  label_semantics='Requested conditions only; not verified synthetic pathology')
    write_json(out/'manifest.json', result)
    return str(out/'manifest.json')
