"""Compact inference pairs with explicit original-checkpoint provenance."""
from copy import deepcopy
from pathlib import Path
from src.utils.integrity import sha256_file
from src.utils.logging import write_json
from .common import read_checkpoint, save_checkpoint, fresh_output


def export_inference_pair(vq_path, diffusion_path, output, weights='auto'):
    vq, diffusion = read_checkpoint(vq_path, 'vq'), read_checkpoint(diffusion_path, 'diffusion')
    if diffusion['latent_manifest']['vq_sha256'] != sha256_file(vq_path):
        raise ValueError('Cannot export checkpoints with mismatched VQ/diffusion lineage')
    selected = diffusion.get('weights_for_inference', 'raw') if weights == 'auto' else weights
    if selected not in ('raw', 'ema') or selected == 'ema' and not diffusion.get('ema_model'):
        raise ValueError('Requested export weights are unavailable')
    root = fresh_output(output)
    keep_vq = ('kind', 'model', 'model_config', 'config', 'shape', 'dataset_fingerprint', 'train_ids', 'cohort_records', 'smoke_test', 'epoch')
    compact_vq = {key: vq[key] for key in keep_vq if key in vq}
    compact_vq.update(inference_only=True, original_checkpoint_sha256=sha256_file(vq_path))
    save_checkpoint(root/'vq.pt', compact_vq)
    latent = deepcopy(diffusion['latent_manifest'])
    latent.update(vq_sha256=sha256_file(root/'vq.pt'), vq_path='vq.pt')
    compact_diffusion = dict(kind='diffusion', model=diffusion['ema_model'] if selected == 'ema' else diffusion['model'],
        model_config=diffusion['model_config'], config=diffusion['config'], latent_manifest=latent,
        smoke_test=diffusion['smoke_test'], epoch=diffusion['epoch'], inference_only=True,
        weights_for_inference='raw', exported_weight_source=selected,
        original_checkpoint_sha256=sha256_file(diffusion_path))
    save_checkpoint(root/'diffusion.pt', compact_diffusion)
    report = dict(vq=dict(file='vq.pt', sha256=sha256_file(root/'vq.pt')),
        diffusion=dict(file='diffusion.pt', sha256=sha256_file(root/'diffusion.pt'), selected_source_weights=selected),
        original_hashes=dict(vq=sha256_file(vq_path), diffusion=sha256_file(diffusion_path)),
        original_bytes=Path(vq_path).stat().st_size+Path(diffusion_path).stat().st_size,
        exported_bytes=(root/'vq.pt').stat().st_size+(root/'diffusion.pt').stat().st_size,
        note='Inference-only: optimizer, history, scheduler, scaler and RNG state omitted; source lineage retained')
    write_json(root/'manifest.json', report)
    return report
