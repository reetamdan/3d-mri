"""Verified representation artifacts shared by presentation pages and live inference."""
from collections import Counter
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from app.artifact_cache import FileSignature, file_signature, read_json, verified_hash, dataset_fingerprint
from src.utils.artifacts import resolve_reference
from src.utils.integrity import confined_path

ARM_NAMES = {
    'ae_vq_baseline': 'VQ baseline · discrete codes',
    'ae_vq_improved': 'VQ · larger codebook',
    'ae_kl_iso': 'KL · isotropic compression',
    'ae_kl_aniso': 'KL · slice-preserving compression',
    'ae_kl_aniso_ssim': 'KL · multi-scale structural loss',
    'ae_kl_aniso_structural': 'KL · structure + edge loss',
    'ae_kl_aniso_perceptual': 'KL · medical feature loss',
    'ae_kl_low_compression': 'KL · 2×2×2 spatial sampling',
    'ae_kl_aniso_deconv': 'KL · learned upsampling',
}
MODEL_ORDER = ('ae_kl_aniso', 'ae_kl_low_compression', 'ae_kl_aniso_structural', 'ae_kl_iso',
               'ae_vq_improved', 'ae_vq_baseline', 'ae_kl_aniso_ssim', 'ae_kl_aniso_perceptual', 'ae_kl_aniso_deconv')


@dataclass
class VerifiedRun:
    record: dict[str, Any]
    report: dict[str, Any]
    folder: Path
    checkpoint: Path
    data: Path
    sources: dict[str, dict[str, str]]
    split_counts: Counter


def published_results(root: Path) -> dict[str, Any] | None:
    path = root/'docs/results/ae_bakeoff_measured.json'
    if not path.is_file():
        return None
    value = read_json(path)
    if (not isinstance(value, dict) or not isinstance(value.get('groups'), dict)
            or not value.get('records') or not value.get('dataset_fingerprint')):
        raise ValueError('Published result summary is incomplete')
    if any(r.get('smoke_test', False) for r in value['records']):
        raise ValueError('Artificial runs cannot be presented as project measurements')
    return value


def verified_run(root: Path, record: dict[str, Any], fingerprint: str) -> VerifiedRun:
    study_root = confined_path(root, record['source_study'].replace('\\', '/'))
    study = read_json(study_root/'study.json')
    configs = study.get('plan', {}).get('configs', {})
    research = study.get('smoke_test') is False
    if 'smoke_test' not in study:
        # Early studies declared the smoke flag per arm, not at the document root.
        research = bool(configs) and all(c.get('smoke_test') is False for c in configs.values())
    if configs and any(c.get('smoke_test') is not False for c in configs.values()):
        research = False
    if (study.get('status') != 'completed' or not research or study['plan']['dataset_fingerprint'] != fingerprint):
        raise ValueError('Only completed matching research studies can supply images')
    matching = next((r for r in study['records'] if r['name'] == record['name']), None)
    if matching is None or any(matching[k] != record[k] for k in ('arm', 'seed', 'checkpoint_sha256')):
        raise ValueError('Published record no longer matches the completed study')
    checkpoint = confined_path(study_root, matching['checkpoint'])
    if verified_hash(checkpoint) != matching['checkpoint_sha256']:
        raise ValueError('Trained checkpoint checksum mismatch')
    folder = confined_path(study_root, f'{matching["name"]}/reconstruction')
    report = read_json(folder/'report.json')
    if (report.get('smoke_test', True) or report['dataset_fingerprint'] != fingerprint
            or report['checkpoint_sha256'] != matching['checkpoint_sha256']):
        raise ValueError('Reconstruction report has stale model or cohort lineage')
    data = resolve_reference(folder, report['input_dataset'])
    if dataset_fingerprint(data) != fingerprint:
        raise ValueError('Reconstruction source data no longer matches its report')
    with (data/'metadata.csv').open(newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    sources = {r['scan_id']: r for r in rows if r['split'] == report['split']}
    if not report['rows'] or any(r['scan_id'] not in sources for r in report['rows']):
        raise ValueError('Reconstruction identities are missing from the declared split')
    return VerifiedRun(matching, report, folder, checkpoint, data, sources, Counter(r['split'] for r in rows))


@st.cache_data(show_spinner=False, max_entries=24)
def _image(signature: FileSignature, checksum: str, shape: tuple[int, ...]) -> np.ndarray:
    path = Path(signature[0])
    if not checksum or verified_hash(path) != checksum:
        raise ValueError('MRI image checksum mismatch')
    array = np.load(path, allow_pickle=False)
    if (array.shape != (1, *shape) or not np.issubdtype(array.dtype, np.floating)
            or not np.isfinite(array).all() or array.min() < -1.0001 or array.max() > 1.0001):
        raise ValueError('MRI image has invalid shape, values or normalization')
    if file_signature(path) != signature:
        raise ValueError('MRI image changed while loading')
    return array[0]


def image_pair(run: VerifiedRun, identity: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    row = next(r for r in run.report['rows'] if r['scan_id'] == identity)
    source = run.sources[identity]
    shape = tuple(run.report['shape'])
    original = confined_path(run.data, source['output_file'])
    restored = confined_path(run.folder, row['file'])
    return (_image(file_signature(original), source['output_sha256'], shape),
            _image(file_signature(restored), row['sha256'], shape), row)


@st.cache_resource(show_spinner=False, max_entries=2)
def _trained_model(path: str, checksum: str, device: str) -> Any:
    from src.training.representation import load_representation
    if verified_hash(path) != checksum:
        raise ValueError('Live model checksum mismatch')
    return load_representation(path, device)[0]


def live_reconstruction(run: VerifiedRun, original: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    import torch
    from src.models.autoencoder import encode_latent, decode_latent
    from src.training.common import device_for
    from src.metrics import reconstruction_metrics
    torch.set_num_threads(4)
    device = device_for()
    model = _trained_model(str(run.checkpoint), run.record['checkpoint_sha256'], str(device))
    with torch.inference_mode():
        x = torch.from_numpy(original.copy())[None, None].to(device)
        result = decode_latent(model, encode_latent(model, x), project=False)[0, 0].float().cpu().numpy()
    return result, reconstruction_metrics(original, result)


def result_frame(summary: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for arm, group in summary['groups'].items():
        metrics = group['metrics']
        rows.append({'Model': ARM_NAMES.get(arm, arm), 'Seeds': len(group['seeds']),
            'Spatial reduction': ' × '.join(map(str, group['spatial_compression'])),
            'PSNR (dB)': metrics['psnr']['mean'], 'PSNR seed SD': metrics['psnr']['seed_sd'],
            'SSIM': metrics['ssim']['mean'], 'SSIM seed SD': metrics['ssim']['seed_sd'],
            'MS-SSIM': metrics['ms_ssim']['mean'], 'Edge energy ratio': metrics['gradient_energy_ratio']['mean'],
            'HF energy ratio': metrics['high_frequency_energy_ratio']['mean'],
            'Peak training VRAM (MiB)': group['train_peak_mib']})
    return pd.DataFrame(rows)
