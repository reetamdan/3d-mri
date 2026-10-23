"""Controlled stage-one experiments using existing dataset, checkpoint and RNG contracts."""
from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from math import prod
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from mri3d.configuration import read_configuration
from src.models.autoencoder import build_autoencoder, autoencoder_step, encode_latent, decode_latent
from src.models.vqvae import VQVAE3D
from src.metrics import reconstruction_metrics, downsample_reconstruction_baseline
from src.utils.cohort import verify_cohort, has_tensor_hashes
from src.utils.artifacts import reference
from src.utils.integrity import sha256_file
from src.utils.logging import write_json
from .common import TrainingDataset, fresh_output, read_checkpoint, save_checkpoint, device_for
from .experiments import setup
from .protocol import evaluation_context, audit_overlap
from .state import precision, rng_state, restore_rng, scheduler_for
from .objectives import ReconstructionObjective, calibrate_weights, ms_ssim_score


def representation_config(path: str | Path | None = None, *, preset='ae_vq_baseline.yaml') -> dict[str, Any]:
    c = read_configuration(path, preset=preset)
    if not isinstance(c.get('model'), dict) or c['model'].get('family') not in ('vq', 'kl'):
        raise ValueError('Representation configuration requires a vq/kl model')
    if len(c.get('shape', [])) != 3 or any(not isinstance(v, int) or v < 8 or v % 4 for v in c['shape']):
        raise ValueError('Representation shape must contain three positive multiples of four >=8')
    for key in ('batch_size', 'epochs', 'steps_per_epoch', 'threads'):
        if not isinstance(c.get(key), int) or c[key] < 1:
            raise ValueError(f'{key} must be positive')
    if not 0 <= c.get('seed', -1) < 2**32 or not np.isfinite(c.get('learning_rate', np.nan)) or c['learning_rate'] <= 0:
        raise ValueError('Invalid seed/learning rate')
    c.setdefault('gradient_accumulation', 1)
    if c['gradient_accumulation'] < 1:
        raise ValueError('gradient_accumulation must be positive')
    c.setdefault('loss', {})
    if c['loss'].get('feature_path') and path:
        c['loss']['feature_path'] = str((Path(path).resolve().parent/c['loss']['feature_path']).resolve())
    if c.get('init_checkpoint') and path:
        c['init_checkpoint'] = str((Path(path).resolve().parent/c['init_checkpoint']).resolve())
    c.setdefault('smoke_test', False)
    return c


def environment(device='auto') -> dict[str, Any]:
    active = device_for(device)
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=False)
    return dict(git_commit=commit.stdout.strip() if commit.returncode == 0 else None,
                torch=str(torch.__version__), cuda=torch.version.cuda,
                monai=importlib.metadata.version('monai'),
                hardware=torch.cuda.get_device_name(active) if active.type == 'cuda' else 'CPU', device=str(active),
                memory_policy={key: os.environ.get(key) for key in ('CUDA_MODULE_LOADING', 'CUDNN_V8_API_LRU_CACHE_LIMIT', 'ONEDNN_PRIMITIVE_CACHE_CAPACITY')},
                source_sha256={name: sha256_file(Path(__file__).parents[1]/name)
                               for name in ('models/autoencoder.py', 'training/objectives.py', 'training/representation.py')})


def load_representation(path, device='cpu') -> tuple[nn.Module, dict[str, Any]]:
    ck = read_checkpoint(path)
    if ck['kind'] == 'vq':
        spec = dict(family='vq', **ck['model_config'])
    elif ck['kind'] == 'autoencoder':
        spec = ck['model_config']
    else:
        raise ValueError('Expected a VQ or representation checkpoint')
    model = build_autoencoder(spec).to(device)
    model.load_state_dict(ck['model'])
    return model.eval(), ck


def _objective(model: nn.Module, train: torch.Tensor, c: dict, device) -> tuple[ReconstructionObjective, dict]:
    weights = dict(c.get('loss', {}))
    calibration = weights.pop('calibrate_structural', False)
    perceptual_calibration = weights.pop('calibrate_perceptual', False)
    perceptual_fraction = weights.pop('perceptual_gradient_fraction', .25)
    if calibration or perceptual_calibration:
        model.eval()
        with torch.no_grad():
            first = train[:c['batch_size']].to(device)
            prediction = decode_latent(model, encode_latent(model, first), project=False)
    if calibration:
        fitted = calibrate_weights(prediction, first, fractions=tuple(weights.pop('gradient_fractions', [.25, .25])))
        weights.update({k: fitted[k] for k in ('ssim', 'gradient')})
        weights['calibration'] = fitted['calibration']
    feature_path = weights.get('feature_path')
    feature_hash = weights.get('feature_sha256')
    feature_model = None
    if feature_path:
        if not feature_hash or sha256_file(feature_path) != feature_hash:
            raise ValueError('Explicit frozen feature weights require a matching SHA-256')
        feature_model = torch.jit.load(feature_path, map_location=device).eval()
        for parameter in feature_model.parameters():
            parameter.requires_grad_(False)
    if perceptual_calibration:
        if feature_model is None:
            raise ValueError('Perceptual calibration requires explicit frozen feature weights')
        with torch.enable_grad():
            candidate = prediction.detach().float().requires_grad_(True)
            original_features = feature_model(first.float()).detach().flatten(1)
            candidate_features = feature_model(candidate).flatten(1)
            feature_loss = F.mse_loss(F.normalize(candidate_features, dim=1), F.normalize(original_features, dim=1))
            l1_norm = float(torch.autograd.grad(F.l1_loss(candidate, first.float()), candidate, retain_graph=True)[0].norm())
            feature_norm = float(torch.autograd.grad(feature_loss, candidate)[0].norm())
        if feature_norm <= 1e-12 or not np.isfinite(feature_norm):
            raise ValueError('Frozen perceptual features have degenerate input gradients')
        weights['perceptual'] = perceptual_fraction*l1_norm/feature_norm
        weights.setdefault('calibration', {}).update(perceptual_gradient_norm=feature_norm,
                                                   perceptual_target_fraction=perceptual_fraction)
    specification = deepcopy(weights)
    for key in ('calibration', 'feature_path', 'feature_sha256'):
        weights.pop(key, None)
    if feature_model is not None:
        weights['feature_model'] = feature_model
    return ReconstructionObjective(**weights).to(device), specification


def train_representation(root, output, config: dict, resume=False) -> str:
    c = deepcopy(config)
    device = setup(c)
    cohort = verify_cohort(root)
    train_set = TrainingDataset(root, 'train', c['shape'], c.get('limit'), strict_shape=True)
    valid_set = TrainingDataset(root, 'validation', c['shape'], c.get('limit'), strict_shape=True)
    if not len(train_set) or (not len(valid_set) and not c.get('smoke_test')):
        raise ValueError('Representation training needs training and validation cases')
    train = torch.stack([train_set[i][0] for i in range(len(train_set))])
    valid = torch.stack([valid_set[i][0] for i in range(len(valid_set))]) if len(valid_set) else None
    out = fresh_output(output, resume)
    model = build_autoencoder(c['model']).to(device)
    pretraining_records, pretraining = [], None
    if c.get('init_checkpoint') and not resume:
        initial, source_ck = load_representation(c['init_checkpoint'], device)
        if type(initial) is not type(model) or initial.config != model.config:
            raise ValueError('Pretraining initialization requires exactly the same architecture/sampling grid')
        pretraining_records = [r for r in source_ck.get('cohort_records', []) if r['split'] in ('train', 'validation')]
        pretraining_records += source_ck.get('representation_pretraining_records', [])
        if not pretraining_records:
            raise ValueError('Pretrained representation needs explicit cohort lineage')
        if not has_tensor_hashes(pretraining_records):
            raise ValueError('Pretraining cohort content hashes are incomplete')
        overlap = audit_overlap(pretraining_records, [r for r in cohort.records() if r['split'] == 'validation'])
        if overlap['overlap_counts']:
            raise ValueError('Pretraining overlaps the representation validation cohort')
        model.load_state_dict(initial.state_dict()); del initial
        pretraining = dict(checkpoint_sha256=sha256_file(c['init_checkpoint']), dataset_fingerprint=source_ck['dataset_fingerprint'])
    if isinstance(model, VQVAE3D) and c.get('initialize_codebook', True) and not resume and not c.get('init_checkpoint'):
        with torch.no_grad():
            features = model.encoder(train[:c['batch_size']].to(device)).movedim(1, -1).reshape(-1, c['model']['latent_channels'])
            model.quantizer.initialize(features)
    optimizer = torch.optim.Adam(model.parameters(), lr=c['learning_rate'])
    amp, dtype, scaler = precision(c, device)
    scheduler = scheduler_for(optimizer, c)
    history: list[dict[str, Any]] = []
    start = 0
    if resume:
        ck = read_checkpoint(out/'last.pt', 'autoencoder')
        old, new = dict(ck['config']), dict(c); old.pop('epochs'); new.pop('epochs')
        if old != new or ck['dataset_fingerprint'] != cohort.fingerprint:
            raise ValueError('Resume requires the same representation config and cohort except total epochs')
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer'])
        history, start = ck['history'], ck['epoch']
        pretraining_records, pretraining = ck.get('representation_pretraining_records', []), ck.get('pretraining')
        if ck['scaler']:
            scaler.load_state_dict(ck['scaler'])
        if scheduler and ck['scheduler']:
            scheduler.load_state_dict(ck['scheduler'])
        resume_c = dict(c, loss={k: v for k, v in ck['objective'].items() if k != 'calibration'})
        objective, _ = _objective(model, train, resume_c, device)
        objective_spec = ck['objective']
        restore_rng(ck, device)
    else:
        objective, objective_spec = _objective(model, train, c, device)
    # Mean KL makes the coefficient invariant to latent lattice size. This is
    # intentionally not MONAI tutorial's unnormalized spatial sum; it is a weak
    # stage-one regularizer, and actual posterior/channel statistics are audited.
    reference_elements = 1
    env = environment(device)
    prior = json.loads((out/'experiment.json').read_text()) if resume and (out/'experiment.json').is_file() else {}
    now = datetime.now(timezone.utc).isoformat()
    sessions = prior.get('sessions', [])+[dict(started_at=now, resume_epoch=start, environment=env)]
    write_json(out/'experiment.json', dict(config=c, objective=objective_spec, dataset_fingerprint=cohort.fingerprint,
        environment=env, started_at=prior.get('started_at', now), sessions=sessions, pretraining=pretraining,
        kl_reduction='mean over batch/channels/spatial lattice', tensor_layout='[B,C,row,column,slice]'))
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(start, c['epochs']):
        began = time.perf_counter(); model.train()
        loader = DataLoader(TensorDataset(train), batch_size=c['batch_size'], shuffle=True,
                            generator=torch.Generator().manual_seed(c['seed']+epoch))
        batches = min(len(loader), c['steps_per_epoch']*c['gradient_accumulation'])
        values, updates = [], 0
        optimizer.zero_grad(set_to_none=True)
        for step, (x,) in enumerate(loader):
            if step >= batches:
                break
            x = x.to(device)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
                restored, regularizer = autoencoder_step(model, x, reference_elements)
                loss, terms = objective(restored, x, regularizer)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite representation objective')
            accumulation = c['gradient_accumulation']
            group = min(accumulation, batches-(step//accumulation)*accumulation)
            scaler.scale(loss/group).backward()
            if (step+1) % accumulation == 0 or step+1 == batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                updates += 1
            values.append(dict(loss=float(loss.detach()), **{k: float(v.detach()) for k, v in terms.items()}))
        model.eval(); total, count = 0., 0
        if valid is not None:
            with torch.no_grad():
                for (x,) in DataLoader(TensorDataset(valid), batch_size=c['batch_size']):
                    x = x.to(device)
                    restored = decode_latent(model, encode_latent(model, x), project=False)
                    total += float(F.l1_loss(restored, x))*len(x); count += len(x)
        record = dict(epoch=epoch+1, optimizer_steps=updates, micro_batches=len(values),
                      validation_l1=total/count if count else None, seconds=time.perf_counter()-began,
                      **{key: float(np.mean([r[key] for r in values])) for key in values[0]})
        history.append(record)
        if scheduler:
            scheduler.step(record['validation_l1'] if count else record['l1'])
        payload = dict(kind='autoencoder', model=model.state_dict(), model_config=c['model'], config=c,
            shape=c['shape'], epoch=epoch+1, history=history, optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(), scheduler=scheduler.state_dict() if scheduler else None,
            objective=objective_spec, **rng_state(), dataset_fingerprint=cohort.fingerprint,
            representation_pretraining_records=pretraining_records, pretraining=pretraining,
            train_ids=train_set.ids, cohort_records=cohort.records(), smoke_test=c.get('smoke_test', False))
        save_checkpoint(out/'last.pt', payload)
        if count and record['validation_l1'] <= min(r['validation_l1'] for r in history):
            save_checkpoint(out/'best.pt', payload)
        write_json(out/'history.json', history)
        selected = out/('best.pt' if (out/'best.pt').is_file() else 'last.pt')
        write_json(out/'run.json', dict(kind='autoencoder', completed_epochs=epoch+1, train_count=len(train_set),
            validation_count=len(valid_set), shape=c['shape'], checkpoint_file=selected.name,
            checkpoint_sha256=sha256_file(selected), dataset_fingerprint=cohort.fingerprint,
            smoke_test=c.get('smoke_test', False), data_root=reference(root, out),
            finished_at=datetime.now(timezone.utc).isoformat(),
            peak_cuda_mib=torch.cuda.max_memory_allocated()/1024**2 if device.type == 'cuda' else None))
        print(f'{out.name} epoch {epoch+1}/{c["epochs"]}: L1={record["l1"]:.5f} val={record["validation_l1"]}', flush=True)
    return str(out/('best.pt' if (out/'best.pt').is_file() else 'last.pt'))


def spectral_edge_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    original = [np.diff(a.astype(np.float64), axis=i) for i in range(3)]
    restored = [np.diff(b.astype(np.float64), axis=i) for i in range(3)]
    energy = lambda axes: float(np.mean([np.square(x).mean() for x in axes]))
    frequency = np.meshgrid(np.fft.fftfreq(a.shape[0]), np.fft.fftfreq(a.shape[1]), np.fft.rfftfreq(a.shape[2]), indexing='ij')
    high = sum(f*f for f in frequency) >= .35**2
    def high_energy(x):
        spectrum = np.abs(np.fft.rfftn(x.astype(np.float64)-x.mean()))**2
        return float(spectrum[high].sum())
    return dict(gradient_energy_ratio=energy(restored)/max(energy(original), 1e-12),
                edge_mae=float(np.mean([np.abs(x-y).mean() for x, y in zip(original, restored)])),
                high_frequency_energy_ratio=high_energy(b)/max(high_energy(a), 1e-12))


@torch.no_grad()
def evaluate_representation(root, checkpoint, output, device='auto') -> dict[str, Any]:
    device = device_for(device)
    model, ck = load_representation(checkpoint, device)
    context = evaluation_context(root, ck, 'validation')
    data = TrainingDataset(root, 'validation', ck['shape'], strict_shape=True)
    if not len(data):
        raise ValueError('No validation cases to evaluate')
    out = fresh_output(output); rows = []
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    began = time.perf_counter()
    for i, row in enumerate(data.rows):
        tensor = data[i][0]; a = tensor[0].numpy()
        latent = encode_latent(model, tensor[None].to(device))
        b = decode_latent(model, latent, project=False)[0, 0].cpu().numpy()
        reference_array, baseline = downsample_reconstruction_baseline(a)
        path = out/f'{row["scan_id"]}.npy'; np.save(path, b[None], allow_pickle=False)
        measured = reconstruction_metrics(a, b)
        measured.update(ms_ssim=float(ms_ssim_score(torch.from_numpy(b)[None, None], tensor[None])),
                        **spectral_edge_metrics(a, b))
        baseline.update(ms_ssim=float(ms_ssim_score(torch.from_numpy(reference_array)[None, None], tensor[None])),
                        **spectral_edge_metrics(a, reference_array))
        rows.append(dict(scan_id=row['scan_id'], file=path.name, sha256=sha256_file(path), **measured,
                         interpolation=baseline))
    keys = tuple(k for k in rows[0] if k not in ('scan_id', 'file', 'sha256', 'interpolation'))
    report = dict(rows=rows, means={key: float(np.mean([r[key] for r in rows])) for key in keys},
        interpolation={key: float(np.mean([r['interpolation'][key] for r in rows])) for key in keys},
        shape=ck['shape'], latent_shape=list(latent.shape[1:]), scalar_compression=prod(ck['shape'])/latent[0].numel(),
        spatial_compression=[a//b for a, b in zip(ck['shape'], latent.shape[2:])],
        parameters=sum(p.numel() for p in model.parameters()), seconds=time.perf_counter()-began,
        inference_peak_cuda_mib=torch.cuda.max_memory_allocated()/1024**2 if device.type == 'cuda' else None,
        checkpoint_sha256=sha256_file(checkpoint), dataset_fingerprint=ck['dataset_fingerprint'],
        input_dataset=reference(root, out), split='validation', evaluation_context=context,
        smoke_test=ck['smoke_test'], metric_protocol='Legacy range-2 SSIM; adaptive Gaussian window-3 MS-SSIM; unit-array-grid derivatives/frequencies',
        anatomy_proxy=None, anatomical_roi=None)
    write_json(out/'report.json', report)
    return report
