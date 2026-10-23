"""Predeclared, matched-update VQ/KL study on a frozen processed knee cohort."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys
import time
from typing import Any

if __name__ == '__main__' and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from src.models.autoencoder import build_autoencoder, autoencoder_step, encode_latent
from src.models.vqvae import VQVAE3D
from src.training.common import TrainingDataset
from src.training.experiments import setup
from src.training.representation import (representation_config, train_representation, evaluate_representation,
                                         _objective, environment)
from src.training.state import precision
from src.training.evidence import slice_grid
from src.training.statistics import bootstrap_interval
from src.utils.cohort import verify_cohort
from src.utils.integrity import generated_volume, sha256_file
from src.utils.logging import write_json
from src.utils.artifacts import reference

DEFAULT_ARMS = ('ae_vq_baseline', 'ae_vq_improved', 'ae_kl_iso', 'ae_kl_aniso', 'ae_kl_aniso_ssim', 'ae_kl_aniso_structural')
ARMS = (*DEFAULT_ARMS, 'ae_kl_aniso_perceptual', 'ae_kl_low_compression', 'ae_kl_aniso_deconv')
GATE: dict[str, Any] = dict(psnr_tolerance_db=.1, ssim_tolerance=0., ms_ssim_tolerance=0.,
            gradient_ratio_min=.6, gradient_ratio_max=1.3, gradient_deviation_improvement=.20,
            high_frequency_ratio_min=.5, high_frequency_ratio_max=1.5,
            edge_error_no_worse=True, anatomy_proxy='Unavailable; no masks or validated knee model supplied',
            visual_review='Inspect fixed baseline best/median/worst and seeded random cases; no severe new artifacts')


def read_json(path):
    return json.loads(Path(path).read_text())


def gate_result(measured: dict, frozen: dict, interpolation: dict) -> dict[str, bool]:
    return dict(psnr=measured['psnr'] >= frozen['psnr']-GATE['psnr_tolerance_db'],
        ssim=measured['ssim'] >= interpolation['ssim'], ms_ssim=measured['ms_ssim'] >= interpolation['ms_ssim'],
        gradient_range=GATE['gradient_ratio_min'] <= measured['gradient_energy_ratio'] <= GATE['gradient_ratio_max'],
        gradient_improvement=abs(1-measured['gradient_energy_ratio']) <= (1-GATE['gradient_deviation_improvement'])*abs(1-frozen['gradient_energy_ratio']),
        high_frequency_range=GATE['high_frequency_ratio_min'] <= measured['high_frequency_energy_ratio'] <= GATE['high_frequency_ratio_max'],
        edge_error=measured['edge_mae'] <= frozen['edge_mae'])


def profile(root, config, updates=100) -> dict[str, Any]:
    device = setup(config)
    dataset = TrainingDataset(root, 'train', config['shape'])
    train = torch.stack([dataset[i][0] for i in range(config['batch_size'])])
    model = build_autoencoder(config['model']).to(device)
    if isinstance(model, VQVAE3D):
        with torch.no_grad():
            features = model.encoder(train.to(device)).movedim(1, -1).reshape(-1, config['model']['latent_channels'])
            model.quantizer.initialize(features)
    objective, weights = _objective(model, train, config, device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    amp, dtype, scaler = precision(config, device)
    x = train.to(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    timings, milestones = [], []
    for step in range(updates):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        began = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
            restored, regularizer = autoencoder_step(model, x, 1)
            loss, _ = objective(restored, x, regularizer)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite profile loss')
        optimizer.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        scaler.step(optimizer); scaler.update()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        timings.append(time.perf_counter()-began)
        if step+1 in (1, 10, 100):
            milestones.append(dict(updates=step+1, loss=float(loss.detach())))
    model.eval()
    with torch.no_grad():
        latent = encode_latent(model, x)
    duration = float(np.median(timings[min(10, len(timings)-1):]))
    return dict(updates=updates, repeated_training_exams=len(train), milestones=milestones,
                seconds_per_update=duration, estimated_training_seconds=duration*config['epochs']*config['steps_per_epoch'],
                peak_cuda_mib=torch.cuda.max_memory_allocated()/1024**2 if device.type == 'cuda' else None,
                parameters=sum(p.numel() for p in model.parameters()), latent_shape=list(latent.shape[1:]),
                estimated_two_full_checkpoint_mib=sum(p.numel() for p in model.parameters())*4*3*2/1024**2,
                objective=weights, purpose='Compute/finite-gradient smoke only; not a validation result')


def comparisons(root, data, frozen, records):
    dataset = TrainingDataset(data, 'validation', frozen['shape'])
    by_id = {r['scan_id']: i for i, r in enumerate(dataset.rows)}
    ranked = sorted(frozen['rows'], key=lambda r: r['psnr'])
    selected = [ranked[0]['scan_id'], ranked[len(ranked)//2]['scan_id'], ranked[-1]['scan_id']]
    selected.extend(np.random.default_rng(42).choice(list(by_id), size=min(3, len(by_id)), replace=False).tolist())
    selected = list(dict.fromkeys(selected))
    reports = {r['name']: read_json(root/r['name']/'reconstruction/report.json') for r in records}
    from src.metrics import downsample_reconstruction_baseline
    for case, identity in enumerate(selected):
        x = dataset[by_id[identity]][0][0].numpy()
        arrays = [x, downsample_reconstruction_baseline(x)[0]]
        labels = ['Original normalized input', 'Interpolation reference']
        row = next(r for r in frozen['rows'] if r['scan_id'] == identity)
        arrays.append(generated_volume(root/'frozen-vq', row, frozen['shape'])[0]); labels.append('Historical VQ checkpoint')
        for name, report in reports.items():
            row = next(r for r in report['rows'] if r['scan_id'] == identity)
            arrays.append(generated_volume(root/name/'reconstruction', row, report['shape'])[0]); labels.append(name)
        slice_grid(arrays, labels, root/f'comparison-{case+1}.png')
        slice_grid([np.clip(np.abs(x-a)/.25-1, -1, 1) for a in arrays[1:]],
                   [f'Absolute error: {label} (fixed range 0..0.5)' for label in labels[1:]], root/f'errors-{case+1}.png')
        slices = tuple(slice(n//4, 3*n//4) for n in x.shape[:2])+(slice(None),)
        slice_grid([a[slices] for a in arrays], [f'Central grid ROI: {label}' for label in labels], root/f'grid-roi-{case+1}.png')
    write_json(root/'visual-cases.json', dict(selection='Frozen-VQ worst/median/best PSNR + seed-42 random cases',
               ids=selected, roi='Fixed central array-grid crop, not an ACL/meniscus annotation'))


def run(args):
    data, root = Path(args.data).resolve(), Path(args.output).resolve()
    snapshot = verify_cohort(data)
    configs = {}
    for name in args.arms:
        path = Path(args.config_dir)/f'{name}.yaml'
        config = representation_config(path) if path.is_file() or args.config_dir != 'configs' else representation_config(preset=f'{name}.yaml')
        if config['loss'].get('calibrate_perceptual'):
            if not args.feature_extractor:
                raise ValueError('Perceptual arm requires --feature-extractor with a provenance-checked export')
            feature = Path(args.feature_extractor).resolve()
            info = read_json(feature.parent/'extractor.json')
            if sha256_file(feature) != info['feature_sha256']:
                raise ValueError('Perceptual descriptor hash mismatch')
            config['loss'].update(feature_path=str(feature), feature_sha256=info['feature_sha256'])
        configs[name] = config
    budgets = {(c['epochs'], c['steps_per_epoch'], c['batch_size']*c['gradient_accumulation']) for c in configs.values()}
    if len(budgets) != 1:
        raise ValueError('Study arms must share epochs, optimizer updates and effective batch budget')
    content = dict(configs=configs, seeds=args.seeds, dataset_fingerprint=snapshot.fingerprint, gate=GATE,
                   run_device=args.device,
                   baseline_sha256=sha256_file(args.baseline), primary_scope='Exploratory model-selection validation',
                   kl_reduction='Mean, dimension-invariant; coefficient is not directly interchangeable with tutorial summed KL')
    if root.exists() and not args.resume:
        raise ValueError('Study output exists; resume the same plan explicitly')
    root.mkdir(parents=True, exist_ok=True)
    if (root/'plan.json').is_file():
        if read_json(root/'plan.json')['content'] != content:
            raise ValueError('Study plan changed; choose a new run')
    else:
        write_json(root/'plan.json', dict(content=content, created_at=datetime.now(timezone.utc).isoformat(), environment=environment(args.device)))
    if args.profile_only:
        results = {}
        for name, config in configs.items():
            print(f'PROFILE {name}', flush=True)
            results[name] = profile(data, dict(config, device=args.device), args.profile_updates)
            write_json(root/'profiles.json', results)
            print(json.dumps({k: results[name][k] for k in ('parameters', 'latent_shape', 'seconds_per_update', 'peak_cuda_mib', 'estimated_training_seconds')}), flush=True)
            gc.collect(); torch.cuda.empty_cache()
        return
    if not (root/'frozen-vq/report.json').is_file():
        evaluate_representation(data, args.baseline, root/'frozen-vq', args.device)
    frozen = read_json(root/'frozen-vq/report.json')
    records = []
    for seed in args.seeds:
        for name, original in configs.items():
            config = deepcopy(original); config.update(seed=seed, device=args.device)
            experiment = root/f'{name}-{seed}'
            print(f'STUDY {experiment.name}', flush=True)
            checkpoint = train_representation(data, experiment/'training', config,
                resume=(experiment/'training/last.pt').is_file())
            if (experiment/'reconstruction/report.json').is_file():
                report = read_json(experiment/'reconstruction/report.json')
                if report['checkpoint_sha256'] != sha256_file(checkpoint):
                    raise ValueError('Existing representation report has stale checkpoint lineage')
            else:
                report = evaluate_representation(data, checkpoint, experiment/'reconstruction', args.device)
            checks = gate_result(report['means'], frozen['means'], report['interpolation'])
            record = dict(name=experiment.name, arm=name, seed=seed, metrics=report['means'],
                smoke_test=report['smoke_test'],
                parameters=report['parameters'], compression=report['spatial_compression'], latent_shape=report['latent_shape'],
                training_peak_cuda_mib=read_json(experiment/'training/run.json')['peak_cuda_mib'],
                inference_peak_cuda_mib=report['inference_peak_cuda_mib'], quantitative_gate=checks,
                quantitative_gate_pass=all(checks.values()), visual_review='pending',
                checkpoint=reference(checkpoint, root), checkpoint_sha256=sha256_file(checkpoint),
                paired_exam_intervals={key: bootstrap_interval([r[key]-next(b for b in frozen['rows'] if b['scan_id'] == r['scan_id'])[key]
                    for r in report['rows']]) for key in ('psnr', 'ssim', 'edge_mae')})
            records.append(record)
            write_json(experiment/'summary.json', record)
            write_json(root/'study.json', dict(plan=content, records=records, independent_test=False,
                       smoke_test=any(c.get('smoke_test', False) for c in configs.values()),
                       anatomy_metrics_available=False, status='in_progress'))
            gc.collect(); torch.cuda.empty_cache()
    comparisons(root, data, frozen, records)
    write_json(root/'study.json', dict(plan=content, records=records, frozen_baseline=frozen['means'],
               smoke_test=any(c.get('smoke_test', False) for c in configs.values()),
               interpolation=frozen['interpolation'], independent_test=False, anatomy_metrics_available=False,
               status='completed', completed_at=datetime.now(timezone.utc).isoformat()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/processed/mrnet-m4-20261021')
    parser.add_argument('--output', default='runs/AE-BAKEOFF')
    parser.add_argument('--baseline', default='runs/M4-MRNET-20261021/vq/best.pt')
    parser.add_argument('--config-dir', default='configs')
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(DEFAULT_ARMS))
    parser.add_argument('--feature-extractor')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--profile-only', action='store_true')
    parser.add_argument('--profile-updates', type=int, default=100)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if (args.profile_updates < 1 or not args.seeds or len(set(args.seeds)) != len(args.seeds)
            or any(not 0 <= seed < 2**32 for seed in args.seeds)):
        parser.error('Use positive profile updates and distinct seeds')
    run(args)


if __name__ == '__main__':
    main()
