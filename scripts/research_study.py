"""Run a declared multi-seed control/upgrade study on one frozen real cohort."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time

if __name__ == '__main__' and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from src.data.mrnet import prepare_mrnet
from src.training.common import model_config, fingerprint, TrainingDataset, read_checkpoint
from src.training.experiments import train_vq, cache_latents, train_diffusion, generate, evaluate_reconstruction, load_vq
from src.training.evaluation import evaluate_generation
from src.training.diagnostics import tokenizer_diagnostics
from src.training.statistics import bootstrap_interval
from src.training.evidence import curve, slice_grid
from src.utils.logging import write_json
from src.utils.artifacts import reference, resolve_reference
from src.utils.integrity import sha256_file


def run(args):
    root, data, source = Path(args.output).resolve(), Path(args.data).resolve(), Path(args.source).resolve()
    if root.exists() and not args.resume:
        raise ValueError('Study exists; resume explicitly or select a new output')
    root.mkdir(parents=True, exist_ok=True)
    if not data.exists():
        print('Preparing the frozen sagittal cohort', flush=True)
        prepare_mrnet(source, data, 'sagittal', shape=(128, 128, 32))
    data_hash = fingerprint(data)
    plan = dict(schema_version=1, seeds=args.seeds, vq_epochs=args.vq_epochs, diffusion_epochs=args.diffusion_epochs,
        steps_per_epoch=args.steps, generated_count=args.count, dataset_fingerprint=data_hash,
        baseline=model_config('configs/research-baseline.yaml'), upgraded=model_config('configs/research-v2.yaml'),
        primary_endpoint='Validation reconstruction SSIM; paired exam bootstrap; not an independent test',
        study_type='Exploratory matched optimizer budgets; combined upgrade, not a factorial causal ablation',
        clinical_review='Pending; no segmentation masks or expert results are invented')
    if (root/'plan.json').exists() and json.loads((root/'plan.json').read_text()) != plan:
        raise ValueError('Resume requires the same predeclared plan')
    write_json(root/'plan.json', plan)
    records = []
    for seed in args.seeds:
        for arm in ('baseline', 'upgraded'):
            output = root/f'{arm}-{seed}'
            if (output/'summary.json').exists():
                summary = json.loads((output/'summary.json').read_text())
                for stage in ('vq', 'diffusion'):
                    if sha256_file(resolve_reference(output, summary[stage])) != summary['checkpoint_hashes'][stage]:
                        raise ValueError('Completed study checkpoint changed; choose a new experiment')
                records.append(summary); continue
            started = time.perf_counter()
            config = deepcopy(plan[arm]); config.update(seed=seed, epochs=args.vq_epochs, steps_per_epoch=args.steps)
            print(f'\nSTUDY {arm} seed {seed}: VQ', flush=True)
            vq = train_vq(data, output/'vq', config, resume=(output/'vq/last.pt').exists())
            if not (output/'latents/manifest.json').exists():
                cache_latents(data, vq, output/'latents', 'cuda')
            config.update(epochs=args.diffusion_epochs, batch_size=4, learning_rate=.0002)
            print(f'STUDY {arm} seed {seed}: diffusion', flush=True)
            diffusion = train_diffusion(output/'latents', output/'diffusion', config,
                                        resume=(output/'diffusion/last.pt').exists())
            if (output/'reconstruction/report.json').exists():
                reconstruction = json.loads((output/'reconstruction/report.json').read_text())
                if reconstruction['checkpoint_sha256'] != sha256_file(vq):
                    raise ValueError('Existing reconstruction belongs to a different selected VQ model')
            else:
                reconstruction = evaluate_reconstruction(data, vq, output/'reconstruction', device='cuda')
            if (output/'tokenizer.json').exists():
                diagnostic = json.loads((output/'tokenizer.json').read_text())
            else:
                model, _ = load_vq(vq, torch.device('cuda'))
                diagnostic = tokenizer_diagnostics(model, TrainingDataset(data, 'train', config['shape']), 'cuda')
                write_json(output/'tokenizer.json', diagnostic)
                del model; torch.cuda.empty_cache()
            sampler = dict(sampler='ddim', sampling_steps=50, guidance_scale=1.5) if arm == 'upgraded' else {}
            if not (output/'generated/manifest.json').exists():
                generate(vq, diffusion, output/'generated', count=args.count, seed=2026, labels=[1, 0, 0], device='cuda', **sampler)
            generation = evaluate_generation(output/'generated', data, output/'generation-report.json',
                                              max_samples=args.count, real_split='validation')
            images = json.loads((output/'generated/manifest.json').read_text())
            if images['diffusion_sha256'] != sha256_file(diffusion):
                raise ValueError('Existing generated samples belong to a different diffusion checkpoint')
            arrays = [np.load(output/'generated'/r['file'], allow_pickle=False)[0] for r in images['rows'][:4]]
            slice_grid(arrays, [f'{arm} seed {seed}; sample {i+1}' for i in range(len(arrays))], output/'generated.png')
            for stage, fields in [('vq', ['reconstruction_l1', 'validation_l1']),
                                  ('diffusion', ['noise_mse', 'validation_noise_mse'])]:
                curve(json.loads((output/stage/'history.json').read_text()), fields, f'{arm} {stage} {seed}', output/f'{stage}-curve.png')
            summary = dict(arm=arm, seed=seed, seconds=time.perf_counter()-started,
                reconstruction=reconstruction['means'], confidence_intervals=reconstruction['confidence_intervals'],
                tokenizer={k: diagnostic[k] for k in ('used_codes', 'dead_code_fraction', 'aggregate_perplexity', 'quantization_mse')},
                generation=generation, vq=reference(vq, output), diffusion=reference(diffusion, output),
                checkpoint_hashes=dict(vq=sha256_file(vq), diffusion=sha256_file(diffusion)),
                parameter_counts=dict(vq=sum(v.numel() for v in read_checkpoint(vq)['model'].values()),
                    diffusion=sum(v.numel() for v in read_checkpoint(diffusion)['model'].values())))
            write_json(output/'summary.json', summary); records.append(summary)
            vq_history = json.loads((output/'vq/history.json').read_text())
            diffusion_history = json.loads((output/'diffusion/history.json').read_text())
            write_json(output/'evidence.json', dict(status=f'Exploratory research-v2 {arm}, seed {seed}; internal validation',
                data=dict(train_count=len(TrainingDataset(data, 'train', config['shape'])),
                          validation_count=len(TrainingDataset(data, 'validation', config['shape']))),
                reconstruction=reconstruction['means'],
                vq=dict(optimizer_steps=sum(r['optimizer_steps'] for r in vq_history)),
                diffusion=dict(optimizer_steps=sum(r['optimizer_steps'] for r in diffusion_history))))
            write_json(root/'progress.json', dict(completed=[(r['arm'], r['seed']) for r in records]))
    comparisons = {}
    for metric in ('mse', 'mae', 'psnr', 'ssim'):
        differences = [next(r for r in records if r['arm']=='upgraded' and r['seed']==seed)['reconstruction'][metric]
                       -next(r for r in records if r['arm']=='baseline' and r['seed']==seed)['reconstruction'][metric]
                       for seed in args.seeds]
        comparisons[metric] = dict(per_seed_upgraded_minus_baseline=differences, mean=float(np.mean(differences)),
                                  standard_deviation=float(np.std(differences, ddof=1)) if len(differences)>1 else None)
    paired = {}
    for seed in args.seeds:
        reports = {arm: json.loads((root/f'{arm}-{seed}/reconstruction/report.json').read_text()) for arm in ('baseline', 'upgraded')}
        baseline = {r['scan_id']: r for r in reports['baseline']['rows']}
        paired[str(seed)] = {metric: bootstrap_interval([r[metric]-baseline[r['scan_id']][metric]
            for r in reports['upgraded']['rows']], seed=seed) for metric in ('mse', 'mae', 'psnr', 'ssim')}
    report = dict(plan=plan, completed_at=datetime.now(timezone.utc).isoformat(), records=records,
        matched_seed_comparison=comparisons, within_seed_paired_exam_intervals=paired,
        environment=dict(python=platform.python_version(), torch=torch.__version__, gpu=torch.cuda.get_device_name(0)),
        independent_test=False, anatomical_review='Not performed',
        interpretation='Validation-selected exploratory study. Neither a favorable pixel metric nor software execution proves realistic anatomy.')
    write_json(root/'study.json', report)
    print(json.dumps(comparisons, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='data/mrnet/MRNet-research-v2')
    p.add_argument('--data', default='data/processed/research-v2')
    p.add_argument('--output', default='runs/RESEARCH-V2')
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    p.add_argument('--vq-epochs', type=int, default=30)
    p.add_argument('--diffusion-epochs', type=int, default=100)
    p.add_argument('--steps', type=int, default=64)
    p.add_argument('--count', type=int, default=8)
    p.add_argument('--resume', action='store_true')
    run(p.parse_args())


if __name__ == '__main__':
    main()
