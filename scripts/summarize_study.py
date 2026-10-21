"""Export sanitized aggregate results and refreshed diagnostics from a completed study."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.vqvae import VQVAE3D
from src.models.diffusion import UNet3D
from src.training.common import read_checkpoint, TrainingDataset, fingerprint
from src.training.evaluation import evaluate_generation
from src.utils.logging import write_json
from src.metrics import downsample_reconstruction_baseline
from src.training.statistics import bootstrap_interval


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', default='runs/RESEARCH-V2')
    p.add_argument('--data', default='data/processed/research-v2')
    p.add_argument('--output', default='docs/results/research-v2.md')
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    if args.threads < 1:
        p.error('threads must be positive')
    torch.set_num_threads(args.threads)
    root = Path(args.study)
    study = json.loads((root/'study.json').read_text())
    if fingerprint(args.data) != study['plan']['dataset_fingerprint']:
        raise ValueError('Summary data do not match the frozen study cohort')
    validation = TrainingDataset(args.data, 'validation', [128, 128, 32])
    interpolation_path = root/'interpolation-reference.json'
    if interpolation_path.exists():
        interpolation = json.loads(interpolation_path.read_text())
    else:
        reference_rows = []
        for i, row in enumerate(validation.rows):
            _, metrics = downsample_reconstruction_baseline(validation[i][0][0].numpy(), 2.)
            reference_rows.append(dict(scan_id=row['scan_id'], **metrics))
        interpolation = dict(rows=reference_rows, means={k: float(np.mean([r[k] for r in reference_rows]))
            for k in ('mse', 'mae', 'psnr', 'ssim')},
            protocol='Half-resolution interpolation on the identical 60 validation tensors; different information budget from VQ')
        write_json(interpolation_path, interpolation)
    rows = []
    for seed in study['plan']['seeds']:
        for arm in ('baseline', 'upgraded'):
            folder = root/f'{arm}-{seed}'
            summary = json.loads((folder/'summary.json').read_text())
            generated = evaluate_generation(folder/'generated', args.data, folder/'generation-report.json',
                                            max_samples=study['plan']['generated_count'], real_split='validation')
            summary['generation'] = generated
            for stage, cls in (('vq', VQVAE3D), ('diffusion', UNet3D)):
                info = json.loads((folder/stage/'run.json').read_text())
                ck = read_checkpoint(folder/stage/info['checkpoint_file'])
                summary['parameter_counts'][stage] = sum(p.numel() for p in cls(**ck['model_config']).parameters())
            write_json(folder/'summary.json', summary)
            reconstruction = json.loads((folder/'reconstruction/report.json').read_text())
            histories = {s: json.loads((folder/s/'history.json').read_text()) for s in ('vq', 'diffusion')}
            write_json(folder/'evidence.json', dict(status=f'Exploratory {arm}, seed {seed}; validation-selected',
                data=dict(train_count=len(TrainingDataset(args.data, 'train', [128, 128, 32])),
                          validation_count=len(TrainingDataset(args.data, 'validation', [128, 128, 32]))),
                reconstruction=reconstruction['means'],
                **{s: dict(optimizer_steps=sum(r['optimizer_steps'] for r in h)) for s, h in histories.items()}))
            rows.append(summary)
    lines = ['# Research v2 — controlled local comparison', '',
        'Measured on the same 256-train / 60-validation sagittal MRNet cohort at 128×128×32.',
        'Seeds: '+', '.join(map(str, study['plan']['seeds']))+'.', '',
        '**Exploratory internal validation, not an independent patient-disjoint test.**', '',
        'Both arms received the same declared optimizer-update and batch-size budgets. The upgraded arm combines',
        'a larger EMA tokenizer, multiscale denoiser, weight EMA, v-prediction, mixed precision and label dropout.',
        'It uses 50-step DDIM with guidance 1.5 for sampling; the control uses full DDPM.',
        'This comparison cannot attribute gains to one component, and training noise losses across epsilon/v targets are not directly comparable.', '',
        '## Reconstruction: mean ± sample standard deviation across training seeds', '',
        '| Arm | MSE ↓ | MAE ↓ | PSNR ↑ | SSIM ↑ |', '|---|---:|---:|---:|---:|']
    for arm in ('baseline', 'upgraded'):
        selected = [r for r in rows if r['arm']==arm]
        values = []
        for metric in ('mse', 'mae', 'psnr', 'ssim'):
            a = np.array([r['reconstruction'][metric] for r in selected])
            values.append(f'{a.mean():.5f} ± {a.std(ddof=1):.5f}' if len(a)>1 else f'{a.mean():.5f}')
        lines.append('| '+arm+' | '+' | '.join(values)+' |')
    lines += ['| Interpolation reference | '+' | '.join(f'{interpolation["means"][k]:.5f}' for k in ('mse', 'mae', 'psnr', 'ssim'))+' |',
              '', 'The interpolation reference is measured on these same 60 tensors. Its information budget differs from the learned codebook.']
    lines += ['', '## Every seed', '', '| Arm | Seed | PSNR | SSIM | Used codes | Dead-code fraction |', '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["arm"]} | {r["seed"]} | {r["reconstruction"]["psnr"]:.5f} | '
                     f'{r["reconstruction"]["ssim"]:.5f} | {r["tokenizer"]["used_codes"]} | {r["tokenizer"]["dead_code_fraction"]:.5f} |')
    lines += ['', 'Within-seed paired exam-bootstrap intervals are recorded in `study.json`;',
        'patient IDs are unavailable, so these are exam-level intervals and do not account for unidentified repeat patients.',
        'Three seeds provide limited evidence about training variability; intervals do not convert validation into a test set.', '',
        '## Model and sampling diagnostics', '']
    comparisons = {}
    references = {r['scan_id']: r for r in interpolation['rows']}
    for seed in study['plan']['seeds']:
        learned = json.loads((root/f'upgraded-{seed}/reconstruction/report.json').read_text())
        comparisons[str(seed)] = {k: bootstrap_interval([r[k]-references[r['scan_id']][k] for r in learned['rows']], seed=seed)
                                  for k in ('mse', 'mae', 'psnr', 'ssim')}
    write_json(root/'learned-vs-interpolation.json', comparisons)
    study['records'] = rows
    study['interpolation_reference'] = interpolation['means']
    write_json(root/'study.json', study)
    for arm in ('baseline', 'upgraded'):
        first = next(r for r in rows if r['arm']==arm)
        lines += [f'- {arm}: {first["parameter_counts"]["vq"]:,} VQ parameters and {first["parameter_counts"]["diffusion"]:,} denoiser parameters.',
            f'  Memorization search covers all {first["generation"]["memorization_reference_count"]} recorded training exams, independently of the diversity sample cap.']
    lines += ['', 'All sample-level quantization distances, clipping trajectories, nearest raw/pooled-feature neighbours,',
        'class-matched validation reference counts and texture descriptors are in each generation report.',
        'The 4×4×4 pooled descriptor is not a validated medical perceptual embedding.', '',
        '## Limits of the result', '',
        '- Anatomical/pathology fidelity and clinical utility are not established by these metrics.',
        '- No independent test cohort or qualified expert-review results were available for this study.',
        '- Synthetic-label fidelity and downstream augmentation benefit remain unproven.',
        '- Feature FID/KID tooling requires a justified external extractor and sufficient samples; no medical FID is invented.',
        '- The mirror license is unspecified; upstream data-use terms still apply.', '',
        '## Reproduce', '', '```bash',
        'python -m pip install -e ".[dev,metrics]"',
        'python scripts/download_mrnet_subset.py --output data/mrnet/MRNet-research-v2 --train 256 --validation 60 --planes sagittal --workers 8 --max-mib 1024',
        'mri3d-study', 'python scripts/sampling_ablation.py', '```', '',
        'Data, checkpoint and sample artifacts remain local. Git contains source, configurations and aggregate results.', '']
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text('\n'.join(lines), encoding='utf-8')
    print('Public aggregate results written:', args.output)


if __name__ == '__main__':
    main()
