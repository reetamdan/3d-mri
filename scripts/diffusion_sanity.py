"""Gated one/four-latent overfit and deliberately artificial conditioning sanity.

These are training-reproduction diagnostics, never independent generative results.
The one-volume check must pass before the four-volume run is attempted.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
from typing import Any

if __name__ == '__main__' and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.nn import functional as F
from src.models.diffusion import UNet3D, DDPM
from src.models.autoencoder import decode_latent
from src.training.common import device_for, fresh_output, read_checkpoint
from src.training.experiments import train_diffusion, generate
from src.training.latent_statistics import cache_representation
from src.training.latents import transform_latents
from src.training.representation import load_representation
from src.training.representation import environment
from src.training.evidence import slice_grid
from src.metrics import reconstruction_metrics
from src.utils.integrity import confined_path, sha256_file, generated_volume
from src.utils.logging import write_json
from src.utils.seed import set_seed
from src.utils.cohort import verify_cohort


def conditioning_control(output, device, updates=500):
    set_seed(42)
    model = UNet3D(channels=2, base=8, label_dim=3, channel_mult=[1, 2], allow_null_labels=True).to(device)
    schedule = DDPM(32, 'v').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    y = torch.tensor([[0., 0., 0.], [1., 1., 1.]], device=device).repeat(4, 1)
    clean = (y[:, :1]*1.-.5)[:, :, None, None, None].expand(-1, 2, 8, 8, 8)
    losses = []
    for _ in range(updates):
        t = torch.randint(32, (len(y),), device=device); noise = torch.randn_like(clean)
        mask = torch.rand(len(y), device=device) >= .1
        prediction = model(schedule.add_noise(clean, t, noise), t, y, condition_mask=mask)
        loss = F.mse_loss(prediction, schedule.training_target(clean, t, noise))
        optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
    samples: list[dict[str, Any]] = []
    for seed in (101, 102, 103, 104):
        for label, expected in (([0., 0., 0.], -.5), ([1., 1., 1.], .5)):
            output_tensor = schedule.sample(model, (1, 2, 8, 8, 8), labels=torch.tensor([label], device=device),
                seed=seed, method='ddim', sampling_steps=32, guidance_scale=1., clip_range=(-1., 1.))
            samples.append(dict(seed=seed, condition=label, expected=expected,
                                mean=float(output_tensor.mean()), mse=float((output_tensor-expected).square().mean())))
    result = dict(artificial_fixture=True, updates=updates, initial_loss=float(np.mean(losses[:20])),
                  final_loss=float(np.mean(losses[-20:])), rows=samples,
                  passed=all(abs(r['mean']-r['expected']) < .2 and r['mse'] < .05 for r in samples),
                  interpretation='Known constant-valued numerical targets; no MRI or pathology-fidelity claim')
    write_json(output/'conditioning-control.json', result)
    return result


def subset_cache(source, output, count):
    original = json.loads((source/'manifest.json').read_text())
    rows = [r for r in original['rows'] if r['split'] == 'train'][:count]
    target = fresh_output(output); (target/'train').mkdir()
    manifest = deepcopy(original)
    manifest.update(rows=rows, packed={}, train_ids=[r['scan_id'] for r in rows], smoke_test=True,
                    diagnostic_subset=count, interpretation='Training overfit diagnostic, no held-out MRI evaluation')
    for row in rows:
        path = confined_path(source, row['file'])
        if sha256_file(path) != row['sha256']:
            raise ValueError('Source cache integrity mismatch')
        shutil.copyfile(path, target/row['file'])
    write_json(target/'manifest.json', manifest)
    return manifest


def overfit(data, checkpoint, output, source, count, updates, device, gamma=None, coordinates=False):
    name = f'{count}-'+('baseline' if gamma is None else 'minsnr')
    manifest = subset_cache(source, output/f'cache-{name}', count)
    epochs = min(20, max(1, updates//100))
    if updates % epochs:
        epochs = 1
    config = dict(shape=manifest['shape'], device=str(device), threads=4, seed=42, epochs=epochs,
        batch_size=2, steps_per_epoch=updates//epochs, learning_rate=.0003, gradient_accumulation=1,
        mixed_precision=device.type == 'cuda', smoke_test=True, conditional=False, unet_base=16,
        channel_mult=[1, 2, 4], diffusion_steps=32, prediction_type='v', ema_decay=.995,
        validation_draws=1, min_snr_gamma=gamma, coordinate_channels=coordinates)
    write_json(output/f'config-{name}.json', config)
    diffusion = train_diffusion(output/f'cache-{name}', output/f'model-{name}', config)
    generate(checkpoint, diffusion, output/f'generated-{name}', count=4, seed=2026, device=device,
             sampler='ddim', sampling_steps=32)
    generated = json.loads((output/f'generated-{name}/manifest.json').read_text())
    model, _ = load_representation(checkpoint, device)
    target_latents, target_volumes = [], []
    with torch.no_grad():
        for row in manifest['rows']:
            z = torch.load(confined_path(output/f'cache-{name}', row['file']), weights_only=True)[None].to(device)
            target_latents.append(transform_latents(z, manifest).cpu().numpy()[0])
            target_volumes.append(decode_latent(model, z, project=False)[0, 0].cpu().numpy())
    results: list[dict[str, Any]] = []
    arrays = []
    for row in generated['rows']:
        z = np.load(confined_path(output/f'generated-{name}', row['latent_file']), allow_pickle=False)
        z = transform_latents(torch.from_numpy(z)[None], manifest).numpy()[0]
        distances = [float(np.square(z-reference).mean()) for reference in target_latents]
        nearest = int(np.argmin(distances))
        array = generated_volume(output/f'generated-{name}', row, manifest['shape'])[0]
        metrics = reconstruction_metrics(target_volumes[nearest], array)
        results.append(dict(seed=row['seed'], nearest_training_id=manifest['rows'][nearest]['scan_id'],
                            normalized_latent_mse=min(distances), paired_to_training_reconstruction=metrics))
        arrays.extend([target_volumes[nearest], array])
    slice_grid(arrays, [text for i in range(len(results)) for text in (f'Nearest training reconstruction {i+1}', f'Overfit sample {i+1}')],
               output/f'overfit-{name}.png')
    mean_ssim = float(np.mean([r['paired_to_training_reconstruction']['ssim'] for r in results]))
    mean_mse = float(np.mean([r['normalized_latent_mse'] for r in results]))
    ck = read_checkpoint(diffusion, 'diffusion')
    denoiser = UNet3D(**ck['model_config']).to(device).eval()
    denoiser.load_state_dict(ck['ema_model'] if ck.get('ema_model') else ck['model'])
    schedule = DDPM(32, 'v').to(device)
    probe = []
    with torch.no_grad():
        generator = torch.Generator(device=device).manual_seed(12345)
        for reference in target_latents:
            clean = torch.from_numpy(reference)[None].to(device)
            for t in (0, 8, 16, 31):
                time = torch.tensor([t], device=device)
                noise = torch.randn(clean.shape, generator=generator, device=device)
                estimate = denoiser(schedule.add_noise(clean, time, noise), time)
                probe.append(float(F.mse_loss(estimate, schedule.training_target(clean, time, noise))))
    coverage = len({r['nearest_training_id'] for r in results})
    result = dict(training_cases=count, updates=updates, min_snr_gamma=gamma, coordinate_channels=coordinates,
                  parameters=sum(p.numel() for p in denoiser.parameters()), samples=results, mean_ssim=mean_ssim,
                  fixed_noise_unweighted_denoising_mse=float(np.mean(probe)),
                  mean_normalized_latent_mse=mean_mse, passed=mean_ssim >= .9 and mean_mse <= .1 and coverage >= min(count, 2),
                  coverage=coverage, smoke_test=True,
                  interpretation='Paired to an intentionally memorized training reconstruction; not generation validity')
    write_json(output/f'overfit-{name}.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/processed/mrnet-m4-20261021')
    parser.add_argument('--autoencoder', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--updates', type=int, default=2000)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--coordinates', action='store_true', help='Small matched-budget spatial-memorization ablation')
    parser.add_argument('--latent-cache', help='Reuse an integrity/lineage-checked frozen cache for matched ablations')
    args = parser.parse_args()
    if args.updates < 1:
        parser.error('Use positive diagnostic updates')
    device = device_for(args.device); torch.set_num_threads(4)
    out = fresh_output(args.output)
    write_json(out/'plan.json', dict(autoencoder_sha256=sha256_file(args.autoencoder), updates=args.updates,
        coordinate_channels=args.coordinates, environment=environment(device),
        source_sha256={name: sha256_file(Path(__file__).parents[1]/name) for name in
                       ('src/models/diffusion.py', 'src/training/experiments.py', 'scripts/diffusion_sanity.py')},
        one_volume_gate=dict(mean_ssim=.9, mean_normalized_latent_mse=.1), four_volume_min_coverage=2, seed=42,
        scope='Technical smoke diagnostics only; representation/full-cohort generation gates are not bypassed'))
    print('Artificial conditioning sanity', flush=True)
    conditional = conditioning_control(out, device)
    print('Fit/audit frozen posterior-mean cache', flush=True)
    source = Path(args.latent_cache) if args.latent_cache else out/'full-cache'
    if args.latent_cache:
        cached = json.loads((source/'manifest.json').read_text())
        if (cached['vq_sha256'] != sha256_file(args.autoencoder)
                or cached['dataset_fingerprint'] != verify_cohort(args.data).fingerprint):
            raise ValueError('Reused diagnostic cache has different model/cohort lineage')
    else:
        cache_representation(args.data, args.autoencoder, source, device)
    print('One-volume overfit', flush=True)
    first = overfit(args.data, args.autoencoder, out, source, 1, args.updates, device, coordinates=args.coordinates)
    print('Matched one-volume Min-SNR ablation', flush=True)
    weighted = overfit(args.data, args.autoencoder, out, source, 1, args.updates, device, gamma=5., coordinates=args.coordinates)
    second = None
    if first['passed'] or weighted['passed']:
        print('One-volume passed: four-volume overfit', flush=True)
        second = overfit(args.data, args.autoencoder, out, source, 4, args.updates, device,
                         gamma=None if first['passed'] else 5., coordinates=args.coordinates)
    write_json(out/'sanity.json', dict(conditioning=conditional, one_volume=first, one_volume_minsnr=weighted, four_volume=second,
        status='technical_pass' if conditional['passed'] and (first['passed'] or weighted['passed']) and second and second['passed'] else 'blocked',
        full_cohort_training_authorized=False, note='No representation gate relaxation or MRI pathology claim'))
    print(json.dumps(dict(one_volume=first['passed'], four_volume=second['passed'] if second else None,
                          one_volume_minsnr=weighted['passed'],
                          artificial_conditioning=conditional['passed']), indent=2), flush=True)


if __name__ == '__main__':
    main()
