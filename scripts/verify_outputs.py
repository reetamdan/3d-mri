"""Audit existing MRNet outputs and their internal-validation utility without retraining the generator.

Writes a fresh, separate audit tree. Uses independent metric calculations, all
available train/validation cases, same-runtime sampling repeats, and a matched
real-data resampling control for the saved synthetic-label distribution.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

if __name__ == '__main__' and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.ndimage import zoom, map_coordinates
from skimage.metrics import structural_similarity
import torch
from torch.nn import functional as F

from src.data.indexing import token
from src.data.mrnet import labels_for
from src.training.common import TrainingDataset, device_for, fresh_output, read_checkpoint
from src.training.diagnostics import foreground_metrics, tokenizer_diagnostics, artifact_descriptors
from src.training.downstream import compare_augmentation, MRIClassifier3D
from src.training.evaluation import evaluate_generation
from src.training.evidence import slice_grid
from src.training.experiments import load_vq, generate
from src.utils.artifacts import resolve_reference
from src.utils.cohort import verify_cohort
from src.utils.integrity import sha256_file, confined_path, generated_volume
from src.utils.logging import write_json
from src.utils.seed import set_seed

METRICS = ('mse', 'mae', 'psnr', 'ssim')
TASKS = ('abnormal', 'acl', 'meniscus')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def independent_metrics(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 3 or min(a.shape) < 3 or not np.isfinite([a, b]).all():
        raise ValueError('Expected matching finite 3D metric inputs')
    error = a-b
    mse = float(np.einsum('ijk,ijk->', error, error)/error.size)
    window = min(7, min(a.shape)); window -= 1-window % 2
    return dict(mse=mse, mae=float(np.abs(error).sum()/error.size),
                psnr=float(10*np.log10(4/mse)) if mse else None,
                ssim=float(structural_similarity(a, b, data_range=2., win_size=window)))


def independent_classification(labels, probabilities):
    y, p = np.asarray(labels), np.asarray(probabilities, dtype=np.float64)
    if (y.ndim != 2 or y.shape[1] != 3 or not len(y) or p.shape != y.shape
            or not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))):
        raise ValueError('Expected N by 3 binary labels and probabilities')
    result = {}
    for i, task in enumerate(TASKS):
        positive, negative = p[y[:, i] == 1, i], p[y[:, i] == 0, i]
        pairs = positive[:, None]-negative[None, :]
        auc = float(((pairs > 0)+.5*(pairs == 0)).mean()) if pairs.size else None
        result[task] = dict(auc=auc, accuracy=float(((p[:, i] >= .5) == y[:, i]).mean()),
                            brier=float(np.square(p[:, i]-y[:, i]).mean()), threshold=.5,
                            positive=len(positive), negative=len(negative),
                            sensitivity=float((positive >= .5).mean()) if len(positive) else None,
                            specificity=float((negative < .5).mean()) if len(negative) else None)
    return result


def paired_interval(values, seed=42, draws=10000):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Paired differences must be a finite nonempty vector')
    rng = np.random.default_rng(seed)
    estimates = values[rng.integers(len(values), size=(draws, len(values)))].mean(1)
    return dict(mean=float(values.mean()), lower=float(np.quantile(estimates, .025)),
                upper=float(np.quantile(estimates, .975)), observations=len(values),
                unit='exam', seed=seed, resamples=draws, confidence=.95)


def close(actual, expected, tolerance, label):
    difference = float(np.max(np.abs(np.asarray(actual, dtype=float)-np.asarray(expected, dtype=float))))
    if not np.isfinite(difference) or difference > tolerance:
        raise ValueError(f'{label}: difference {difference} exceeds {tolerance}')
    return difference


def means(rows, keys=METRICS):
    return {k: float(np.mean([row[k] for row in rows])) for k in keys}


def interpolation(array):
    low = zoom(np.asarray(array, dtype=np.float32), .5, order=1)
    grid = np.meshgrid(*(np.linspace(0, old-1, new) for old, new in zip(low.shape, array.shape)), indexing='ij')
    return map_coordinates(low, grid, order=1, mode='nearest')


@torch.no_grad()
def verify_cohort_and_cache(run, data, source, model, checkpoint, device):
    manifest = read_json(source/'source_manifest.json')
    source_by_scan = {}
    for row in manifest['files']:
        path = confined_path(source, row['file'])
        if sha256_file(path) != row['sha256']:
            raise ValueError('Source array checksum mismatch')
        split, plane, filename = Path(row['file']).parts
        source_by_scan[token(f'MRNet/{Path(filename).stem}/{plane}')] = (path, split, Path(filename).stem, row['sha256'])
    labels = {split: labels_for(source, split) for split in ('train', 'valid')}
    if set(labels['train']) & set(labels['valid']):
        raise ValueError('Official exam splits overlap')
    for split, count_key in (('train', 'train_count'), ('valid', 'validation_count')):
        if len(labels[split]) != manifest['selection'][count_key]:
            raise ValueError('Source selection count mismatch')
        for index, task in enumerate(TASKS):
            if sum(y[index] for y in labels[split].values()) != manifest['class_counts'][split][task]:
                raise ValueError('Source class support mismatch')
    snapshot = verify_cohort(data)
    if snapshot.fingerprint != checkpoint['dataset_fingerprint']:
        raise ValueError('Checkpoint dataset fingerprint mismatch')
    cache = read_json(run/'latents/manifest.json')
    cached = {r['scan_id']: r for r in cache['rows']}
    raw_errors, latent_errors, agreements, sums, squares, count = [], [], [], 0., 0., 0
    drift_probe = None
    split_hashes: dict[str, set[str]] = {'train': set(), 'validation': set()}
    for row in snapshot.rows:
        path, official, exam, source_hash = source_by_scan[row['scan_id']]
        if (row['split'] != {'train': 'train', 'valid': 'validation'}[official]
                or json.loads(row['labels']) != labels[official][exam] or row['source_sha256'] != source_hash):
            raise ValueError('Processed labels/split/source identity mismatch')
        x = np.load(confined_path(data, row['output_file']), allow_pickle=False)
        if x.shape != (1, *checkpoint['shape']) or not np.isfinite(x).all() or x.min() < -1.0001 or x.max() > 1.0001:
            raise ValueError('Invalid processed tensor')
        raw = np.load(path, allow_pickle=False)
        if raw.dtype != np.uint8 or raw.ndim != 3 or raw.shape[1:] != (256, 256):
            raise ValueError('Unexpected MRNet source geometry/type')
        resized = F.interpolate(torch.from_numpy(raw.transpose(1, 2, 0).copy().astype(np.float32))[None, None],
                                size=checkpoint['shape'], mode='trilinear', align_corners=False)[0, 0].numpy()
        rebuilt = ((resized.astype(np.float64)-resized.min())/(resized.max()-resized.min())*2-1).astype(np.float32)
        raw_errors.append(close(x[0], rebuilt, 2e-6, 'Raw-to-processed replay'))
        record = cached[row['scan_id']]
        latent_path = confined_path(run/'latents', record['file'])
        if sha256_file(latent_path) != record['sha256']:
            raise ValueError('Stored latent checksum mismatch')
        latent = torch.load(latent_path, weights_only=True).numpy()
        replay = model.encode(torch.from_numpy(x)[None].to(device))[0][0].cpu().numpy()
        replay_indices = model.quantizer.last_indices.clone()
        latent_batch = torch.from_numpy(latent)[None].to(device)
        projected = model.quantizer(latent_batch)[0][0].cpu().numpy()
        close(projected, latent, 1e-6, 'Cached latent membership in selected codebook')
        stored_indices = model.quantizer.last_indices.clone()
        agreement = float((replay_indices == stored_indices).float().mean())
        agreements.append(agreement)
        error = float(np.square(replay.astype(float)-latent).mean())
        latent_errors.append(error)
        if agreement < 1 and drift_probe is None:
            cpu_model = deepcopy(model).cpu()
            cpu_model.encode(torch.from_numpy(x)[None])
            drift_probe = dict(scan_id=row['scan_id'], default_device_code_agreement=agreement,
                               cpu_code_agreement=float((cpu_model.quantizer.last_indices == stored_indices.cpu()).float().mean()))
            if device.type == 'cuda':
                previous = torch.backends.cudnn.allow_tf32
                try:
                    torch.backends.cudnn.allow_tf32 = False
                    model.encode(torch.from_numpy(x)[None].to(device))
                    drift_probe['cuda_tf32_disabled_code_agreement'] = float((model.quantizer.last_indices == stored_indices).float().mean())
                finally:
                    torch.backends.cudnn.allow_tf32 = previous
            del cpu_model
        split_hashes[row['split']].add(row['output_sha256'])
        if row['split'] == 'train':
            z = latent.astype(np.float64)
            sums += z.sum(); squares += np.square(z).sum(); count += z.size
    if split_hashes['train'] & split_hashes['validation']:
        raise ValueError('Processed tensors overlap across splits')
    close(sums/count, cache['mean'], 1e-9, 'Latent normalization mean')
    close(np.sqrt(squares/count-(sums/count)**2), cache['std'], 1e-9, 'Latent normalization std')
    return dict(source_arrays_checked=len(manifest['files']), processed_tensors_checked=len(snapshot.rows),
                latent_tensors_checked=len(latent_errors), dataset_fingerprint=snapshot.fingerprint,
                raw_replay_max_abs_error=max(raw_errors), latent_replay_max_mse=max(latent_errors),
                latent_replay_mean_mse=float(np.mean(latent_errors)),
                latent_code_agreement_mean=float(np.mean(agreements)), latent_code_agreement_min=min(agreements),
                cached_latents_in_selected_codebook=True, precision_drift_probe=drift_probe,
                historical_replay_scope='Cross-GPU/PyTorch code assignments measured, not assumed byte-identical',
                split_counts=dict(Counter(r['split'] for r in snapshot.rows)),
                patient_identifiers_available=all(bool(r.get('patient_id')) for r in snapshot.rows),
                train_label_positives=np.sum(list(labels['train'].values()), axis=0).tolist(),
                validation_label_positives=np.sum(list(labels['valid'].values()), axis=0).tolist())


@torch.no_grad()
def reconstruction_audit(run, output, model, train, validation, device):
    saved = read_json(run/'reconstruction/report.json')
    baseline = read_json(run/'baseline-report.json')
    saved_by_id = {r['scan_id']: r for r in saved['rows']}
    base_by_id = {r['scan_id']: r for r in baseline['rows']}
    training_mean = np.mean(np.stack([train[i][0][0].numpy() for i in range(len(train))]), axis=0)
    rows, training_rows, saved_metric_errors, replay_errors = [], [], [], []
    for split, dataset in (('train', train), ('validation', validation)):
        for i, row in enumerate(dataset.rows):
            x = dataset[i][0][0].numpy()
            restored = model(torch.from_numpy(x)[None, None].to(device))[0][0, 0].cpu().numpy()
            measured = independent_metrics(x, restored)
            if split == 'train':
                training_rows.append(measured)
                continue
            original = generated_volume(run/'reconstruction', saved_by_id[row['scan_id']], saved['shape'])[0]
            replay_errors.append(float(np.square(restored.astype(float)-original).mean()))
            verified = independent_metrics(x, original)
            reference = interpolation(x)
            reference_metrics = independent_metrics(x, reference)
            for key in METRICS:
                saved_metric_errors.append(close(verified[key], saved_by_id[row['scan_id']][key], 1e-9, f'Saved {key}'))
                close(reference_metrics[key], base_by_id[row['scan_id']][key], 1e-9, f'Interpolation {key}')
            rows.append(dict(scan_id=row['scan_id'], learned=verified, current_replay=measured, interpolation=reference_metrics,
                             train_mean=independent_metrics(x, training_mean),
                             foreground_learned=foreground_metrics(x, original),
                             foreground_interpolation=foreground_metrics(x, reference)))
            if i < 3:
                slice_grid([x, original, reference], ['Validation input', 'VQ reconstruction', 'Interpolation reference'],
                           output/f'reconstruction-{i+1}.png')
    learned, reference = [r['learned'] for r in rows], [r['interpolation'] for r in rows]
    for key in METRICS:
        close(means(learned)[key], saved['means'][key], 1e-9, f'Aggregate saved {key}')
        close(means(reference)[key], baseline['means'][key], 1e-9, f'Aggregate baseline {key}')
    result = dict(validation_count=len(rows), train_count=len(train), learned=means(learned),
                  current_runtime_replay=means([r['current_replay'] for r in rows]),
                  interpolation=means(reference), train_mean_baseline=means([r['train_mean'] for r in rows]),
                  training_reconstruction=means(training_rows), saved_metric_max_error=max(saved_metric_errors),
                  historical_replay_max_mse=max(replay_errors), rows=rows,
                  paired_learned_minus_interpolation={k: paired_interval([r['learned'][k]-r['interpolation'][k] for r in rows])
                                                      for k in METRICS},
                  learned_wins={k: sum(r['learned'][k] < r['interpolation'][k] if k in ('mse', 'mae')
                                      else r['learned'][k] > r['interpolation'][k] for r in rows) for k in METRICS},
                  uncertainty='Paired exam bootstrap on model-selection validation; not independent patients/test performance')
    write_json(output/'reconstruction-audit.json', result)
    return result


@torch.no_grad()
def generation_audit(run, output, data, model, vq, diffusion, device, train):
    manifest = read_json(run/'generated/manifest.json')
    if manifest['vq_sha256'] != sha256_file(vq) or manifest['diffusion_sha256'] != sha256_file(diffusion):
        raise ValueError('Saved generation checkpoint lineage mismatch')
    decoded_errors, arrays = [], []
    for row in manifest['rows']:
        volume = generated_volume(run/'generated', row, manifest['shape'])[0]
        latent_path = confined_path(run/'generated', row['latent_file'])
        if sha256_file(latent_path) != row['latent_sha256']:
            raise ValueError('Generated latent checksum mismatch')
        latent = np.load(latent_path, allow_pickle=False)
        decoded = model.decode(torch.from_numpy(latent)[None].to(device), quantize=True)[0, 0].cpu().numpy()
        error = float(np.square(decoded.astype(float)-volume).mean())
        if not np.isfinite(decoded).all() or decoded.shape != volume.shape:
            raise ValueError('Latent-to-image replay produced invalid output')
        decoded_errors.append(error); arrays.append(volume)
    fresh = evaluate_generation(run/'generated', data, output/'generation-diagnostics.json', max_samples=len(arrays))
    for i in range(0, len(arrays), 4):
        slice_grid(arrays[i:i+4], [f'Saved generated sample {j+1}' for j in range(i, min(i+4, len(arrays)))],
                   output/f'generated-{i//4+1}.png')
    settings = manifest.get('sampling', {})
    options = dict(sampler=settings.get('method', 'ddpm'), sampling_steps=settings.get('sampling_steps'),
                   guidance_scale=settings.get('guidance_scale', 1.), eta=settings.get('eta', 0.),
                   weights=settings.get('weights', 'auto'))
    labels = manifest['rows'][0].get('conditioning_labels')
    count, seed = len(arrays), manifest['rows'][0]['seed']
    for name in ('repeat-a', 'repeat-b'):
        generate(vq, diffusion, output/name, count=count, seed=seed, labels=labels, device=device, **options)
    a, b = read_json(output/'repeat-a/manifest.json'), read_json(output/'repeat-b/manifest.json')
    identical = all(x['sha256'] == y['sha256'] and x['latent_sha256'] == y['latent_sha256'] for x, y in zip(a['rows'], b['rows']))
    if not identical:
        raise ValueError('Same-runtime same-seed generation is not byte reproducible')
    if len({r['latent_sha256'] for r in a['rows']}) != count:
        raise ValueError('Different seeds produced identical latent files')
    alternative = [1, 1, 0]
    support = sum(json.loads(r['labels']) == alternative for r in train.rows)
    condition_effect = None
    if labels is not None and support:
        generate(vq, diffusion, output/'alternate-condition', count=min(3, count), seed=seed,
                 labels=alternative, device=device, **options)
        alternate = read_json(output/'alternate-condition/manifest.json')
        condition_effect = dict(requested=alternative, training_support=support,
            same_noise_image_mse=[float(np.square(generated_volume(output/'repeat-a', x, a['shape'])
                                                  -generated_volume(output/'alternate-condition', y, alternate['shape'])).mean())
                                  for x, y in zip(a['rows'], alternate['rows'])],
            pathology_verified=False)
    train_descriptors = [artifact_descriptors(train[i][0][0].numpy()) for i in range(len(train))]
    result = dict(count=count, latent_decode_max_mse=max(decoded_errors),
                  same_runtime_seed_repeat_byte_identical=identical, different_seed_latents_unique=True,
                  fresh_generation_diagnostics=fresh,
                  training_gradient_energy_mean=float(np.mean([d['gradient_energy'] for d in train_descriptors])),
                  generated_gradient_energy_mean=float(np.mean([artifact_descriptors(x)['gradient_energy'] for x in arrays])),
                  same_noise_condition_change=condition_effect,
                  new_sampling_mean_clipped_fraction=float(np.mean([d['clipped_fraction'] for r in a['rows'] for d in r['sampling_diagnostics']])),
                  new_sampling_mean_quantization_mse=float(np.mean([r['quantization_mse'] for r in a['rows']])),
                  original_byte_repeat_matches=sum(x['sha256'] == y['sha256'] for x, y in zip(a['rows'], manifest['rows'])),
                  reproducibility_scope='Current device/runtime repeated twice; historical artifacts used different GPU/PyTorch')
    write_json(output/'generation-audit.json', result)
    return result


@torch.no_grad()
def classifier_predictions(model, dataset, device):
    model.eval()
    return np.stack([model(x[None].to(device)).sigmoid()[0].cpu().numpy() for x, _ in dataset])


def utility_audit(run, output, data, train, valid, device, steps, seeds):
    report = compare_augmentation(data, run/'generated', output/'downstream', shape=train.shape, limit=len(train),
                                  steps=steps, seeds=seeds, device=device, bootstrap_samples=1000)
    truth = np.stack([y.numpy() for _, y in valid])
    train_labels = np.stack([y.numpy() for _, y in train])
    source = read_json(run/'generated/manifest.json')
    conditions = np.array([r['conditioning_labels'] for r in source['rows']])
    predictions: dict[str, dict[int, Any]] = {arm: {} for arm in ('real_only', 'real_plus_synthetic', 'label_matched_real')}
    metrics = []
    for record in report['results']:
        model = MRIClassifier3D().to(device)
        path = output/'downstream'/f'{record["arm"]}-{record["seed"]}.pt'
        model.load_state_dict(read_checkpoint(path)['model'])
        probabilities = classifier_predictions(model, valid, device)
        measured = independent_classification(truth, probabilities)
        for task in TASKS:
            for key in ('auc', 'accuracy', 'brier', 'sensitivity', 'specificity'):
                close(measured[task][key], record['metrics'][task][key], 2e-7, f'Downstream {task}/{key}')
        predictions[record['arm']][record['seed']] = probabilities
        metrics.append(dict(arm=record['arm'], seed=record['seed'], metrics=measured,
                            training_metrics=independent_classification(train_labels, classifier_predictions(model, train, device))))
    for seed in seeds:
        set_seed(seed)
        model = MRIClassifier3D().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        rng, control_rng = np.random.default_rng(seed), np.random.default_rng(seed+20000)
        for step in range(steps):
            if step % 2:
                condition = conditions[int(rng.integers(len(conditions)))]
                matching = np.flatnonzero(np.all(train_labels == condition, axis=1))
                if not len(matching):
                    raise ValueError('No real support for a generated condition')
                index = int(control_rng.choice(matching))
            else:
                index = int(rng.integers(len(train)))
            x, y = train[index]
            loss = F.binary_cross_entropy_with_logits(model(x[None].to(device)), y[None].to(device))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        probabilities = classifier_predictions(model, valid, device)
        predictions['label_matched_real'][seed] = probabilities
        torch.save(dict(model=model.state_dict(), seed=seed, arm='label_matched_real'),
                   output/'downstream'/f'label_matched_real-{seed}.pt')
        metrics.append(dict(arm='label_matched_real', seed=seed,
                            metrics=independent_classification(truth, probabilities),
                            training_metrics=independent_classification(train_labels, classifier_predictions(model, train, device))))
    np.savez_compressed(output/'downstream/predictions.npz', truth=truth,
                        **{f'{arm}_{seed}': p for arm, entries in predictions.items() for seed, p in entries.items()})
    comparison: dict[str, dict[str, Any]] = {}
    draws = np.random.default_rng(42).integers(len(truth), size=(1000, len(truth)))
    for control in ('real_only', 'label_matched_real'):
        comparison[control] = {}
        for task in TASKS:
            deltas = [independent_classification(truth, predictions['real_plus_synthetic'][seed])[task]['auc']
                      -independent_classification(truth, predictions[control][seed])[task]['auc'] for seed in seeds]
            estimates = []
            for index in draws:
                values = []
                for seed in seeds:
                    first = independent_classification(truth[index], predictions['real_plus_synthetic'][seed][index])[task]['auc']
                    second = independent_classification(truth[index], predictions[control][seed][index])[task]['auc']
                    if first is not None and second is not None:
                        values.append(first-second)
                if values:
                    estimates.append(float(np.mean(values)))
            comparison[control][task] = dict(auc_augmented_minus_control=float(np.mean(deltas)), per_seed=deltas,
                seed_sd=float(np.std(deltas, ddof=1)) if len(deltas) > 1 else None,
                paired_exam_interval=[float(np.quantile(estimates, .025)), float(np.quantile(estimates, .975))],
                bootstrap_defined=len(estimates), uncertainty_unit='validation exam; fixed fitted training seeds')
    result = dict(steps_per_arm=steps, seeds=seeds, metrics=metrics, auc_comparisons=comparison,
                  prevalence_only_baseline=independent_classification(truth, np.tile(train_labels.mean(0), (len(valid), 1))),
                  real_train_positive_rates=train_labels.mean(0).tolist(),
                  synthetic_positive_rates=conditions.mean(0).tolist(),
                  augmented_draw_positive_rates=(.5*train_labels.mean(0)+.5*conditions.mean(0)).tolist(),
                  predictions_sha256=sha256_file(output/'downstream/predictions.npz'),
                  independent_test=False, patient_linkage_available=False,
                  control='Odd updates resample real cases matching synthetic conditions; even real updates match augmented arm',
                  conclusion='Exploratory validation comparison; requested synthetic labels are not pathology annotations')
    write_json(output/'utility-audit.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='runs/M4-MRNET-20261021')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument('--skip-utility', action='store_true')
    args = parser.parse_args()
    if args.steps < 1 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error('Use positive classifier updates and distinct seeds')
    torch.set_num_threads(4)
    run = Path(args.run).resolve()
    status = read_json(run/'status.json')
    data, source = (resolve_reference(run, status[k]) for k in ('data_root', 'source_root'))
    vq, diffusion = (resolve_reference(run, status[k]) for k in ('vq_checkpoint', 'diffusion_checkpoint'))
    device = device_for(args.device)
    output = fresh_output(args.output)
    started = time.perf_counter()
    model, ck = load_vq(vq, device)
    dc = read_checkpoint(diffusion, 'diffusion')
    pair = dict(vq=sha256_file(vq), diffusion=sha256_file(diffusion))
    if (dc['latent_manifest']['vq_sha256'] != pair['vq'] or ck['smoke_test'] or dc['smoke_test']
            or sha256_file(source/'source_manifest.json') != status['source_manifest_sha256']):
        raise ValueError('Research model/source lineage mismatch')
    cache = read_json(run/'latents/manifest.json')
    saved = read_json(run/'reconstruction/report.json')
    if (cache['vq_sha256'] != pair['vq'] or cache['dataset_fingerprint'] != ck['dataset_fingerprint']
            or saved['checkpoint_sha256'] != pair['vq'] or saved['dataset_fingerprint'] != ck['dataset_fingerprint']
            or status['dataset_fingerprint'] != ck['dataset_fingerprint']):
        raise ValueError('Cache/reconstruction/status lineage mismatch')
    for stage in ('vq', 'diffusion'):
        if read_json(run/stage/'run.json')['checkpoint_sha256'] != pair[stage]:
            raise ValueError('Selected checkpoint hash mismatch')
    write_json(output/'plan.json', dict(created_at=datetime.now(timezone.utc).isoformat(), run=str(run),
               verifier_sha256=sha256_file(Path(__file__)),
               dataset_fingerprint=ck['dataset_fingerprint'], checkpoint_hashes=pair,
               classifier_steps=args.steps, seeds=args.seeds, utility=not args.skip_utility,
               scope='Existing cohort/output verification and exploratory internal validation; no new cohort',
               environment=dict(python=platform.python_version(), torch=str(torch.__version__),
                                device=str(device), gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else None)))
    train, valid = (TrainingDataset(data, split, ck['shape'], require_labels=True) for split in ('train', 'validation'))
    print('Verifying all source/processed/cache files and replaying preprocessing/encoding', flush=True)
    cohort = verify_cohort_and_cache(run, data, source, model, ck, device)
    write_json(output/'cohort-audit.json', cohort)
    print('Independently recomputing all reconstruction metrics and paired baseline intervals', flush=True)
    reconstruction = reconstruction_audit(run, output, model, train, valid, device)
    tokenizer = tokenizer_diagnostics(model, train, device)
    write_json(output/'tokenizer-audit.json', tokenizer)
    print('Verifying saved latents/decodes, full-cohort neighbors, repeated seeds and condition changes', flush=True)
    generation = generation_audit(run, output, data, model, vq, diffusion, device, train)
    utility = None
    if not args.skip_utility:
        print(f'Running three classifier arms, {args.steps} updates each, seeds {args.seeds}', flush=True)
        utility = utility_audit(run, output, data, train, valid, device, args.steps, args.seeds)
    histories = {}
    for stage, checkpoint, key in (('vq', ck, 'validation_l1'), ('diffusion', dc, 'validation_noise_mse')):
        history = read_json(run/stage/'history.json')
        selected = next(r for r in history if r['epoch'] == checkpoint['epoch'])
        close(selected[key], min(r[key] for r in history), 1e-12, f'{stage} checkpoint selection')
        histories[stage] = dict(selected_epoch=checkpoint['epoch'], completed_epochs=history[-1]['epoch'],
                                optimizer_updates=sum(r['optimizer_steps'] for r in history), selection_metric=key)
    result = dict(status='completed', seconds=time.perf_counter()-started, checkpoint_hashes=pair, cohort=cohort,
                  reconstruction={k: v for k, v in reconstruction.items() if k != 'rows'},
                  tokenizer={k: v for k, v in tokenizer.items() if k not in ('rows', 'counts')},
                  generation=generation, utility=utility, training=histories,
                  research_v2_local_artifacts_available=(run.parent/'RESEARCH-V2/study.json').exists())
    write_json(output/'audit.json', result)
    print(json.dumps(dict(status='completed', seconds=result['seconds'], output=str(output)), indent=2), flush=True)


if __name__ == '__main__':
    main()
