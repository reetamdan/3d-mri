"""Conventional/resampling/synthetic controls with explicit development thresholds."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from src.utils.cohort import verify_cohort
from src.utils.integrity import sha256_file, generated_volume
from src.utils.logging import write_json
from src.utils.seed import set_seed
from .common import TrainingDataset, fresh_output, device_for
from .downstream import MRIClassifier3D, classification_metrics
from .protocol import audit_overlap, evaluation_context

TASKS = ('abnormal', 'acl', 'meniscus')


def average_precision(labels, scores) -> float | None:
    y, p = np.asarray(labels), np.asarray(scores)
    positives = int(y.sum())
    if not positives:
        return None
    order = np.argsort(-p, kind='stable'); y, p = y[order], p[order]
    ends = np.flatnonzero(np.r_[p[1:] != p[:-1], True])
    true_positives = np.cumsum(y)[ends]
    precision = true_positives/(ends+1)
    return float(np.sum(np.diff(np.r_[0, true_positives])/positives*precision))


def select_thresholds(labels, probabilities) -> list[float]:
    """Development-only Youden maximization; ties choose the higher threshold."""
    classification_metrics(labels, probabilities)
    y, p = np.asarray(labels), np.asarray(probabilities, dtype=float)
    thresholds = []
    for i in range(3):
        positive, negative = y[:, i] == 1, y[:, i] == 0
        if not positive.any() or not negative.any():
            thresholds.append(.5); continue
        candidates = np.r_[np.nextafter(p[:, i].max(), np.inf), np.unique(p[:, i])]
        scores = [(float((p[positive, i] >= t).mean()+(p[negative, i] < t).mean()-1), float(t)) for t in candidates]
        thresholds.append(max(scores)[1])
    return thresholds


def downstream_metrics(labels, probabilities, thresholds):
    base = classification_metrics(labels, probabilities)
    y, p = np.asarray(labels), np.asarray(probabilities, dtype=float)
    if len(thresholds) != 3 or not np.isfinite(thresholds).all():
        raise ValueError('One finite frozen threshold is required per task')
    for i, task in enumerate(TASKS):
        positive, negative = y[:, i] == 1, y[:, i] == 0
        sensitivity = float((p[positive, i] >= thresholds[i]).mean()) if positive.any() else None
        specificity = float((p[negative, i] < thresholds[i]).mean()) if negative.any() else None
        base[task].update(auprc=average_precision(y[:, i], p[:, i]), sensitivity=sensitivity, specificity=specificity,
            balanced_accuracy=(sensitivity+specificity)/2 if sensitivity is not None and specificity is not None else None,
            accuracy=float(((p[:, i] >= thresholds[i]) == y[:, i]).mean()), threshold=float(thresholds[i]))
    return base


def conventional_augmentation(x: torch.Tensor) -> torch.Tensor:
    """Weak in-plane transforms on a declared unit voxel grid; no knee flips."""
    import torchio as tio
    image = tio.ScalarImage(tensor=((x.cpu()+1)/2).clamp(0, 1))
    transform = tio.Compose([
        tio.RandomAffine(scales=(.95, 1.05, .95, 1.05, 1., 1.), degrees=(0, 0, 5), translation=(2, 2, 0),
                         image_interpolation='linear', default_pad_value=0., p=.5),
        tio.RandomBiasField(coefficients=.1, order=2, p=.3),
        tio.RandomNoise(mean=0., std=(0., .02), p=.25)])
    return transform(image).data.clamp(0, 1)*2-1


def _dataset(root, split, shape):
    data = TrainingDataset(root, split, shape, require_labels=True, strict_shape=True)
    pairs = []
    for i in range(len(data)):
        x, y = data[i]; pairs.append((x, y))
    if not pairs:
        raise ValueError('Downstream cohort is empty')
    return data, pairs


def _probabilities(model, pairs, device):
    model.eval()
    with torch.no_grad():
        return np.stack([model(x[None].to(device)).sigmoid()[0].cpu().numpy() for x, _ in pairs])


def bootstrap_metrics(labels, probabilities, thresholds, groups, seed=42, resamples=500):
    if resamples < 100:
        raise ValueError('Use at least 100 downstream bootstrap resamples')
    identities = np.asarray(groups)
    clusters = [np.flatnonzero(identities == value) for value in np.unique(identities)]
    draws: dict[str, dict[str, list[float]]] = {task: {key: [] for key in ('auc', 'auprc', 'sensitivity', 'specificity', 'balanced_accuracy', 'brier')}
                                            for task in TASKS}
    rng = np.random.default_rng(seed)
    for _ in range(resamples):
        indices = np.concatenate([clusters[i] for i in rng.integers(len(clusters), size=len(clusters))])
        measured = downstream_metrics(np.asarray(labels)[indices], np.asarray(probabilities)[indices], thresholds)
        for task in TASKS:
            for key in draws[task]:
                if measured[task][key] is not None:
                    draws[task][key].append(measured[task][key])
    return {task: {key: dict(lower=float(np.quantile(values, .025)) if values else None,
                            upper=float(np.quantile(values, .975)) if values else None,
                            defined_resamples=len(values), clusters=len(clusters)) for key, values in keys.items()}
            for task, keys in draws.items()}


def compare_downstream(root, output, *, arms=('real_only', 'torchio', 'resample'), generated=None,
                       development_root=None, evaluation_root=None, evaluation_split='test', shape=(128, 128, 32),
                       steps=2000, batch_size=2, seeds=(42, 43, 44, 45, 46), device='auto',
                       generation_gate=None, smoke_test=False, bootstrap_samples=500):
    if (steps < 1 or batch_size < 1 or not seeds or len(set(seeds)) != len(seeds)
            or any(arm not in ('real_only', 'torchio', 'resample', 'synthetic', 'torchio_synthetic') for arm in arms)):
        raise ValueError('Declare valid arms, distinct seeds and positive matched budgets')
    torch.set_num_threads(4); device = device_for(device)
    training = verify_cohort(root)
    train, real = _dataset(root, 'train', shape)
    development_root = development_root or root
    dev_snapshot = verify_cohort(development_root)
    dev, development = _dataset(development_root, 'validation', shape)
    audit = audit_overlap([r for r in training.records() if r['split'] == 'train'],
                          [r for r in dev_snapshot.records() if r['split'] == 'validation'])
    if audit['overlap_counts']:
        raise ValueError('Classifier development overlaps its fitting cohort')
    if evaluation_root is None:
        evaluation, eval_data = development, dev
        role = 'development-only; thresholds fitted on these cases, no final-test performance claim'
    else:
        evaluation_snapshot = verify_cohort(evaluation_root)
        seen = [r for r in training.records() if r['split'] in ('train', 'validation')]
        seen += [r for r in dev_snapshot.records() if r['split'] == 'validation']
        overlap = audit_overlap(seen, [r for r in evaluation_snapshot.records() if r['split'] == evaluation_split])
        if overlap['overlap_counts']:
            raise ValueError('Final evaluation overlaps fitting/development identities or contents')
        eval_data, evaluation = _dataset(evaluation_root, evaluation_split, shape)
        role = 'separate final evaluation; thresholds frozen from development'
    labels = np.stack([y.numpy() for _, y in real])
    conditions = sorted({tuple(int(v) for v in y) for y in labels})
    real_by_condition = {condition: np.flatnonzero(np.all(labels == condition, axis=1)) for condition in conditions}
    synthetic: list[tuple[torch.Tensor, torch.Tensor]] = []
    synthetic_by_condition: dict[tuple[int, ...], list[int]] = {}
    manifest = None
    if any('synthetic' in arm for arm in arms):
        if generated is None:
            raise ValueError('Synthetic comparison requires generated volumes')
        generated = Path(generated)
        manifest = json.loads((generated/'manifest.json').read_text())
        if manifest['dataset_fingerprint'] != training.fingerprint or not set(manifest['generator_train_ids']).issubset(train.ids):
            raise ValueError('Generator must be fitted exclusively within the classifier fitting cohort')
        if not smoke_test:
            if manifest.get('smoke_test', True) or generation_gate is None:
                raise ValueError('Serious synthetic comparison requires a passed generation quality gate')
            gate = json.loads(Path(generation_gate).read_text())
            if (gate.get('basic_realism_pass') is not True or gate.get('diffusion_sha256') != manifest['diffusion_sha256']
                    or gate.get('autoencoder_sha256') != manifest['vq_sha256']):
                raise ValueError('Generation quality gate is missing, failed, or belongs to different models')
        evaluation_context(development_root, manifest, 'validation', training_root=root)
        if evaluation_root is not None:
            evaluation_context(evaluation_root, manifest, evaluation_split, training_root=root)
        for row in manifest['rows']:
            requested = row.get('conditioning_labels')
            if not isinstance(requested, list) or len(requested) != 3 or any(v not in (0, 1) for v in requested):
                raise ValueError('Synthetic data require recorded binary requested conditions')
            condition = tuple(requested)
            if condition not in conditions:
                raise ValueError('Synthetic condition has no support in fitting data')
            x = torch.from_numpy(generated_volume(generated, row, shape).copy())
            synthetic_by_condition.setdefault(condition, []).append(len(synthetic))
            synthetic.append((x, torch.tensor(condition, dtype=torch.float32)))
        if set(synthetic_by_condition) != set(conditions):
            raise ValueError('Fair synthetic comparison needs coverage of every observed joint-label stratum')
    out = fresh_output(output); results = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        real_indices = rng.integers(len(real), size=(steps, batch_size))
        joint_indices = rng.integers(len(conditions), size=(steps, batch_size))
        for arm in arms:
            set_seed(seed)
            model = MRIClassifier3D().to(device)
            initial = hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            draw_rng = np.random.default_rng(seed+10000)
            model.train()
            for step in range(steps):
                batch = []
                for j in range(batch_size):
                    if arm in ('resample', 'synthetic', 'torchio_synthetic') and step % 2:
                        condition = conditions[int(joint_indices[step, j])]
                        if 'synthetic' in arm:
                            index = int(draw_rng.choice(synthetic_by_condition[condition])); x, y = synthetic[index]
                        else:
                            index = int(draw_rng.choice(real_by_condition[condition])); x, y = real[index]
                    else:
                        x, y = real[int(real_indices[step, j])]
                    if arm in ('torchio', 'torchio_synthetic'):
                        x = conventional_augmentation(x)
                    batch.append((x, y))
                x = torch.stack([r[0] for r in batch]).to(device); y = torch.stack([r[1] for r in batch]).to(device)
                loss = F.binary_cross_entropy_with_logits(model(x), y)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite downstream objective')
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            dev_probabilities = _probabilities(model, development, device)
            dev_truth = np.stack([y.numpy() for _, y in development])
            thresholds = select_thresholds(dev_truth, dev_probabilities)
            probabilities = _probabilities(model, evaluation, device)
            truth = np.stack([y.numpy() for _, y in evaluation])
            name = f'{arm}-{seed}'
            np.savez_compressed(out/f'{name}-predictions.npz', truth=truth, probabilities=probabilities,
                                development_truth=dev_truth, development_probabilities=dev_probabilities)
            torch.save(dict(model=model.state_dict(), seed=seed, arm=arm, thresholds=thresholds, train_ids=train.ids), out/f'{name}.pt')
            results.append(dict(arm=arm, seed=seed, initial_parameter_sha256=initial, thresholds=thresholds,
                metrics=downstream_metrics(truth, probabilities, thresholds),
                confidence_intervals=bootstrap_metrics(truth, probabilities, thresholds,
                    [r.get('patient_id') or r['scan_id'] for r in eval_data.rows], seed, bootstrap_samples),
                predictions_sha256=sha256_file(out/f'{name}-predictions.npz')))
    aggregate = {arm: {task: {key: dict(mean=float(np.mean(values)), seed_sd=float(np.std(values, ddof=1)) if len(values) > 1 else None)
                 for key in ('auc', 'auprc', 'sensitivity', 'specificity', 'balanced_accuracy', 'brier')
                 if (values := [r['metrics'][task][key] for r in results if r['arm'] == arm and r['metrics'][task][key] is not None])}
                 for task in TASKS} for arm in arms}
    report = dict(results=results, aggregate=aggregate, steps_per_arm=steps, batch_size=batch_size, seeds=list(seeds),
        train_count=len(real), development_count=len(development), evaluation_count=len(evaluation), evaluation_role=role,
        threshold_protocol='Development Youden index; ties choose higher threshold; .5 only if one class absent',
        condition_policy='Odd replacement updates use uniform observed joint strata; resampling and synthetic share label draws',
        augmentation_protocol='TorchIO 1.2 unit-grid in-plane affine plus weak positive-intensity bias/noise; no flips',
        separate_final_evaluation=evaluation_root is not None,
        independent_test=evaluation_root is not None and overlap['patient_disjoint_verified'],
        patient_linkage_available=all(bool(r.get('patient_id')) for r in eval_data.rows),
        smoke_test=smoke_test, dataset_fingerprint=training.fingerprint,
        generated_manifest_sha256=sha256_file(Path(generated)/'manifest.json') if generated else None,
        label_caveat='Synthetic labels are requested conditions, not expert-confirmed pathology')
    write_json(out/'report.json', report)
    return report
